"""E04 part hypersim (surfaces), step 2 of 2: truth from the .hdf5 files (needs h5py and numpy; run where h5py is
installed, e.g. the rented pod). Reads built_e04/hypersim_selection.json (step 1, hypersim.py) and Apple's scene zips by
HTTP range; writes built_e04/hypersim.json (SCHEMA rows) and built_e04/hypersim_truth_log.json. No images are touched.

Per row, in the selection's order, until 600 rows pass:
- mask: semantic.hdf5 == the NYU40 id and semantic_instance.hdf5 == the class's single instance id (exactly one id >= 0
  inside the class region, else the row is dropped); its pixel count must equal step 1's count from the preview PNGs;
  it must cover >= 2% of the image; at least 80% of its pixels must have a finite normal.
- normal_cam (SCHEMA: x right, y down, z forward): Hypersim's camera frame is x right, y up, z back, so (x, -y, -z).
  Main surface = the largest group of the object's pixels within 10 degrees of its most common normal: the most common
  normal is the sampled pixel normal (2,000 pixels, seed 2604) with the most sampled pixels within 10 degrees; the
  group is every finite mask pixel within 10 degrees of it; the truth is the normalised mean of the group's normals.
- surface_class from normal_world (mean over the same group, normalised) and the view ray (camera position minus the
  group's mean position.hdf5, asset coordinates): "up" if within 30 degrees of world up (+z, gravity), else "towards"
  if within 30 degrees of the direction to the camera, else "sideways".
- dims_m: the object-aligned box extents of that instance x meters_per_asset_unit, sorted descending.
Then split 60/40 by scene (seed 2604, _common.split_by_group).

Run: python -I built_e04/code/hypersim_truth.py            (self-test of the geometry without h5py: --selftest)
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, SEED, RemoteZip, split_by_group  # noqa: E402

PART = 'hypersim'
URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip'
N_ROWS, MIN_SHARE, MIN_FINITE = 600, 0.02, 0.8
COS10, COS30 = np.cos(np.radians(10)), np.cos(np.radians(30))


def main_surface(n, rng):
    """n: (N, 3) unit normals of the mask's finite pixels -> (group boolean mask, mean normal of the group)."""
    s = n[rng.choice(len(n), min(2000, len(n)), replace=False)]
    votes = ((s @ s.T) >= COS10).sum(1)
    mode = s[int(np.argmax(votes))]
    g = (n @ mode) >= COS10
    m = n[g].mean(0)
    return g, m / np.linalg.norm(m)


def surface_class(nw, to_cam):
    if nw @ np.array([0.0, 0.0, 1.0]) >= COS30:
        return 'up'
    if nw @ (to_cam / np.linalg.norm(to_cam)) >= COS30:
        return 'towards'
    return 'sideways'


def truth(row, z, cache):
    import h5py
    rd = lambda k: h5py.File(io.BytesIO(z.read(row['members'][k])), 'r')['dataset'][()]
    o, f = row['objects'][0], row['source']['frame']
    sem, ins = rd('semantic'), rd('semantic_instance')
    H, W = sem.shape
    ids = [int(k) for k in np.unique(ins[sem == o['nyu40_id']]) if k >= 0]
    if len(ids) != 1:
        return None, f'instances of the class: {len(ids)}'
    m = (sem == o['nyu40_id']) & (ins == ids[0])
    if int(m.sum()) != o['mask_px']:
        return None, f"mask {int(m.sum())} px != preview {o['mask_px']} px"
    if m.mean() < MIN_SHARE:
        return None, 'under 2% of the image'
    nc = rd('normal_cam').astype(np.float64)[m]
    nw = rd('normal_world').astype(np.float64)[m]
    pos = rd('position').astype(np.float64)[m]
    ok = np.isfinite(nc).all(1) & np.isfinite(nw).all(1) & np.isfinite(pos).all(1)
    if ok.mean() < MIN_FINITE:
        return None, f'finite normals {ok.mean():.2f}'
    nc, nw, pos = nc[ok], nw[ok], pos[ok]
    nc /= np.linalg.norm(nc, axis=1, keepdims=True)
    nw /= np.linalg.norm(nw, axis=1, keepdims=True)
    g, n_cam_h = main_surface(nc, np.random.default_rng(SEED))
    n_w = nw[g].mean(0)
    n_w /= np.linalg.norm(n_w)
    key = row['group']
    if key not in cache:
        cache.clear()
        cache[key] = {'cam': rd('camera_positions'), 'ext': rd('bbox_extents')}
    cam = cache[key]['cam'][f]
    to_cam = cam - pos[g].mean(0)
    ext = cache[key]['ext'][ids[0]]
    if not np.isfinite(ext).all():
        return None, 'box extents not finite'
    dims = sorted((ext * row['meters_per_asset_unit']).tolist(), reverse=True)
    ys, xs = np.nonzero(m)
    out = {'name': o['name'], 'box': [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H],
           'normal_cam': [float(n_cam_h[0]), float(-n_cam_h[1]), float(-n_cam_h[2])],
           'surface_class': surface_class(n_w, to_cam), 'dims_m': [round(float(d), 4) for d in dims],
           'normal_world': [float(v) for v in n_w], 'main_group_share': round(float(g.mean()), 4),
           'area_frac': round(float(m.mean()), 5), 'nyu40_id': o['nyu40_id'], 'instance_id': ids[0]}
    return out, None


def main():
    sel = json.loads((OUT/f'{PART}_selection.json').read_text())
    rows, log, zips, cache = [], [], {}, {}
    for row in sorted(sel, key=lambda r: r['order']):
        if len(rows) >= N_ROWS:
            break
        scene = row['group']
        if scene not in zips:
            zips.clear()
            zips[scene] = RemoteZip(URL.format(scene), 1 << 18)
        try:
            obj, why = truth(row, zips[scene], cache)
        except Exception as e:                                   # a damaged member is a dropped row, logged
            obj, why = None, f'error {e!r}'[:200]
        log.append({'id': row['id'], 'kept': obj is not None, 'why': why})
        if obj is None:
            print(row['id'], 'dropped:', why, flush=True)
            continue
        rows.append({k: row[k] for k in ('id', 'measure', 'domain', 'image', 'size', 'group', 'source')} |
                    {'source': row['source'] | {'meters_per_asset_unit': row['meters_per_asset_unit']}, 'objects': [obj]})
        if len(rows) % 50 == 0:
            print('rows', len(rows), flush=True)
    split_by_group(rows)
    (OUT/f'{PART}.json').write_text(json.dumps(rows, indent=0))
    (OUT/f'{PART}_truth_log.json').write_text(json.dumps(log, indent=0))
    print('rows', len(rows), 'dropped', sum(not l['kept'] for l in log), 'scenes', len({r['group'] for r in rows}))
    print('classes', Counter(r['objects'][0]['name'] for r in rows))
    print('surface_class', Counter(r['objects'][0]['surface_class'] for r in rows))
    print('split rows', Counter(r['split'] for r in rows), 'scenes', Counter(r['split'] for r in {r['group']: r for r in rows}.values()))


def selftest():
    rng = np.random.default_rng(0)
    top = np.tile([0.0, 0.0, 1.0], (700, 1)) + rng.normal(0, 0.02, (700, 3))
    side = np.tile([1.0, 0.0, 0.0], (300, 1)) + rng.normal(0, 0.02, (300, 3))
    n = np.vstack([top, side])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    g, m = main_surface(n, np.random.default_rng(SEED))
    assert g[:700].mean() > 0.99 and g[700:].mean() < 0.01 and m[2] > 0.999, (g.mean(), m)
    assert surface_class(np.array([0, 0, 1.0]), np.array([1.0, 0, 0])) == 'up'
    assert surface_class(np.array([1.0, 0, 0]), np.array([2.0, 0.1, 0.5])) == 'towards'
    assert surface_class(np.array([0, 1.0, 0]), np.array([2.0, 0.1, 0.5])) == 'sideways'
    assert surface_class(np.array([0, 0, -1.0]), np.array([0, 0, -1.0])) == 'towards'
    print('selftest ok: main group share', round(float(g.mean()), 3), 'normal', m.round(3))


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
