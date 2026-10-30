"""E04 object masks for the MoGe-2 surface reference (PREREG: "the median normal inside the object's mask"), at the
saved image's size. Each mask is rebuilt exactly as its builder chose the object, and checked against the stored box.
  nyu       the labelled .mat (needs h5py, e.g. on the rented pod): (labels == class) & (instances == its single id),
            the class named in the row (src/build_e04.py build_nyu)
  hypersim  the release's preview PNGs, read by HTTP range (no h5py): semantic colour == the NYU40 colour and instance
            colour == the selected instance (hypersim.py frame_objects; hypersim_truth.py checked these pixel counts
            against the .hdf5 masks); resized from 1024 x 768 to the saved 640 x 480 by nearest neighbour
Writes {id: packed bits} plus {id + ':shape': [H, W]} as an .npz.

  python src/e04_parts/masks.py nyu --mat nyu_depth_v2_labeled.mat --data DATA_E04 --out masks_nyu.npz
  python src/e04_parts/masks.py hypersim --data DATA_E04 --out masks_hypersim.npz
"""
import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


def box_of(m):
    H, W = m.shape
    ys, xs = np.nonzero(m)
    return [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H]


def nyu(rows, mat):
    import h5py
    f = h5py.File(mat, 'r')
    names = [''.join(chr(c) for c in f[r][()].ravel()) for r in f['names'][0]]
    out = {}
    for r in rows:
        i, o = r['source']['index'], r['objects'][0]
        lab = f['labels'][i].T; ins = f['instances'][i].T
        c = names.index(o['name']) + 1
        ids = [k for k in np.unique(ins[lab == c]) if k > 0]
        assert len(ids) == 1, r['id']
        m = (lab == c) & (ins == ids[0])
        assert list(m.shape[::-1]) == r['size'], (r['id'], m.shape, r['size'])
        assert np.allclose(box_of(m), o['box']), (r['id'], box_of(m), o['box'])
        out[r['id']] = m
    return out


def hypersim(rows, sel):
    from _common import RemoteZip
    from hypersim import NYU40_RGB, URL, code
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r['source']['scene'], []).append(r)

    def one_scene(scene):
        z = RemoteZip(URL.format(scene))
        res = {}
        for r in by_scene[scene]:
            s, o = sel[r['id']], sel[r['id']]['objects'][0]
            prev = f"{scene}/images/scene_{r['source']['camera']}_geometry_preview/frame.{r['source']['frame']:04d}.{{}}.png"
            sc = code(np.asarray(Image.open(io.BytesIO(z.read(prev.format('semantic')))).convert('RGB')))
            ic = code(np.asarray(Image.open(io.BytesIO(z.read(prev.format('semantic_instance')))).convert('RGB')))
            rgb, irgb = NYU40_RGB[o['nyu40_id']], o['png_instance_rgb']
            m = (sc == (rgb[0] << 16 | rgb[1] << 8 | rgb[2])) & (ic == (irgb[0] << 16 | irgb[1] << 8 | irgb[2]))
            assert int(m.sum()) == o['mask_px'], (r['id'], int(m.sum()), o['mask_px'])
            assert np.allclose(box_of(m), o['box']), r['id']
            W, H = r['size']
            res[r['id']] = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        print('hypersim', scene, len(res), flush=True)
        return res

    out = {}
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one_scene, sorted(by_scene)):
            out.update(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['nyu', 'hypersim'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mat')
    a = ap.parse_args()
    data = Path(a.data)
    rows = [r for r in json.loads((data/'dataset.json').read_text()) if r['domain'] == a.part]
    if a.part == 'nyu':
        masks = nyu(rows, a.mat)
    else:
        sel = {s['id']: s for s in json.loads((data/'hypersim_selection.json').read_text())}
        masks = hypersim(rows, sel)
    assert len(masks) == len(rows), (len(masks), len(rows))
    np.savez_compressed(a.out, **{k: np.packbits(m) for k, m in masks.items()},
                        **{k + ':shape': np.array(m.shape) for k, m in masks.items()})
    print('saved', a.out, len(masks), 'masks; mean area', round(float(np.mean([m.mean() for m in masks.values()])), 3))


if __name__ == '__main__':
    main()
