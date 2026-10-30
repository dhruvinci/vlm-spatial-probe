"""E04 data (experiments/E04-more-measures/PREREG.md): the measures E01 did not cover. Output format:
`built_e04/SCHEMA.md` (one row per image or clip; truth per object). Parts built here:
  kitti_obj   orientation: 1,000 KITTI object training images not used in E01-E03, one object each, 125 per 45-degree bin
              of the observation angle; facing from rotation_y and the object's position; dimensions
  h4d         orientation: 600 frames of Harmony4D's calm sequences, cameras 01/06/11/16, frames not in E01/E03, both
              people; facing from the SMPL fit's global orientation; elbow and knee flexion
  h4d_motion  motion: 100 clips of 4 frames (about 0.6 s at 20 frames per second), one person each; pelvis ground speed
              and heading
  nyu         surfaces: 400 NYU labelled images not in E01, one object with a planar main surface (RANSAC on raw depth,
              >= 60% inliers, within 10 degrees of horizontal or vertical against a fitted floor)
  coco        counts: 1,000 COCO val2017 images not in E01, one class each (no crowd region), 250 per count band
  finish      merges every part (with the parts built by `built_e04/code/`: kitti_trk, hypersim, objectron) into
              data_e04/dataset.json and checks the format
Seed 2604; images at most 640 px (save_image).

  ENGAGE3D_SRC=.../engage3d/src python src/build_e04.py PART --src SRC --out BUILT_E04 [--cache CACHE --cache-e03 CACHE_E03]
"""
import argparse
import io
import json
import math
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import HTTPRangeFile, KITTI_CLASSES, NYU_K, SEED, save_image   # noqa: E402

KITTI_IMG = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
LIC = {'kitti': ('CC BY-NC-SA 3.0', 'https://creativecommons.org/licenses/by-nc-sa/3.0/'),
       'h4d': ('MIT (dataset card)', 'https://huggingface.co/datasets/Jyun-Ting/Harmony4D'),
       'nyu': ('no licence stated by the authors; research use', 'https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html'),
       'coco': ('annotations CC BY 4.0; images under their Flickr licences', 'https://cocodataset.org/#termsofuse')}
CALM = ['01_hugging', '05_sword2', '08_ballroom2', '09_karate']
CAMS = ['cam01', 'cam06', 'cam11', 'cam16']
SMPL = dict(pelvis=0, l_hip=1, r_hip=2, l_knee=4, r_knee=5, l_ankle=7, r_ankle=8, l_shoulder=16, r_shoulder=17,
            l_elbow=18, r_elbow=19, l_wrist=20, r_wrist=21)


def facing_deg(f, pos, up):
    """SCHEMA convention: 0 = the front points at the camera, 90 = towards the image's right, 180 = away, 270 = left."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    v = -np.asarray(pos, float)
    f = np.asarray(f, float) - (np.dot(f, up)) * up
    v = v - np.dot(v, up) * up
    f /= np.linalg.norm(f); v /= np.linalg.norm(v)
    return float(math.degrees(math.atan2(np.dot(np.cross(v, f), up), np.dot(v, f))) % 360)


def joint_angle(a, b, c):
    u, w = np.asarray(a) - b, np.asarray(c) - b
    return float(math.degrees(math.acos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1))))


def split_groups(rows, seed=SEED, test=0.4):
    groups = sorted({r['group'] for r in rows})
    random.Random(seed).shuffle(groups)
    te = set(groups[:round(test * len(groups))])
    for r in rows:
        r['split'] = 'test' if r['group'] in te else 'train'
    return rows


def used_ids(cache, cache_e03):
    ids = set()
    for p in (Path(cache)/'data/dataset.json', Path(cache_e03)/'data/dataset.json'):
        if p.exists():
            ids |= {r['id'] for r in json.loads(p.read_text())}
    return ids


# ------------------------------------------------------------------ KITTI object: orientation
def build_kitti_obj(src, out, used):
    labels = zipfile.ZipFile(Path(src)/'kitti/labels.zip')
    cands = {b: [] for b in range(8)}
    for n in sorted(x for x in labels.namelist() if x.endswith('.txt')):
        stem = Path(n).stem
        if f'kitti_{stem}' in used:
            continue
        lines = [l.split() for l in labels.read(n).decode().strip().splitlines()]
        classes = [f[0] for f in lines]
        best = None
        for f in lines:
            if f[0] not in KITTI_CLASSES or f[0] == 'Tram' or classes.count(f[0]) != 1:
                continue
            trunc, occ, alpha = float(f[1]), int(f[2]), float(f[3])
            x1, y1, x2, y2 = map(float, f[4:8]); h, w, l = map(float, f[8:11]); x, y, z = map(float, f[11:14]); ry = float(f[14])
            if occ > 1 or trunc > 0.3 or y2 - y1 < 25 or z <= 2:
                continue
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best[0]:
                best = (area, f[0], alpha, [x1, y1, x2, y2], [h, w, l], [x, y - h / 2, z], ry)
        if best:
            b = int(((best[2] + math.pi) % (2 * math.pi)) // (math.pi / 4))
            cands[b].append((stem, best))
    rng = random.Random(SEED)
    picked, short = [], {}
    for b in range(8):
        rng.shuffle(cands[b]); picked += cands[b][:125]
        short[b] = max(0, 125 - len(cands[b]))
    if sum(short.values()):                                      # fill a short bin from the largest bins (reported)
        rest = sorted((c for b in range(8) for c in cands[b][125:]), key=lambda c: c[0])
        rng.shuffle(rest); picked += rest[:sum(short.values())]
    zimg = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(KITTI_IMG), 1 << 20))
    members = {Path(m).stem: m for m in zimg.namelist() if m.startswith('training/image_2/') and m.endswith('.png')}
    rows = []
    for stem, (_, cls, alpha, box, dims, cen, ry) in picked:
        img = Image.open(io.BytesIO(zimg.read(members[stem])))
        W, H = img.size
        p = out/'images'/'kitti_obj'/f'{stem}.jpg'
        save_image(img, p)
        f = np.array([math.cos(ry), 0.0, -math.sin(ry)])
        rows.append({'id': f'kitti_obj_{stem}', 'measure': 'orientation', 'domain': 'kitti_obj', 'image': str(p.relative_to(out)),
                     'size': list(Image.open(p).size), 'group': stem,
                     'source': {'dataset': 'KITTI object (training)', 'frame': stem, 'licence': LIC['kitti'][0], 'licence_url': LIC['kitti'][1]},
                     'objects': [{'name': KITTI_CLASSES[cls], 'box': [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                                  'facing_deg': facing_deg(f, cen, (0, -1, 0)), 'alpha_rad': alpha, 'rotation_y_rad': ry,
                                  'dims_m': dims}]})
        if len(rows) % 100 == 0:
            print('kitti_obj', len(rows), flush=True)
    split_groups(rows)
    (out/'kitti_obj.json').write_text(json.dumps(rows))
    print('kitti_obj', len(rows), 'images; short bins filled', short)


# ------------------------------------------------------------------ Harmony4D: orientation and motion
def h4d_people(clip, f, tocam, up_cam):
    """Both people (left first): box (fractions of the original frame), pelvis (camera), facing, joint angles."""
    gt = clip._smpl(f)
    boxes = clip.truth_boxes(f)
    W, H = clip.size
    out = []
    for k, person in enumerate(sorted(gt)):
        go = np.asarray(gt[person]['global_orient'], float).ravel()[:3]
        th = np.linalg.norm(go)
        Rg = np.eye(3) if th < 1e-9 else _rodrigues(go)
        J = tocam(np.asarray(gt[person]['joints'][:24], float))
        fwd = tocam(np.stack([np.zeros(3), Rg @ np.array([0.0, 0.0, 1.0])]))
        f_cam = fwd[1] - fwd[0]
        x1, y1, x2, y2 = boxes[k]
        out.append({'box': [x1 / W, y1 / H, x2 / W, y2 / H], 'cx': (x1 + x2) / 2, 'pelvis': J[SMPL['pelvis']],
                    'facing_deg': facing_deg(f_cam, J[SMPL['pelvis']], up_cam),
                    'elbow_l_deg': joint_angle(J[SMPL['l_shoulder']], J[SMPL['l_elbow']], J[SMPL['l_wrist']]),
                    'elbow_r_deg': joint_angle(J[SMPL['r_shoulder']], J[SMPL['r_elbow']], J[SMPL['r_wrist']]),
                    'knee_l_deg': joint_angle(J[SMPL['l_hip']], J[SMPL['l_knee']], J[SMPL['l_ankle']]),
                    'knee_r_deg': joint_angle(J[SMPL['r_hip']], J[SMPL['r_knee']], J[SMPL['r_ankle']])})
    return out


def _rodrigues(r):
    th = np.linalg.norm(r); k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def h4d_sequences(root):
    return [(act, s.name) for act in CALM for s in sorted((root/act).iterdir()) if any((s/'exo'/c).exists() for c in CAMS)]


def build_h4d(src, out, used, motion=False):
    sys.path.insert(0, os.environ['ENGAGE3D_SRC'])
    import engage_demo as DM
    from engage_clip import H4DClip
    from engage_paths import H4D
    seqs = h4d_sequences(Path(H4D))
    rng = random.Random(SEED + (1 if motion else 0))
    per = (100 if motion else 600) // len(seqs) + 1
    rows = []
    for act, s in seqs:
        for cam in CAMS:
            try:
                clip = H4DClip(s, cam)
            except Exception:
                continue
            _, _, tocam = DM.cam_of(clip.seq, cam)
            up_cam = tocam(np.array([[0, 0, 0], [0, 0, 1.0]]))
            up_cam = up_cam[1] - up_cam[0]
            fr = clip.frames
            n_cam = per // len(CAMS) + 1
            if motion:
                step = max(1, round(4 / max(1, np.median(np.diff(fr)))))                   # about 4 raw frames per step
                starts = list(range(0, len(fr) - 3 * step))
                rng.shuffle(starts)
                for i0 in starts[:n_cam * 3]:
                    if sum(1 for r in rows if r['group'] == s and r['source']['camera'] == cam) >= n_cam:
                        break
                    fs = [fr[i0 + j * step] for j in range(4)]
                    try:
                        P = [h4d_people(clip, f, tocam, up_cam) for f in fs]
                    except Exception:
                        continue
                    order0 = np.argsort([p['cx'] for p in P[0]]); order3 = np.argsort([p['cx'] for p in P[3]])
                    if list(order0) != list(order3):                                       # sides swapped: skip
                        continue
                    side = len(rows) % 2
                    k = order0[side]
                    dt = (fs[-1] - fs[0]) / 20.0
                    d = P[3][k]['pelvis'] - P[0][k]['pelvis']; d = d - (d @ up_cam) * up_cam / (up_cam @ up_cam)
                    speed = float(np.linalg.norm(d) / dt)
                    heading = facing_deg(d, P[0][k]['pelvis'], up_cam) if speed > 1e-3 else None
                    paths = []
                    for j, f in enumerate(fs):
                        p = out/'images'/'h4d_motion'/f'{s}_{cam}_{fs[0]:05d}_f{j}.jpg'
                        save_image(Image.open(clip.image_path(f)), p); paths.append(str(p.relative_to(out)))
                    rows.append({'id': f'h4d_motion_{s}_{cam}_{fs[0]}', 'measure': 'motion', 'domain': 'h4d_motion', 'frames': paths,
                                 'size': list(Image.open(out/paths[-1]).size), 'group': s,
                                 'source': {'dataset': 'Harmony4D', 'sequence': s, 'camera': cam, 'frames': fs, 'fps_assumed': 20,
                                            'licence': LIC['h4d'][0], 'licence_url': LIC['h4d'][1]},
                                 'objects': [{'name': f"person on the {'left' if side == 0 else 'right'}", 'box': P[3][k]['box'],
                                              'speed_mps': speed, 'heading_deg': heading, 'moving': speed >= 0.5, 'dt_s': dt}]})
            else:
                cand = [f for f in fr if f'h4d_{s}_{f}' not in used or cam != 'cam01']
                rng.shuffle(cand)
                for f in cand:                                            # every valid frame, up to 60 per camera
                    if sum(1 for r in rows if r['group'] == s and r['source']['camera'] == cam) >= 60:
                        break
                    try:
                        P = h4d_people(clip, f, tocam, up_cam)
                    except Exception:
                        continue
                    if abs(P[0]['cx'] - P[1]['cx']) < 0.05 * clip.size[0]:
                        continue
                    P = sorted(P, key=lambda p: p['cx'])
                    p = out/'images'/'h4d'/f'{s}_{cam}_{f:05d}.jpg'
                    save_image(Image.open(clip.image_path(f)), p)
                    objs = [{'name': f'person on the {side}', 'box': q['box'], 'facing_deg': q['facing_deg'],
                             **{k_: q[k_] for k_ in ('elbow_l_deg', 'elbow_r_deg', 'knee_l_deg', 'knee_r_deg')}}
                            for side, q in zip(('left', 'right'), P)]
                    rows.append({'id': f'h4d_{s}_{cam}_{f}', 'measure': 'orientation', 'domain': 'h4d', 'image': str(p.relative_to(out)),
                                 'size': list(Image.open(p).size), 'group': s,
                                 'source': {'dataset': 'Harmony4D', 'sequence': s, 'camera': cam, 'frame': int(f),
                                            'licence': LIC['h4d'][0], 'licence_url': LIC['h4d'][1]}, 'objects': objs})
        print(act, s, len(rows), flush=True)
    random.Random(SEED + 2).shuffle(rows)                                                  # no sequence order bias
    if not motion:                                                                         # 600, round-robin over sequences
        by = {}
        for r in rows:
            by.setdefault(r['group'], []).append(r)
        rows, k = [], 0
        while len(rows) < 600 and any(k < len(v) for v in by.values()):
            rows += [v[k] for v in by.values() if k < len(v)][:600 - len(rows)]; k += 1
    rows = rows[:100 if motion else 600]
    split_groups(rows)
    name = 'h4d_motion' if motion else 'h4d'
    (out/f'{name}.json').write_text(json.dumps(rows))
    print(name, len(rows), 'rows')


# ------------------------------------------------------------------ NYU: surfaces
NYU_SURF = {'table', 'desk', 'counter', 'bed', 'dresser', 'night stand', 'cabinet', 'bookshelf', 'shelves', 'whiteboard',
            'door', 'picture', 'mirror', 'television', 'refrigerator', 'sofa', 'box', 'blinds', 'window'}


def ransac_plane(X, thr, rng, iters=300):
    best, inl = 0, None
    for _ in range(iters):
        s = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        m = np.abs((X - s[0]) @ n) < thr
        if m.sum() > best:
            best, inl = int(m.sum()), m
    Q = X[inl]
    n = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][2]
    return n, Q.mean(0), best / len(X)


def build_nyu(src, out, used):
    import h5py
    f = h5py.File(Path(src)/'nyu/nyu_depth_v2_labeled.mat', 'r')
    names = [''.join(chr(c) for c in f[r][()].ravel()) for r in f['names'][0]]
    scenes = [''.join(chr(c) for c in f[r][()].ravel()) for r in f['scenes'][0]]
    fx, fy, cx, cy = NYU_K['fx'], NYU_K['fy'], NYU_K['cx'], NYU_K['cy']
    order = [i for i in range(f['images'].shape[0]) if f'nyu_{i}' not in used]
    random.Random(SEED).shuffle(order)
    rng = np.random.default_rng(SEED)
    floor_id = names.index('floor') + 1
    rows, gate = [], {'no_floor': 0, 'no_planar_object': 0}
    for i in order:
        if len(rows) >= 400:
            break
        lab = f['labels'][i].T; ins = f['instances'][i].T; raw = f['rawDepths'][i].T.astype(np.float64)
        H, W = lab.shape
        vv, uu = np.mgrid[0:H, 0:W]

        def pts(m):
            z = raw[m]; ok = z > 0
            return np.stack([(uu[m][ok] - cx) * z[ok] / fx, (vv[m][ok] - cy) * z[ok] / fy, z[ok]], 1)
        F = pts(lab == floor_id)
        if len(F) < 500:
            gate['no_floor'] += 1; continue
        up, _, share = ransac_plane(F[::max(1, len(F) // 5000)], 0.02, rng)
        if up[1] > 0:
            up = -up                                                                   # camera y points down
        best = None
        for c in np.unique(lab):
            if c == 0 or names[c - 1] not in NYU_SURF:
                continue
            ids = [k for k in np.unique(ins[lab == c]) if k > 0]
            if len(ids) != 1:
                continue
            m = (lab == c) & (ins == ids[0])
            if m.sum() < 0.02 * H * W:
                continue
            X = pts(m)
            if len(X) < 300:
                continue
            n, p0, share = ransac_plane(X[::max(1, len(X) // 5000)], 0.015, rng)
            if share < 0.6:
                continue
            ang_up = math.degrees(math.acos(min(1, abs(n @ up))))
            if not (ang_up <= 10 or ang_up >= 80):
                continue
            if ang_up <= 10:
                n = n if n @ up > 0 else -n
            else:
                n = n if n @ (-p0) > 0 else -n
            vcam = -p0 / np.linalg.norm(p0)
            cls = 'up' if math.degrees(math.acos(np.clip(n @ up, -1, 1))) <= 30 else (
                'towards' if math.degrees(math.acos(np.clip(n @ vcam, -1, 1))) <= 30 else 'sideways')
            area = int(m.sum())
            if best is None or area > best[0]:
                ys, xs = np.nonzero(m)
                best = (area, names[c - 1], n, cls, [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H], share)
        if best is None:
            gate['no_planar_object'] += 1; continue
        img = Image.fromarray(np.transpose(f['images'][i], (2, 1, 0)))
        p = out/'images'/'nyu'/f'nyu_{i:04d}.jpg'
        save_image(img, p)
        rows.append({'id': f'nyu_e04_{i}', 'measure': 'surfaces', 'domain': 'nyu', 'image': str(p.relative_to(out)),
                     'size': list(Image.open(p).size), 'group': scenes[i],
                     'source': {'dataset': 'NYU Depth v2 (labelled)', 'index': i, 'licence': LIC['nyu'][0], 'licence_url': LIC['nyu'][1]},
                     'objects': [{'name': best[1], 'box': best[4], 'normal_cam': best[2].tolist(), 'surface_class': best[3],
                                  'plane_inlier_share': best[5], 'dims_m': None}]})
    split_groups(rows)
    (out/'nyu.json').write_text(json.dumps(rows))
    from collections import Counter
    print('nyu', len(rows), 'images; classes', Counter(r['objects'][0]['surface_class'] for r in rows), 'skipped', gate)


# ------------------------------------------------------------------ COCO val2017: counts
PLURAL = {'person': 'people', 'sheep': 'sheep', 'skis': 'pairs of skis', 'mouse': 'computer mice', 'knife': 'knives',
          'bus': 'buses', 'couch': 'couches', 'sandwich': 'sandwiches', 'wine glass': 'wine glasses', 'dining table': 'dining tables',
          'toothbrush': 'toothbrushes', 'bench': 'benches', 'giraffe': 'giraffes', 'scissors': 'pairs of scissors',
          'sports ball': 'sports balls', 'hair drier': 'hair driers', 'skateboard': 'skateboards', 'cow': 'cows'}


def plural(n):
    return PLURAL.get(n, n + ('es' if n.endswith(('s', 'sh', 'ch', 'x')) else 's'))


def build_coco(src, out, used):
    import urllib.request
    z = zipfile.ZipFile(Path(src)/'coco/annotations_trainval2017.zip')
    ann = json.loads(z.read('annotations/instances_val2017.json'))
    cats = {c['id']: c['name'] for c in ann['categories']}
    per = {}
    for a in ann['annotations']:
        d = per.setdefault((a['image_id'], cats[a['category_id']]), {'n': 0, 'ge32': 0, 'crowd': False})
        if a['iscrowd']:
            d['crowd'] = True
        else:
            d['n'] += 1; d['ge32'] += a['area'] >= 32 * 32
    imgs = {im['id']: im for im in ann['images']}
    bands = [(1, 2), (3, 5), (6, 9), (10, 13)]
    rng = random.Random(SEED)
    pairs = [(iid, c, d) for (iid, c), d in per.items() if not d['crowd'] and d['n'] >= 1 and f'coco_{iid}' not in used]
    rng.shuffle(pairs)
    chosen, taken, filled = [], set(), {}
    for lo, hi in bands:
        got = [p for p in pairs if lo <= p[2]['n'] <= hi and p[0] not in taken][:250]
        taken |= {p[0] for p in got}; chosen += got; filled[f'{lo}-{hi}'] = len(got)
    rows = []
    for iid, c, d in chosen:
        im = imgs[iid]
        p = out/'images'/'coco'/f'coco_{iid:012d}.jpg'
        if not p.exists():
            data = urllib.request.urlopen(f"http://images.cocodataset.org/val2017/{im['file_name']}", timeout=60).read()
            save_image(Image.open(io.BytesIO(data)), p)
        rows.append({'id': f'coco_e04_{iid}', 'measure': 'counts', 'domain': 'coco', 'image': str(p.relative_to(out)),
                     'size': list(Image.open(p).size), 'group': str(iid),
                     'source': {'dataset': 'COCO val2017', 'image_id': iid, 'flickr_url': im.get('flickr_url'), 'image_licence': im.get('license'),
                                'licence': LIC['coco'][0], 'licence_url': LIC['coco'][1]},
                     'objects': [{'name': plural(c), 'class': c, 'box': [0, 0, 1, 1], 'count': d['n'], 'count_ge32': int(d['ge32'])}]})
        if len(rows) % 100 == 0:
            print('coco', len(rows), flush=True)
    split_groups(rows)
    (out/'coco.json').write_text(json.dumps(rows))
    print('coco', len(rows), 'images; per band', filled)


# ------------------------------------------------------------------ merge
PARTS = ['kitti_obj', 'h4d', 'objectron', 'hypersim', 'nyu', 'kitti_trk', 'h4d_motion', 'coco']
NEED = {'orientation': ['facing_deg'], 'surfaces': ['normal_cam', 'surface_class'], 'motion': ['speed_mps', 'moving'],
        'counts': ['count']}


def finish(out):
    rows, report = [], {}
    for part in PARTS:
        p = out/f'{part}.json'
        if not p.exists():
            report[part] = 'missing'; continue
        rs = json.loads(p.read_text())
        for r in rs:
            assert r['measure'] in NEED and r['split'] in ('train', 'test') and r['objects'], r['id']
            for o in r['objects']:
                assert all(k in o for k in NEED[r['measure']] + ['name', 'box']), (r['id'], o.keys())
            for q in r.get('frames', [r.get('image')]):
                assert (out/q).exists(), q
        report[part] = {'rows': len(rs), 'test': sum(r['split'] == 'test' for r in rs), 'objects': sum(len(r['objects']) for r in rs)}
        rows += rs
    (out/'dataset.json').write_text(json.dumps(rows))
    print(json.dumps(report, indent=1))
    print('images', sum(len(r.get('frames', [1])) for r in rows), 'objects', sum(len(r['objects']) for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['kitti_obj', 'h4d', 'h4d_motion', 'nyu', 'coco', 'finish'])
    ap.add_argument('--src', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default='')
    ap.add_argument('--cache-e03', default='')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    used = used_ids(a.cache, a.cache_e03) if a.cache else set()
    if a.part == 'finish':
        return finish(out)
    {'kitti_obj': lambda: build_kitti_obj(a.src, out, used), 'h4d': lambda: build_h4d(a.src, out, used),
     'h4d_motion': lambda: build_h4d(a.src, out, used, motion=True), 'nyu': lambda: build_nyu(a.src, out, used),
     'coco': lambda: build_coco(a.src, out, used)}[a.part]()


if __name__ == '__main__':
    main()
