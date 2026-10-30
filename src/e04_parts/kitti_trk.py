"""E04 part kitti_trk (motion): 300 four-frame clips from the KITTI tracking training sequences (CC BY-NC-SA 3.0).

Pre-registration (vlm-spatial-probe experiments/E04-more-measures/PREREG.md): frames t, t+3, t+6, t+9 (10 Hz, so
dt = 0.9 s), one object per clip: 150 cars, 100 pedestrians, 50 cyclists, spread evenly over still (< 0.5 m/s),
slow (0.5-3) and fast (> 3) where supply allows (a short bin is filled from the others, evenly). Object filters in all
four frames: occluded <= 1, truncated <= 0.3 (the tracking labels give truncation as an integer 0/1/2, so this means
0), box height >= 25 px.

Truth (built_e04/SCHEMA.md):
- speed_mps: ground speed with the car's own motion removed: the label's bottom-centre point (rectified camera 0
  frame) -> IMU -> world through the calibration (R_rect, Tr_velo_cam, Tr_imu_velo) and the oxts pose of each frame
  (KITTI raw devkit convertOxtsToPose: Mercator x/y, altitude, R = Rz(yaw) Ry(pitch) Rx(roll)); speed = horizontal
  world displacement between the first and last frame / 0.9 s.
- heading_deg: that displacement rotated into the FIRST frame's camera frame, SCHEMA angle with v = unit vector from
  the object (first frame) to the camera and up = (0, -1, 0). null when the object is still (heading undefined).
- moving: speed >= 0.5 m/s. dt_s = 0.9. box: the label box in the LAST frame, as fractions of the image.
Clips of one track never overlap (starts at least 10 frames apart); within a speed bin, unused tracks are taken first.
group = sequence; split 60/40 by sequence (seed 2604). Images: the four frames, 640 px JPEG.

Run: ~/Code/kakashi-research/dataset/.venv-depth/bin/python -I built_e04/code/kitti_trk.py
"""
import io
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, SEED, RemoteZip, facing_deg, save_image, split_by_group  # noqa: E402

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'
CLASSES = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'cyclist'}
QUOTA = {'Car': 150, 'Pedestrian': 100, 'Cyclist': 50}
BINS = ('still', 'slow', 'fast')
STEP, DT = 3, 0.9
PART = 'kitti_trk'


def speed_bin(v):
    return 'still' if v < 0.5 else 'slow' if v <= 3 else 'fast'


def h4(m):
    M = np.eye(4)
    M[:m.shape[0], :m.shape[1]] = m
    return M


def calib(txt):
    cal = {}
    for line in txt.splitlines():
        if not line.strip():
            continue
        key, vals = (line.split(':', 1) if ':' in line else (line.split()[0], ' '.join(line.split()[1:])))
        cal[key.strip()] = np.array(vals.split(), float)
    return h4(cal['R_rect'].reshape(3, 3)), h4(cal['Tr_velo_cam'].reshape(3, 4)), h4(cal['Tr_imu_velo'].reshape(3, 4))


def oxts_poses(txt):
    a = np.array([[float(x) for x in l.split()] for l in txt.strip().splitlines()])
    er = 6378137.0
    scale = np.cos(a[0, 0] * np.pi / 180)
    T = []
    for lat, lon, alt, ro, pi, ya in a[:, :6]:
        t = [scale * er * lon * np.pi / 180, scale * er * np.log(np.tan((90 + lat) * np.pi / 360)), alt]
        cx, sx, cy, sy, cz, sz = np.cos(ro), np.sin(ro), np.cos(pi), np.sin(pi), np.cos(ya), np.sin(ya)
        R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
             @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, t
        T.append(M)
    return np.array(T)


def candidates(L, C, O):
    cands = []
    for sq in range(21):
        s = f'{sq:04d}'
        Rr, Tvc, Tiv = calib(C.read(f'training/calib/{s}.txt').decode())
        cam_to_imu = np.linalg.inv(Tiv) @ np.linalg.inv(Tvc) @ np.linalg.inv(Rr)
        T = oxts_poses(O.read(f'training/oxts/{s}.txt').decode())
        tracks = defaultdict(dict)
        for line in L.read(f'training/label_02/{s}.txt').decode().splitlines():
            f = line.split()
            if f[2] not in CLASSES:
                continue
            fr, tid, trunc, occ = int(f[0]), int(f[1]), float(f[3]), int(f[4])
            x1, y1, x2, y2 = map(float, f[6:10])
            ok = occ <= 1 and trunc <= 0.3 and (y2 - y1) >= 25
            tracks[(f[2], tid)][fr] = {'ok': ok, 'box': [x1, y1, x2, y2], 'X': np.array([*map(float, f[13:16]), 1.0])}
        for (cls, tid), d in sorted(tracks.items()):
            last = -10**9
            for t in sorted(d):
                fr = [t + STEP * k for k in range(4)]
                if t < last + 10 or not all(k in d and d[k]['ok'] for k in fr):
                    continue
                last = t
                if fr[-1] >= len(T):
                    raise ValueError(f'oxts shorter than labels in {s}')
                W0 = T[fr[0]] @ cam_to_imu
                P0, P3 = W0 @ d[fr[0]]['X'], T[fr[-1]] @ cam_to_imu @ d[fr[-1]]['X']
                disp = P3[:3] - P0[:3]
                speed = float(np.linalg.norm(disp[:2]) / DT)        # world z is up (oxts)
                d_cam0 = np.linalg.inv(W0)[:3, :3] @ disp
                X0 = d[fr[0]]['X'][:3]
                heading = facing_deg(d_cam0, -X0 / np.linalg.norm(X0), (0, -1, 0))
                cands.append({'seq': s, 'cls': cls, 'track': tid, 'frames': fr, 'speed': speed, 'heading': heading,
                              'bin': speed_bin(speed), 'box_last': d[fr[-1]]['box'], 'depth0': float(X0[2])})
    return cands


def select(cands):
    rng = random.Random(SEED)
    keyed = [(rng.random(), c) for c in cands]
    chosen = []
    for cls, n in QUOTA.items():
        pool = {b: [c for _, c in sorted(((k, c) for k, c in keyed if c['cls'] == cls and c['bin'] == b), key=lambda x: x[0])]
                for b in BINS}
        target = {b: n // 3 + (1 if i < n % 3 else 0) for i, b in enumerate(BINS)}
        while True:                                          # move the deficit of short bins to bins with spare supply
            short = {b: target[b] - len(pool[b]) for b in BINS if len(pool[b]) < target[b]}
            if not short:
                break
            spare = [b for b in BINS if len(pool[b]) > target[b]]
            if not spare:
                break
            for b, d in short.items():
                target[b] -= d
                for i in range(d):
                    room = [s for s in spare if len(pool[s]) > target[s]]
                    if not room:
                        break
                    target[sorted(room, key=lambda s: (target[s], target[s] - len(pool[s])))[0]] += 1   # evenly
        for b in BINS:
            used, picks = Counter(), []
            for c in pool[b]:                                 # pass 1: one clip per track
                if len(picks) < target[b] and used[(c['seq'], c['track'])] == 0:
                    picks.append(c); used[(c['seq'], c['track'])] += 1
            for c in pool[b]:                                 # pass 2: further clips of a track, if still short
                if len(picks) < target[b] and c not in picks:
                    picks.append(c)
            chosen += picks
        print(cls, 'supply', {b: len(pool[b]) for b in BINS}, 'target', target, flush=True)
    return chosen


def main():
    L = RemoteZip(BASE + 'data_tracking_label_2.zip')
    C = RemoteZip(BASE + 'data_tracking_calib.zip')
    O = RemoteZip(BASE + 'data_tracking_oxts.zip')
    cands = candidates(L, C, O)
    print('candidate clips', Counter((c['cls'], c['bin']) for c in cands), flush=True)
    chosen = select(cands)
    Z = RemoteZip(BASE + 'data_tracking_image_2.zip')
    need = sorted({(c['seq'], f) for c in chosen for f in c['frames']})
    print('clips', len(chosen), 'unique frames to fetch', len(need), flush=True)

    imgdir = OUT/'images'/PART
    imgdir.mkdir(parents=True, exist_ok=True)
    rid = lambda c: f"{PART}_{c['seq']}_{c['track']:03d}_{c['frames'][0]:06d}"
    users = defaultdict(list)
    for c in chosen:
        for k, f in enumerate(c['frames']):
            users[(c['seq'], f)].append(imgdir/f'{rid(c)}_f{k}.jpg')
    orig_size = {}

    def work(key):
        s, f = key
        paths = users[key]
        img =Image.open(io.BytesIO(Z.read(f'training/image_02/{s}/{f:06d}.png')))
        img.load()
        orig_size[key] = img.size
        sizes = [save_image(img, p) for p in paths]
        return key, sizes[0]

    saved = {}
    with ThreadPoolExecutor(6) as ex:
        for n, (key, sz) in enumerate(ex.map(work, need)):
            saved[key] = sz
            if n % 100 == 0:
                print('frames', n, '/', len(need), flush=True)

    rows = []
    for c in chosen:
        i = rid(c)
        W, H = orig_size[(c['seq'], c['frames'][-1])]
        x1, y1, x2, y2 = c['box_last']
        moving = c['speed'] >= 0.5
        rows.append({
            'id': i, 'measure': 'motion', 'domain': PART,
            'frames': [f'images/{PART}/{i}_f{k}.jpg' for k in range(4)],
            'size': saved[(c['seq'], c['frames'][-1])], 'group': c['seq'],
            'source': {'dataset': 'KITTI tracking (training)', 'sequence': c['seq'], 'track_id': c['track'],
                       'frames': c['frames'], 'class': c['cls'], 'camera': 'image_02',
                       'licence': 'CC BY-NC-SA 3.0', 'licence_url': 'https://creativecommons.org/licenses/by-nc-sa/3.0/'},
            'objects': [{'name': CLASSES[c['cls']],
                         'box': [max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H)],
                         'speed_mps': round(c['speed'], 4), 'heading_deg': round(c['heading'], 2) if moving else None,
                         'moving': moving, 'dt_s': DT, 'speed_bin': c['bin'], 'depth_first_m': round(c['depth0'], 3)}]})
    split_by_group(rows)
    rows.sort(key=lambda r: r['id'])
    (OUT/f'{PART}.json').write_text(json.dumps(rows, indent=0))
    print('rows', len(rows))
    print('class x bin', sorted(Counter((r['source']['class'], r['objects'][0]['speed_bin']) for r in rows).items()))
    print('split rows', Counter(r['split'] for r in rows), 'groups', Counter(r['split'] for r in {r['group']: r for r in rows}.values()))
    print('test sequences', sorted({r['group'] for r in rows if r['split'] == 'test'}))


if __name__ == '__main__':
    main()
