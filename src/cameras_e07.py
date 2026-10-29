"""E07 arm A3: each object's angular size from its box and its camera (experiments/E07-transfer-diagnosis/PREREG.md).

Angular size s = sqrt(w * h), with the box's width and height in normalised camera coordinates (pixels / focal length).
NYU and KITTI: the intrinsics in dataset.json (pinhole). Hypersim: the scene's projection matrix (f = M_proj_00 * W / 2,
M_proj_11 * H / 2). Harmony4D: camera 1's fisheye intrinsics and distortion per sequence (engage3d's clip loader), with
the box's edge midpoints undistorted. Writes {id: [s of each object]}.

  ENGAGE3D_SRC=.../engage3d/src python src/cameras_e07.py --cache CACHE --e03 CACHE_E03 \
      --hypersim-meta HYPERSIM_META/metadata_camera_parameters.csv --out CACHE_E03/e07_cameras.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

DOMAINS = ('indoor', 'indoor_synth', 'two_people', 'street')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--e03', required=True)
    ap.add_argument('--hypersim-meta', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = [r for r in json.loads((Path(a.cache)/'data/dataset.json').read_text())
            if r['domain'] in DOMAINS and r.get('activity') not in ('mma', 'grappling')]
    rows += json.loads((Path(a.e03)/'data/dataset.json').read_text())
    proj = {r['scene_name']: r for r in csv.DictReader(open(a.hypersim_meta))}
    h4d = {}
    out = {}
    for r in rows:
        W, H = r['size']
        B = np.array([o['box'] for o in r['objects']], float) * [W, H, W, H]
        if r['domain'] == 'two_people':
            import cv2
            if r['sequence'] not in h4d:
                sys.path.insert(0, os.environ['ENGAGE3D_SRC'])
                from engage_clip import H4DClip
                c = H4DClip(r['sequence'], 'cam01')
                assert tuple(c.size) == (W, H)
                h4d[r['sequence']] = (c.K, c.D)
            K, D = h4d[r['sequence']]
            cx, cy = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
            pts = np.stack([np.stack([B[:, 0], cy], 1), np.stack([B[:, 2], cy], 1),
                            np.stack([cx, B[:, 1]], 1), np.stack([cx, B[:, 3]], 1)], 1)    # [n, 4, 2]: left, right, top, bottom
            u = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 4, 2)
            w, h = np.abs(u[:, 1, 0] - u[:, 0, 0]), np.abs(u[:, 3, 1] - u[:, 2, 1])
        else:
            if r['domain'] == 'indoor_synth':
                p = proj[r['scene']]
                fx, fy = float(p['M_proj_00']) * W / 2, float(p['M_proj_11']) * H / 2
            else:
                fx, fy = r['K'][0][0], r['K'][1][1]
            w, h = (B[:, 2] - B[:, 0]) / fx, (B[:, 3] - B[:, 1]) / fy
        out[r['id']] = np.sqrt(w * h).tolist()
    Path(a.out).write_text(json.dumps(out))
    from collections import Counter
    print('saved', a.out, len(out), 'images', dict(Counter(r['domain'] for r in rows)))


if __name__ == '__main__':
    main()
