"""E04 part objectron, step 1 of 2: the front-axis audit (pre-registration: 20 frames per class, a class that is not
consistent in at least 18 of 20 is dropped).

Objectron (Ahmadyan et al., CVPR 2021; C-UDA 1.0), as pre-processed by Omni3D: annotations from Omni3D_json.zip
(datasets/Omni3D/Objectron_test.json; Omni3D's JSON is CC BY-NC 4.0) and images from objectron_images.zip, both read by
HTTP range from dl.fbaipublicfiles.com. For chair, laptop and bicycle, 20 test frames from 20 different videos (seed
2604; the same frame rule as the build, see objectron.py: frame_candidates) are drawn with the 3D box and the three
columns of R_cam as arrows from the box centre: +X red, +Y green, +Z blue (camera frame x right, y down, z forward), with the +X face shaded red and the +Z face blue. Omni3D puts the box extents l, h, w along X, Y, Z (checked against bbox3D_cam).
The grids are written to built_e04/objectron_audit/<class>_<k>.jpg, the frame list to
built_e04/objectron_audit/frames.json. The decision (per class: column, sign, consistent count) is made by looking and
written by hand to built_e04/objectron_audit.json and built_e04/objectron_audit.md.

Run: ~/Code/kakashi-research/dataset/.venv-depth/bin/python -I built_e04/code/objectron_audit.py
"""
import io
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, SEED, RemoteZip  # noqa: E402
from objectron import CLASSES, IMG_URL, JSON_URL, frame_candidates  # noqa: E402

N_AUDIT = 20


def draw(img, a, K, tag):
    im = np.ascontiguousarray(np.asarray(img.convert('RGB'))[..., ::-1])
    proj = lambda X: (K @ X.T).T[:, :2] / (K @ X.T).T[:, 2:]
    V = proj(np.array(a['bbox3D_cam']))
    s = max(im.shape) / 640
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(im, tuple(int(v) for v in V[i]), tuple(int(v) for v in V[j]), (255, 255, 255), max(1, int(2 * s)))
    c = np.array(a['center_cam']); R = np.array(a['R_cam']); d = np.array(a['dimensions'])
    half = np.array([d[2], d[1], d[0]]) / 2          # Omni3D: R_cam columns X, Y, Z carry l, h, w (checked on bbox3D_cam)
    over = im.copy()
    for k, col in ((0, (0, 0, 255)), (2, (255, 120, 0))):          # shade the +X face red and the +Z face blue
        idx = [i for i in range(3) if i != k]
        quad = [c + R @ (half * np.array([1.0 if i == k else s[idx.index(i)] if i in idx else 0 for i in range(3)]))
                for s in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
        cv2.fillPoly(over, [proj(np.array(quad)).astype(np.int32)], col)
    im = cv2.addWeighted(over, 0.35, im, 0.65, 0)
    L = 0.75 * float(np.max(d))
    for k, (col, name) in enumerate((((0, 0, 255), 'X'), ((0, 200, 0), 'Y'), ((255, 120, 0), 'Z'))):
        p = proj(np.array([c, c + R[:, k] * L]))
        p0, p1 = tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1])
        cv2.arrowedLine(im, p0, p1, col, max(2, int(6 * s)), tipLength=0.15)
        cv2.putText(im, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 1.6 * s, col, max(2, int(4 * s)))
    cv2.putText(im, tag, (int(20 * s), int(70 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2.0 * s, (0, 255, 255), max(2, int(5 * s)))
    return Image.fromarray(im[..., ::-1])


def main():
    J = RemoteZip(JSON_URL)
    d = json.loads(J.read('datasets/Omni3D/Objectron_test.json'))
    Z = RemoteZip(IMG_URL)
    outdir = OUT/'objectron_audit'
    outdir.mkdir(parents=True, exist_ok=True)
    cands = frame_candidates(d)
    listing = {}
    for cls in CLASSES:
        vids = sorted(cands[cls])
        rng = random.Random(f'{SEED}:audit:{cls}')
        pick = rng.sample(vids, N_AUDIT)
        tiles, listing[cls] = [], []
        for n, vid in enumerate(pick):
            img_rec, a = rng.choice(cands[cls][vid])
            img = Image.open(io.BytesIO(Z.read('datasets/' + img_rec['file_path'])))
            K = np.array(img_rec['K'])
            tiles.append(draw(img, a, K, f'{n + 1}').resize((540, 720)))
            listing[cls].append({'n': n + 1, 'video': vid, 'file_path': img_rec['file_path'], 'image_id': img_rec['id'],
                                 'annotation_id': a['id']})
        for k in range(0, N_AUDIT, 4):
            grid = Image.new('RGB', (1080, 1440))
            for j, t in enumerate(tiles[k:k + 4]):
                grid.paste(t, ((j % 2) * 540, (j // 2) * 720))
            grid.save(outdir/f'{cls}_{k // 4 + 1}.jpg', quality=88)
        print(cls, 'videos', len(vids), 'audited', len(pick), flush=True)
    (outdir/'frames.json').write_text(json.dumps(listing, indent=1))


def bicycle_sheets():
    """Second audit finding: Objectron's 'bike' class (Omni3D: 'bicycle') also holds motorcycles and scooters (10 of the
    20 audited frames). Every bicycle test video is drawn once (its first candidate frame, unannotated, numbered) so each
    can be labelled by eye as a pedal bicycle or a motorised two-wheeler; the labels go into objectron_audit.json."""
    J = RemoteZip(JSON_URL)
    d = json.loads(J.read('datasets/Omni3D/Objectron_test.json'))
    Z = RemoteZip(IMG_URL)
    outdir = OUT/'objectron_audit'
    vids = sorted(frame_candidates(d)['bicycle'].items())
    tiles = []
    for n, (vid, lst) in enumerate(vids):
        im, a = lst[0]
        img = Image.open(io.BytesIO(Z.read('datasets/' + im['file_path']))).convert('RGB').resize((270, 360))
        arr = np.ascontiguousarray(np.asarray(img)[..., ::-1])
        cv2.putText(arr, f'{n + 1}', (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(Image.fromarray(arr[..., ::-1]))
    for k in range(0, len(tiles), 16):
        grid = Image.new('RGB', (1080, 1440))
        for j, t in enumerate(tiles[k:k + 16]):
            grid.paste(t, ((j % 4) * 270, (j // 4) * 360))
        grid.save(outdir/f'bicycle_videos_{k // 16 + 1}.jpg', quality=88)
    (outdir/'bicycle_videos.json').write_text(json.dumps([{'n': n + 1, 'video': v, 'file_path': lst[0][0]['file_path']}
                                                          for n, (v, lst) in enumerate(vids)], indent=1))
    print('bicycle videos', len(vids))


if __name__ == '__main__':
    bicycle_sheets() if '--bicycle-sheets' in sys.argv else main()
