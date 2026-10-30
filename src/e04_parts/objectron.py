"""E04 part objectron (orientation), step 2 of 2: 400 frames, one per video, balanced across the classes kept by the
audit (objectron_audit.py, decision in built_e04/objectron_audit.json).

Objectron (Ahmadyan et al., CVPR 2021; C-UDA 1.0) test set as pre-processed by Omni3D (Brazil et al., CVPR 2023; the
JSON is CC BY-NC 4.0): annotations from Omni3D_json.zip (datasets/Omni3D/Objectron_test.json), images from
objectron_images.zip, both read by HTTP range.

Audit outcome (built_e04/objectron_audit.md): chair and bicycle kept, front = +X (first R_cam column); laptop dropped.
Objectron's bike class also holds motorcycles and scooters, so bicycle uses only the test videos labelled by eye as
pedal bicycles (objectron_audit.json: videos_bicycle).

Frames (frame_candidates): the class's annotation is the only one of that class in the image, valid3D, not behind the
camera; within a video, frames with Omni3D's computed truncation <= 0.3 are preferred when the video has any. One frame
per video (seed 2604). Balanced: 400 split evenly over the kept classes; a class with fewer videos gives all it has
and the rest is shared evenly by the others.

Truth (built_e04/SCHEMA.md): facing_deg with f = the audited R_cam column times its sign (the object's front), v = the
unit vector from the box centre to the camera, and up = the camera's -y = (0, -1, 0), because the annotation carries no
gravity. Also facing_deg_box_up (exploratory, not pre-registered): the same with up = the box's audited vertical axis.
dims_m = [h, w, l] from Omni3D's `dimensions` ([w, h, l], metres). box = Omni3D's bbox2D_trunc (the 3D box projected
and clipped to the image; Objectron has no tight 2D box), as fractions. group = video; split 60/40 by video.

Run: ~/Code/kakashi-research/dataset/.venv-depth/bin/python -I built_e04/code/objectron.py
"""
import io
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, SEED, RemoteZip, facing_deg, save_image, split_by_group  # noqa: E402

PART = 'objectron'
JSON_URL = 'https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip'
IMG_URL = 'https://dl.fbaipublicfiles.com/omni3d_data/objectron_images.zip'
CLASSES = ('chair', 'laptop', 'bicycle')
N_ROWS = 400


def frame_candidates(d):
    """{class: {video: [(image record, annotation), ...]}} under the frame rule in the module docstring."""
    ims = {i['id']: i for i in d['images']}
    per_img = Counter((a['image_id'], a['category_name']) for a in d['annotations'])
    out = {c: {} for c in CLASSES}
    for a in d['annotations']:
        c = a['category_name']
        if c not in CLASSES or per_img[(a['image_id'], c)] != 1 or not a['valid3D'] or a['behind_camera']:
            continue
        im = ims[a['image_id']]
        vid = Path(im['file_path']).stem.rsplit('_', 1)[0]
        out[c].setdefault(vid, []).append((im, a))
    for c in CLASSES:
        for vid, lst in out[c].items():
            low = [x for x in lst if 0 <= x[1]['truncation'] <= 0.3]
            out[c][vid] = sorted(low or lst, key=lambda x: x[0]['file_path'])
    return out


def main():
    audit = json.loads((OUT/'objectron_audit.json').read_text())
    kept = [c for c in CLASSES if audit[c]['keep']]
    J = RemoteZip(JSON_URL)
    d = json.loads(J.read('datasets/Omni3D/Objectron_test.json'))
    cands = frame_candidates(d)
    for c in kept:                                             # bicycle: only the videos labelled as pedal bicycles
        if 'videos_bicycle' in audit[c]:
            cands[c] = {v: x for v, x in cands[c].items() if v in set(audit[c]['videos_bicycle'])}
    supply = {c: len(cands[c]) for c in kept}
    quota = {c: 0 for c in kept}
    for k in range(N_ROWS):                                    # evenly, capped by each class's number of videos
        room = [c for c in kept if quota[c] < supply[c]]
        if not room:                                           # total supply under 400: take all of it (reported)
            break
        quota[min(room, key=lambda c: (quota[c], kept.index(c)))] += 1
    print('supply (videos)', supply, 'quota', quota, flush=True)

    picks = []
    for c in kept:
        rng = random.Random(f'{SEED}:{c}')
        vids = sorted(cands[c])
        rng.shuffle(vids)
        for vid in vids[:quota[c]]:
            im, a = rng.choice(cands[c][vid])
            picks.append((c, vid, im, a))
    Z = RemoteZip(IMG_URL)

    def work(p):
        c, vid, im, a = p
        rid = f"{PART}_{Path(im['file_path']).stem}"
        img = Image.open(io.BytesIO(Z.read('datasets/' + im['file_path'])))
        W, H = img.size
        assert [W, H] == [im['width'], im['height']], (W, H, im['width'], im['height'])
        size = save_image(img, OUT/'images'/PART/f'{rid}.jpg')
        R = np.array(a['R_cam'])
        col, sign = audit[c]['front_column'], audit[c]['front_sign']
        f = sign * R[:, col]
        ctr = np.array(a['center_cam'])
        v = -ctr / np.linalg.norm(ctr)
        obj = {'name': c, 'box': [max(0.0, a['bbox2D_trunc'][0] / W), max(0.0, a['bbox2D_trunc'][1] / H),
                                  min(1.0, a['bbox2D_trunc'][2] / W), min(1.0, a['bbox2D_trunc'][3] / H)],
               'facing_deg': round(facing_deg(f, v, (0.0, -1.0, 0.0)), 2),
               'dims_m': [round(float(x), 4) for x in (a['dimensions'][1], a['dimensions'][0], a['dimensions'][2])],
               'center_cam_m': [round(float(x), 4) for x in ctr], 'truncation': a['truncation']}
        if audit[c].get('up_column') is not None:
            up = audit[c]['up_sign'] * R[:, audit[c]['up_column']]
            obj['facing_deg_box_up'] = round(facing_deg(f, v, up), 2)
        return {'id': rid, 'measure': 'orientation', 'domain': PART, 'image': f'images/{PART}/{rid}.jpg', 'size': size,
                'group': vid,
                'source': {'dataset': 'Objectron test set, via Omni3D', 'video': vid, 'file_path': im['file_path'],
                           'omni3d_image_id': im['id'], 'omni3d_annotation_id': a['id'],
                           'licence': 'C-UDA 1.0 (Objectron); Omni3D annotations CC BY-NC 4.0',
                           'licence_url': 'https://github.com/google-research-datasets/Objectron/blob/master/LICENSE',
                           'omni3d_licence_url': 'https://github.com/facebookresearch/omni3d/blob/main/LICENSE.md'},
                'objects': [obj]}

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(work, picks))
    split_by_group(rows)
    rows.sort(key=lambda r: r['id'])
    (OUT/f'{PART}.json').write_text(json.dumps(rows, indent=0))
    print('rows', len(rows), 'classes', Counter(r['objects'][0]['name'] for r in rows))
    print('split', Counter((r['objects'][0]['name'], r['split']) for r in rows))
    fd = np.array([r['objects'][0]['facing_deg'] for r in rows])
    print('facing_deg in 45-degree bins', np.histogram(fd, bins=8, range=(0, 360))[0].tolist())


if __name__ == '__main__':
    main()
