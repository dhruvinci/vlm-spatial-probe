"""Build the probe dataset: real photos with sensor or annotation ground truth, three domains, fixed splits.

  people  COCO val2017 (Lin et al. 2014): one person plus at least one other object of a unique class; 2D boxes and
          person keypoints from the human annotations. Annotations CC BY 4.0; each image keeps its Flickr licence.
  indoor  NYU Depth v2, labelled set (Silberman et al. 2012): Kinect depth and human instance labels; 3D from depth
          and the dataset's colour-camera intrinsics.
  street  KITTI object training set (Geiger et al. 2012): 3D boxes from LiDAR-aided human annotation, CC BY-NC-SA 3.0.
  two_people  Harmony4D multi-person capture (Khirodkar et al. 2024; MIT): two people, one camera; metric 3D from the
          dataset's multi-view body fits. Mostly calm sequences (hugging, ballroom, sword, karate), with a minority of
          MMA and grappling frames; figures use the calm sequences only.

Named objects are instances whose class occurs exactly once in the image (so "the chair" is unambiguous). Per object:
2D position (box or mask centre, fraction of width and height); indoor and street also metric depth (m), a 3D
centre, and a size (m): KITTI the box's longest side, NYU the visible extent (5th-95th percentile spread of the
back-projected mask along its longest principal axis). Per pair: 3D distance (m) and relations with clear margins
(left/right and above/below: centres at least 10% of the width or height apart; closer/farther: depths differ by
at least 15%; bigger/smaller: sizes differ by at least 1.5x).

Splits (seed 2604): people 70% train / 30% test by image; indoor 70/30 by scene (NYU's scene label, so no room is in
both); two_people by time within each sequence (first 65% train, last 25% test, the 10% between dropped; grappling,
one frame per sequence, 70/30 by sequence); street is
entirely held out (a domain the probes never train on). Images are stored with their longer side at most 640 px.

  python src/build_dataset.py coco   --src DIR --out DATA      (local)
  python src/build_dataset.py kitti  --src DIR --out DATA      (local; images read from the zip by byte range)
  python src/build_dataset.py nyu    --src DIR --out DATA      (needs h5py; run on the pod)
  python src/build_dataset.py h4d    --src DIR --out DATA      (Harmony4D benchmark folder; needs engage3d's loader)
  python src/build_dataset.py kitti_classes --src DIR --out DATA   (adds every labelled class per frame, for the naming check)
  python src/build_dataset.py figure1 --src DIR --out DATA     (Figure 1: the baseball photo, COCO train2017 246285)
  python src/build_dataset.py finish --out DATA                (splits, relations, pairs -> DATA/dataset.json)
"""
import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 2604
N_PER_DOMAIN = 400
MAX_SIDE = 640
COCO_SPORT = {'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'skateboard', 'surfboard', 'skis',
              'snowboard', 'frisbee', 'kite', 'bicycle', 'horse', 'motorcycle', 'dog'}
KEYPOINTS = ['nose', 'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
NYU_NAMES = {'chair', 'table', 'bed', 'sofa', 'lamp', 'door', 'window', 'picture', 'cabinet', 'television', 'sink',
             'toilet', 'pillow', 'desk', 'bookshelf', 'refrigerator', 'counter', 'mirror', 'towel', 'box', 'bag',
             'monitor', 'night stand', 'dresser', 'stool', 'bathtub', 'shelves', 'curtain', 'clock', 'plant',
             'computer', 'keyboard', 'microwave', 'oven', 'stove', 'bottle', 'cup', 'bowl', 'blinds', 'whiteboard',
             'tv stand', 'coffee table', 'ottoman', 'piano', 'laptop', 'printer', 'fan', 'vase', 'speaker', 'basket'}
KITTI_CLASSES = {'Car': 'car', 'Van': 'van', 'Truck': 'truck', 'Pedestrian': 'pedestrian', 'Cyclist': 'cyclist', 'Tram': 'tram'}
NYU_K = dict(fx=518.857901, fy=519.469611, cx=325.582449, cy=253.736166)   # colour camera of the labelled set


def save_image(img, path):
    w, h = img.size
    s = min(1.0, MAX_SIDE / max(w, h))
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(path, quality=93)
    return img.size, s


# ------------------------------------------------------------------ COCO (people / sports)
def build_coco(src, out):
    import urllib.request
    z = zipfile.ZipFile(Path(src)/'annotations_trainval2017.zip')
    inst = json.loads(z.read('annotations/instances_val2017.json'))
    kps = json.loads(z.read('annotations/person_keypoints_val2017.json'))
    cats = {c['id']: c['name'] for c in inst['categories']}
    lic = {l['id']: (l['name'], l['url']) for l in inst['licenses']}
    imgs = {i['id']: i for i in inst['images']}
    by_img = {}
    for a in inst['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)
    kp_by_ann = {a['id']: a for a in kps['annotations']}
    kp_names = kps['categories'][0]['keypoints']
    cands = []
    for iid, anns in by_img.items():
        if any(a.get('iscrowd') for a in anns):
            continue
        names = [cats[a['category_id']] for a in anns]
        if names.count('person') != 1:
            continue
        uniq = [a for a in anns if names.count(cats[a['category_id']]) == 1 and cats[a['category_id']] != 'person']
        im = imgs[iid]; area = im['width'] * im['height']
        person = next(a for a in anns if cats[a['category_id']] == 'person')
        if person['area'] < 0.05 * area or kp_by_ann.get(person['id'], {}).get('num_keypoints', 0) < 10:
            continue
        uniq = [a for a in uniq if a['area'] >= 0.002 * area]
        if not uniq:
            continue
        sport = any(cats[a['category_id']] in COCO_SPORT for a in uniq)
        cands.append((not sport, iid, person, sorted(uniq, key=lambda a: -a['area'])[:2]))
    random.Random(SEED).shuffle(cands)
    cands.sort(key=lambda c: c[0])                      # sports images first, then the rest, each in shuffled order
    rows = []
    for _, iid, person, others in cands[:N_PER_DOMAIN]:
        im = imgs[iid]
        p = out/'images'/'people'/f'coco_{iid:012d}.jpg'
        if not p.exists():
            data = urllib.request.urlopen(im['coco_url'], timeout=60).read()
            save_image(Image.open(io.BytesIO(data)), p)
        W, H = im['width'], im['height']
        objs = []
        for a in [person] + others:
            x, y, w, h = a['bbox']
            o = {'name': cats[a['category_id']], 'xy': [(x + w / 2) / W, (y + h / 2) / H],
                 'box': [x / W, y / H, (x + w) / W, (y + h) / H]}
            if a is person:
                k = np.array(kp_by_ann[a['id']]['keypoints'], float).reshape(-1, 3)
                o['keypoints'] = {n: [k[kp_names.index(n), 0] / W, k[kp_names.index(n), 1] / H]
                                  for n in KEYPOINTS if k[kp_names.index(n), 2] == 2}
            objs.append(o)
        ln, lu = lic[im['license']]
        rows.append({'id': f'coco_{iid}', 'domain': 'people', 'image': str(p.relative_to(out)), 'size': [W, H],
                     'classes': sorted({cats[a['category_id']] for a in by_img[iid]}),
                     'source': {'dataset': 'COCO val2017', 'coco_id': iid, 'flickr_url': im.get('flickr_url'),
                                'licence': ln, 'licence_url': lu}, 'objects': objs})
    (out/'coco.json').write_text(json.dumps(rows))
    print('coco', len(rows), 'images;', sum(len(r['objects']) for r in rows), 'objects;',
          sum(1 for _, iid, _, o in cands[:N_PER_DOMAIN] if any(cats[a['category_id']] in COCO_SPORT for a in o)), 'with sports objects')


# ------------------------------------------------------------------ KITTI (street)
class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP Range requests (enough for zipfile)."""
    def __init__(self, url, block=1 << 20):
        import urllib.request
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length'])

    def _fetch(self, i):
        import urllib.request
        if i not in self.cache:
            a = i * self.block; b = min(self.size, a + self.block) - 1
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={a}-{b}'})
            self.cache = {i: urllib.request.urlopen(req, timeout=120).read()} if len(self.cache) > 64 else {**self.cache, i: urllib.request.urlopen(req, timeout=120).read()}
        return self.cache[i]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            i = p // self.block; blk = self._fetch(i); s = p - i * self.block
            take = blk[s:s + n - len(out)]; out += take; p += len(take)
        buf[:n] = out; self.pos += n
        return n


def kitti_objects(label_txt, P2):
    objs = []
    for line in label_txt.strip().splitlines():
        f = line.split()
        if f[0] not in KITTI_CLASSES:
            continue
        trunc, occ = float(f[1]), int(f[2])
        x1, y1, x2, y2 = map(float, f[4:8]); h, w, l = map(float, f[8:11]); x, y, z = map(float, f[11:14])
        objs.append({'cls': f[0], 'trunc': trunc, 'occ': occ, 'box_px': [x1, y1, x2, y2], 'dims': [h, w, l],
                     'centre': [x, y - h / 2, z]})
    return objs


def build_kitti(src, out):
    labels = zipfile.ZipFile(Path(src)/'labels.zip'); calib = zipfile.ZipFile(Path(src)/'calib.zip')
    names = sorted(n for n in labels.namelist() if n.endswith('.txt'))
    cands = []
    for n in names:
        objs = kitti_objects(labels.read(n).decode(), None)
        classes = [o['cls'] for o in objs]
        good = [o for o in objs if classes.count(o['cls']) == 1 and o['occ'] <= 1 and o['trunc'] <= 0.3
                and o['box_px'][3] - o['box_px'][1] >= 25 and o['centre'][2] > 2]
        if len(good) >= 2:
            cands.append((Path(n).stem, good))
    random.Random(SEED).shuffle(cands)
    cands = cands[:N_PER_DOMAIN]
    zimg = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'), 1 << 20))
    members = {Path(m).stem: m for m in zimg.namelist() if m.startswith('training/image_2/') and m.endswith('.png')}
    rows = []
    for stem, objs in cands:
        p = out/'images'/'street'/f'kitti_{stem}.jpg'
        img = Image.open(io.BytesIO(zimg.read(members[stem])))
        W, H = img.size
        if not p.exists():
            save_image(img, p)
        cal = {l.split(':')[0]: np.array(l.split(':')[1].split(), float) for l in calib.read(f'training/calib/{stem}.txt').decode().splitlines() if ':' in l}
        P2 = cal['P2'].reshape(3, 4)
        ob = []
        for o in sorted(objs, key=lambda o: -(o['box_px'][2] - o['box_px'][0]) * (o['box_px'][3] - o['box_px'][1]))[:3]:
            x1, y1, x2, y2 = o['box_px']
            ob.append({'name': KITTI_CLASSES[o['cls']], 'xy': [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H],
                       'box': [x1 / W, y1 / H, x2 / W, y2 / H], 'depth_m': o['centre'][2], 'centre_m': o['centre'],
                       'size_m': max(o['dims']), 'dims_m': o['dims']})
        rows.append({'id': f'kitti_{stem}', 'domain': 'street', 'image': str(p.relative_to(out)), 'size': [W, H],
                     'K': P2[:, :3].tolist(), 'source': {'dataset': 'KITTI object (training)', 'frame': stem,
                     'licence': 'CC BY-NC-SA 3.0', 'licence_url': 'https://creativecommons.org/licenses/by-nc-sa/3.0/'},
                     'objects': ob})
        print(len(rows), end=' ', flush=True)
    (out/'kitti.json').write_text(json.dumps(rows))
    print('\nkitti', len(rows), 'images;', sum(len(r['objects']) for r in rows), 'objects')


def kitti_classes(src, out):
    labels = zipfile.ZipFile(Path(src)/'labels.zip')
    rows = json.loads((out/'kitti.json').read_text())
    for r in rows:
        txt = labels.read(f"training/label_2/{r['source']['frame']}.txt").decode()
        r['classes'] = sorted({KITTI_CLASSES.get(l.split()[0], l.split()[0].lower()) for l in txt.strip().splitlines()} - {'dontcare'})
    (out/'kitti.json').write_text(json.dumps(rows))
    print('kitti classes', sum(len(r['classes']) for r in rows))


# ------------------------------------------------------------------ NYU Depth v2 (indoor)
def build_nyu(src, out):
    import h5py
    f = h5py.File(Path(src)/'nyu_depth_v2_labeled.mat', 'r')
    names = [''.join(chr(c) for c in f[r][()].ravel()) for r in f['names'][0]]
    scenes = [''.join(chr(c) for c in f[r][()].ravel()) for r in f['scenes'][0]]
    N = f['images'].shape[0]
    order = list(range(N)); random.Random(SEED).shuffle(order)
    rows, gt = [], {}
    fx, fy, cx, cy = NYU_K['fx'], NYU_K['fy'], NYU_K['cx'], NYU_K['cy']
    for i in order:
        if len(rows) >= N_PER_DOMAIN:
            break
        lab = f['labels'][i].T; ins = f['instances'][i].T; dep = f['depths'][i].T.astype(np.float32)   # [480, 640]
        H, W = lab.shape
        objs = []
        present = {}
        for c in np.unique(lab):
            if c == 0:
                continue
            nm = names[c - 1]
            if nm not in NYU_NAMES:
                continue
            ids = [k for k in np.unique(ins[lab == c]) if k > 0]
            present[nm] = len(ids)
            if len(ids) != 1:
                continue
            m = (lab == c) & (ins == ids[0])
            if m.sum() < 0.01 * H * W:
                continue
            d = dep[m]; d = d[d > 0]
            if len(d) < 0.8 * m.sum():
                continue
            v, u = np.nonzero(m); z = dep[v, u]; keep = z > 0
            X = np.stack([(u[keep] - cx) * z[keep] / fx, (v[keep] - cy) * z[keep] / fy, z[keep]], 1)
            cen = np.median(X, 0)
            Xc = X - X.mean(0); ax = np.linalg.svd(Xc[::max(1, len(Xc) // 4000)], full_matrices=False)[2][0]
            proj = Xc @ ax; size = float(np.percentile(proj, 95) - np.percentile(proj, 5))
            ys, xs = np.nonzero(m)
            objs.append({'name': nm, 'xy': [xs.mean() / W, ys.mean() / H],
                         'box': [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H],
                         'depth_m': float(np.median(d)), 'centre_m': cen.tolist(), 'size_m': size, 'area': int(m.sum())})
        if len(objs) < 2:
            continue
        objs = sorted(objs, key=lambda o: -o['area'])[:3]
        img = Image.fromarray(np.transpose(f['images'][i], (2, 1, 0)))
        p = out/'images'/'indoor'/f'nyu_{i:04d}.jpg'
        save_image(img, p)
        rows.append({'id': f'nyu_{i}', 'domain': 'indoor', 'image': str(p.relative_to(out)), 'size': [W, H], 'scene': scenes[i],
                     'classes': sorted({names[c - 1] for c in np.unique(lab) if c > 0}),
                     'K': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], 'source': {'dataset': 'NYU Depth v2 (labelled)', 'index': i,
                     'licence': 'no licence stated by the authors; research use', 'licence_url': 'https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html'},
                     'objects': objs})
        gt[f'nyu_{i}'] = {'depth': dep.astype(np.float16), 'labels': lab.astype(np.uint16), 'instances': ins.astype(np.uint8)}
    (out/'nyu.json').write_text(json.dumps(rows))
    np.savez_compressed(out/'nyu_dense.npz', **{f'{k}__{kk}': vv for k, v in gt.items() for kk, vv in v.items()})
    print('nyu', len(rows), 'images;', sum(len(r['objects']) for r in rows), 'objects')


# ------------------------------------------------------------------ Harmony4D (two people, metric 3D)
H4D_PLAN = {'01_hugging': 40, '05_sword2': 35, '08_ballroom2': 18, '09_karate': 70,     # frames per sequence
            '15_mma4': 25, '16_mma5': 25, '02_grappling': 1, '03_grappling2': 1}         # (a minority, never in figures)
H4D_GRAPPLING_SEQS = 40                                             # grappling: one random frame from each of 40 sequences
COCO17 = {'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10,
          'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}


def build_h4d(src, out):
    """Harmony4D multi-person capture (Khirodkar et al., NeurIPS 2024; MIT): one camera (cam01) of two people, ground
    truth from the dataset's multi-view SMPL fits (camera frame, metres; depth = pelvis z). Uses engage3d's clip loader (github.com/dhruvinci/engage3d, src/engage_clip.py;
    set ENGAGE3D_SRC to its folder). The two people are named by side: "the person on the left / right"."""
    import os
    import sys
    sys.path.insert(0, os.environ['ENGAGE3D_SRC'])                    # engage3d's src folder (H4DClip)
    from engage_clip import H4DClip
    root = Path(src)
    rng = random.Random(SEED)
    grap = sorted((a, s.name) for a in ('02_grappling', '03_grappling2') for s in (root/a).iterdir() if (s/'exo/cam01').exists())
    rng.shuffle(grap)
    grap = set(grap[:H4D_GRAPPLING_SEQS])
    rows = []
    for act in sorted(H4D_PLAN):
        for s in sorted((root/act).iterdir()):
            if not (s/'exo/cam01').exists() or (act in ('02_grappling', '03_grappling2') and (act, s.name) not in grap):
                continue
            clip = H4DClip(s.name, 'cam01')
            fr = clip.frames
            k = H4D_PLAN[act]
            picks = [fr[i] for i in np.linspace(0, len(fr) - 1, k).round().astype(int)] if k > 1 else [fr[rng.randrange(len(fr))]]
            W, H = clip.size
            for f in sorted(set(picks)):
                try:
                    boxes = clip.truth_boxes(f); bodies = clip.truth_bodies(f); j2 = clip.truth_2d(f)
                except Exception:
                    continue
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                if abs(cx[0] - cx[1]) < 0.05 * W or ((boxes[:, 3] - boxes[:, 1]) < 0.15 * H).any():
                    continue
                lab = clip.truth_labels(f, (W // 4, H // 4))
                order = np.argsort(cx)                              # left person first
                objs = []
                for side, n in zip(('left', 'right'), order):
                    x1, y1, x2, y2 = boxes[n]
                    pel = bodies[n][1]['pelvis']
                    kp = {}
                    for name, j in COCO17.items():
                        u, v = j2[n, j]
                        if 0 <= u < W and 0 <= v < H and lab[int(v) // 4, int(u) // 4] == n + 1:     # visible, own body
                            kp[name] = [float(u / W), float(v / H)]
                    objs.append({'name': f'person on the {side}', 'xy': [float((x1 + x2) / 2 / W), float((y1 + y2) / 2 / H)],
                                 'box': [float(x1 / W), float(y1 / H), float(x2 / W), float(y2 / H)],
                                 'depth_m': float(pel[2]), 'centre_m': [float(v) for v in pel], 'keypoints': kp})
                p = out/'images'/'two_people'/f'h4d_{s.name}_{f:05d}.jpg'
                if not p.exists():
                    save_image(Image.open(clip.image_path(f)), p)
                t = (f - fr[0]) / max(fr[-1] - fr[0], 1)
                rows.append({'id': f'h4d_{s.name}_{f}', 'domain': 'two_people', 'image': str(p.relative_to(out)), 'size': [W, H],
                             'activity': act.split('_', 1)[1].rstrip('0123456789'), 'sequence': s.name, 'time_frac': float(t),
                             'source': {'dataset': 'Harmony4D', 'sequence': s.name, 'camera': 'cam01', 'frame': int(f),
                                        'licence': 'MIT (dataset card)', 'licence_url': 'https://huggingface.co/datasets/Jyun-Ting/Harmony4D'},
                             'objects': objs})
            print(act, s.name, len(rows), flush=True)
    (out/'h4d.json').write_text(json.dumps(rows))
    from collections import Counter
    print('h4d', len(rows), 'frames', Counter(r['activity'] for r in rows))


# ------------------------------------------------------------------ Figure 1 (not scored)
FIGURE1 = {'coco_id': 246285, 'credit': '"Zach MacPhee Dives Back into First" by Barry Stahl, CC BY 2.0',
           'flickr': 'https://www.flickr.com/photo.gne?id=5505334912',
           'names': {(297, 133): 'player standing on the right', (50, 191): 'player diving to the base',
                     (120, 59): 'player in the background', (282, 218): "standing player's glove"}}


def figure1(src, out):
    """The baseball photo (COCO train2017 246285): four uniquely described objects, 2D truth from COCO's boxes."""
    import urllib.request
    z = zipfile.ZipFile(Path(src)/'annotations_trainval2017.zip')
    inst = json.loads(z.read('annotations/instances_train2017.json'))
    im = next(i for i in inst['images'] if i['id'] == FIGURE1['coco_id'])
    anns = [a for a in inst['annotations'] if a['image_id'] == im['id']]
    p = out/'images'/'figure'/f"coco_{im['id']:012d}.jpg"
    if not p.exists():
        save_image(Image.open(io.BytesIO(urllib.request.urlopen(im['coco_url'], timeout=60).read())), p)
    W, H = im['width'], im['height']
    objs = []
    for a in anns:
        x, y, w, h = a['bbox']
        name = FIGURE1['names'].get((round(x), round(y)))
        if name:
            objs.append({'name': name, 'xy': [(x + w / 2) / W, (y + h / 2) / H], 'box': [x / W, y / H, (x + w) / W, (y + h) / H]})
    row = {'id': f"coco_{im['id']}", 'domain': 'figure', 'split': 'figure', 'image': str(p.relative_to(out)), 'size': [W, H],
           'source': {'dataset': 'COCO train2017', 'coco_id': im['id'], 'flickr_url': FIGURE1['flickr'], 'credit': FIGURE1['credit'],
                      'licence': 'CC BY 2.0', 'licence_url': 'https://creativecommons.org/licenses/by/2.0/'}, 'objects': objs}
    (out/'figure1.json').write_text(json.dumps([row]))
    print('figure1', [o['name'] for o in objs])


# ------------------------------------------------------------------ splits, pairs, relations
def finish(out):
    rows = []
    for n in ('coco', 'nyu', 'kitti', 'h4d', 'figure1'):
        rows += json.loads((out/f'{n}.json').read_text())
    rng = random.Random(SEED)
    for dom, unit in (('people', 'id'), ('indoor', 'scene')):         # indoor: whole scenes go to one side
        units = sorted({r[unit] for r in rows if r['domain'] == dom}); rng.shuffle(units)
        test = set(units[:round(0.3 * len(units))])
        for r in rows:
            if r['domain'] == dom:
                r['split'] = 'test' if r[unit] in test else 'train'
    for r in rows:
        if r['domain'] == 'two_people' and r['activity'] != 'grappling':
            r['split'] = 'train' if r['time_frac'] < 0.65 else 'test' if r['time_frac'] >= 0.75 else 'gap'
    seqs = sorted({r['sequence'] for r in rows if r['domain'] == 'two_people' and r['activity'] == 'grappling'}); rng.shuffle(seqs)
    test = set(seqs[:round(0.3 * len(seqs))])
    for r in rows:
        if r['domain'] == 'two_people' and r['activity'] == 'grappling':
            r['split'] = 'test' if r['sequence'] in test else 'train'
    rows = [r for r in rows if r.get('split') != 'gap']
    for r in rows:
        if r['domain'] == 'street':
            r['split'] = 'heldout'
        pairs = []
        ob = r['objects']
        for a in range(len(ob)):
            for b in range(a + 1, len(ob)):
                A, B = ob[a], ob[b]
                rel = {}
                if abs(A['xy'][0] - B['xy'][0]) >= 0.10:
                    rel['left_of'] = A['xy'][0] < B['xy'][0]
                if abs(A['xy'][1] - B['xy'][1]) >= 0.10:
                    rel['above'] = A['xy'][1] < B['xy'][1]
                if 'depth_m' in A and 'depth_m' in B and max(A['depth_m'], B['depth_m']) / min(A['depth_m'], B['depth_m']) >= 1.15:
                    rel['closer'] = A['depth_m'] < B['depth_m']
                if 'size_m' in A and 'size_m' in B and max(A['size_m'], B['size_m']) / max(min(A['size_m'], B['size_m']), 1e-3) >= 1.5:
                    rel['bigger'] = A['size_m'] > B['size_m']
                p = {'a': a, 'b': b, 'relations': rel}
                if 'centre_m' in A and 'centre_m' in B:
                    p['distance_m'] = float(np.linalg.norm(np.subtract(A['centre_m'], B['centre_m'])))
                pairs.append(p)
        r['pairs'] = pairs
    (out/'dataset.json').write_text(json.dumps(rows, indent=0))
    from collections import Counter
    print('images', Counter((r['domain'], r['split']) for r in rows))
    print('objects', sum(len(r['objects']) for r in rows), 'pairs', sum(len(r['pairs']) for r in rows))
    print('relations', Counter(k for r in rows for p in r['pairs'] for k in p['relations']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['coco', 'kitti', 'kitti_classes', 'nyu', 'h4d', 'figure1', 'finish'])
    ap.add_argument('--src', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    {'coco': build_coco, 'kitti': build_kitti, 'kitti_classes': kitti_classes, 'nyu': build_nyu, 'h4d': build_h4d, 'figure1': figure1}.get(a.part, lambda s, o: finish(o))(a.src, out) if a.part != 'finish' else finish(out)
