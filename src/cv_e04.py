"""E04 specialist references (experiments/E04-more-measures/PREREG.md), on the same stored images as the VLMs:
  normals  surfaces items: MoGe-2's predicted normal (camera frame: x right, y down, z forward), the median inside the
           object's mask (PREREG; masks from src/e04_parts/masks.py, at the saved image's size), renormalised
           (MoGe-2 ViT-L, normal head). Also, exploratory: `normal_cam_box`, the median over the central half of the box
  counts   counts items: RF-DETR Medium (COCO) detections of the item's class with score >= 0.5

  python src/cv_e04.py normals --data DATA_E04 --out OUT.json          (env: MoGe-2; DATA_E04/masks_{nyu,hypersim}.npz)
  python src/cv_e04.py counts  --data DATA_E04 --out OUT.json --weights rf-detr-medium.pth   (env: rfdetr)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cv_reference import central_half


def median_normal(v):
    v = v.reshape(-1, 3)
    v = v[np.isfinite(v).all(1)]
    m = np.median(v, 0) if len(v) else None
    return (m / np.linalg.norm(m)).tolist() if m is not None and np.linalg.norm(m) > 0 else None


def run_normals(rows, data):
    import torch
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
    masks = {}
    for part in ('nyu', 'hypersim'):
        z = np.load(data/f'masks_{part}.npz')
        masks.update({k: np.unpackbits(z[k])[:int(np.prod(z[k + ':shape']))].reshape(z[k + ':shape']).astype(bool)
                      for k in z.files if not k.endswith(':shape')})
    out = {}
    for n, r in enumerate(rows):
        img = np.asarray(Image.open(data/r['image']).convert('RGB'))
        H, W = img.shape[:2]
        with torch.no_grad():
            res = model.infer(torch.from_numpy(img).cuda().permute(2, 0, 1).float() / 255)
        nm = res['normal'].float().cpu().numpy()
        objs = []
        for o in r['objects']:                                      # one object per surfaces row; its mask is the row's
            assert len(r['objects']) == 1 and masks[r['id']].shape == (H, W), r['id']
            x1, y1, x2, y2 = central_half(o['box'], W, H)
            objs.append({'normal_cam': median_normal(nm[masks[r['id']]]), 'normal_cam_box': median_normal(nm[y1:y2, x1:x2])})
        out[r['id']] = {'objects': objs}
        if n % 100 == 0:
            print('normals', n, len(rows), flush=True)
    return out


def run_counts(rows, data, weights):
    import rfdetr
    from rfdetr.assets.coco_classes import COCO_CLASSES
    det = rfdetr.RFDETRMedium(pretrain_weights=weights)
    out = {}
    for n, r in enumerate(rows):
        img = Image.open(data/r['image']).convert('RGB')
        d = det.predict(img, threshold=0.5)
        names = [COCO_CLASSES[int(c)] for c in d.class_id]
        out[r['id']] = {'objects': [{'count': int(sum(1 for x in names if x == o['class']))} for o in r['objects']]}
        if n % 100 == 0:
            print('counts', n, len(rows), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['normals', 'counts'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights')
    a = ap.parse_args()
    data = Path(a.data)
    rows = json.loads((data/'dataset.json').read_text())
    if a.part == 'normals':
        res = run_normals([r for r in rows if r['measure'] == 'surfaces'], data)
    else:
        res = run_counts([r for r in rows if r['measure'] == 'counts'], data, a.weights)
    Path(a.out).write_text(json.dumps(res))
    print('saved', a.out, len(res))


if __name__ == '__main__':
    main()
