"""Specialist computer-vision references on the same objects (no VLM): MoGe-2 metric depth and 3D points, RF-DETR
boxes. Each model sees only the stored image (longer side 640 px), with no intrinsics, like the VLMs.

  moge    per object: median depth (m) and median 3D point over the central half of its box (MoGe-2 ViT-L, normal head)
  rfdetr  per object of a COCO class (people domain; two_people: the two most confident person boxes, ordered left to
          right): the most confident box of that class (RF-DETR Medium, COCO-trained)

  python src/cv_reference.py moge   --data DATA --out OUT.json     (env: MoGe-2)
  python src/cv_reference.py rfdetr --data DATA --out OUT.json --weights rf-detr-medium.pth   (env: rfdetr)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def central_half(box, W, H):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    return (int(max(0, (cx - w / 2) * W)), int(max(0, (cy - h / 2) * H)),
            int(min(W, np.ceil((cx + w / 2) * W))), int(min(H, np.ceil((cy + h / 2) * H))))


def run_moge(rows, data):
    import torch
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
    out = {}
    for n, r in enumerate(rows):
        img = np.asarray(Image.open(data/r['image']).convert('RGB'))
        H, W = img.shape[:2]
        x = torch.from_numpy(img).cuda().permute(2, 0, 1).float() / 255
        with torch.no_grad():
            res = model.infer(x)
        pts = res['points'].float().cpu().numpy()
        objs = []
        for o in r['objects']:
            a, b, c, d = central_half(o['box'], W, H)
            p = pts[b:max(d, b + 1), a:max(c, a + 1)].reshape(-1, 3)
            p = p[np.isfinite(p).all(1)]
            objs.append({'depth_m': float(np.median(p[:, 2])) if len(p) else None,
                         'centre_m': np.median(p, 0).tolist() if len(p) else None})
        out[r['id']] = {'objects': objs, 'fov_x_deg': float(np.degrees(2 * np.arctan(0.5 / res['intrinsics'][0, 0].item())))}
        if n % 100 == 0:
            print('moge', n, len(rows), flush=True)
    return out


def run_rfdetr(rows, data, weights):
    import rfdetr
    from rfdetr.assets.coco_classes import COCO_CLASSES
    det = rfdetr.RFDETRMedium(pretrain_weights=weights)
    name_to_id = {v: k for k, v in COCO_CLASSES.items()}
    out = {}
    for n, r in enumerate(rows):
        if r['domain'] not in ('people', 'two_people', 'figure'):
            continue
        img = Image.open(data/r['image']).convert('RGB')
        W, H = img.size
        d = det.predict(img, threshold=0.05)
        boxes = d.xyxy / np.array([W, H, W, H])
        objs = []
        if r['domain'] == 'two_people':
            ppl = np.argsort(-d.confidence[d.class_id == name_to_id['person']])[:2]
            pb = boxes[d.class_id == name_to_id['person']][ppl]
            pb = pb[np.argsort((pb[:, 0] + pb[:, 2]) / 2)] if len(pb) == 2 else []
            for k in range(2):
                objs.append({'box': pb[k].tolist() if len(pb) == 2 else None})
        else:
            for o in r['objects']:
                cid = name_to_id.get(o['name'])
                m = d.class_id == cid if cid is not None else np.zeros(len(d.class_id), bool)
                objs.append({'box': boxes[m][np.argmax(d.confidence[m])].tolist() if m.any() else None})
        out[r['id']] = {'objects': objs}
        if n % 100 == 0:
            print('rfdetr', n, len(rows), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['moge', 'rfdetr'])
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights')
    a = ap.parse_args()
    data = Path(a.data)
    rows = json.loads((data/'dataset.json').read_text())
    res = run_moge(rows, data) if a.part == 'moge' else run_rfdetr(rows, data, a.weights)
    Path(a.out).write_text(json.dumps(res))
    print('saved', a.out, len(res))


if __name__ == '__main__':
    main()
