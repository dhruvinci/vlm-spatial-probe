"""E05 (experiments/E05-dense-vision-head/PREREG.md): dense heads on Qwen3.8-27B's frozen vision tower.

  depth   NYU Depth v2 (E01's split by scene): per-patch MLP on blocks 6/13/20/26 -> log depth on a 2x2 sub-patch grid,
          image at 1024x768 (the pre-registered 960x720 is not a multiple of 32); scored per valid Kinect pixel (abs-rel, delta<1.25) against MoGe-2 on the same pixels
  joints  COCO person keypoints: per-patch MLP on blocks 13/20/26 of a 384x288 crop (true box padded 25%) -> which
          patch holds each joint plus an offset inside it; PCK at 5% / 10% of the box diagonal against ViTPose+ base

  python src/dense_head.py data_joints --coco ANNOTATIONS.zip --out DIR
  python src/dense_head.py depth  --shard SHARD --cfg QWEN_CONFIG_DIR --nyu DIR --out DIR      (env: transformers)
  python src/dense_head.py joints --shard SHARD --cfg QWEN_CONFIG_DIR --crops DIR --out DIR
  python src/dense_head.py vitpose --crops DIR --out DIR                                        (env: ViTPose)
  python src/dense_head.py moge --nyu DIR --out DIR                                             (env: MoGe-2)
  python src/dense_head.py score --out DIR
"""
import argparse
import io
import json
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 2604
DEPTH_LAYERS, JOINT_LAYERS = (6, 13, 20, 26), (13, 20, 26)
DEPTH_HW, CROP_HW = (768, 1024), (384, 288)                     # (height, width), multiples of 32 (patch 16 x merge 2)
PH, PW = DEPTH_HW[0] // 16, DEPTH_HW[1] // 16                     # 48 x 64 patches; 96 x 128 sub-patch cells


# ---------------------------------------------------------------------------- the frozen tower
def load_tower(shard, cfg_dir, dev):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from extract import fast_patch_embed
    cfg = AutoConfig.from_pretrained(cfg_dir)
    tower = Qwen3_5VisionModel._from_config(cfg.vision_config, dtype=torch.bfloat16)
    sd = {}
    with safe_open(shard, 'pt') as f:
        for k in f.keys():
            if k.startswith('model.visual.'):
                sd[k[len('model.visual.'):]] = f.get_tensor(k)
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    missing = [m for m in missing if 'rotary' not in m and 'inv_freq' not in m]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    tower.to(dev).eval()
    fast_patch_embed(tower.patch_embed, {})
    return tower, AutoImageProcessor.from_pretrained(cfg_dir)


def tower_features(tower, ip, images, layers, dev):
    """[N, h, w, len(layers) * d] in image order (restore_grid on the processor's 2x2-grouped patches)."""
    import torch
    from extract import restore_grid
    enc = ip(images=images, do_resize=False, return_tensors='pt')
    outs = {}
    hooks = [tower.blocks[l].register_forward_hook(lambda m, i, o, l=l: outs.__setitem__(l, o)) for l in layers]
    with torch.no_grad():
        tower(enc['pixel_values'].to(dev), grid_thw=enc['image_grid_thw'].to(dev))
    for h in hooks:
        h.remove()
    thw = enc['image_grid_thw']
    sizes = thw.prod(-1).tolist()
    per = [torch.split(outs[l], sizes) for l in layers]
    feats = [torch.cat([restore_grid(per[j][i], int(thw[i, 1]), int(thw[i, 2])) for j in range(len(layers))], -1) for i in range(len(images))]
    return torch.stack(feats).float()


def head(d_in, d_out):
    import torch.nn as nn
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d_out))


# ---------------------------------------------------------------------------- depth (NYU)
def nyu_sets(nyu):
    rows = {r['id']: r for r in json.loads((Path(nyu)/'dataset.json').read_text()) if r['domain'] == 'indoor'}
    e01 = {r['id']: r for r in json.loads((Path(nyu)/'e01_dataset.json').read_text()) if r['domain'] == 'indoor'}
    test_scenes = {r['scene'] for r in e01.values() if r['split'] == 'test'}
    dense = np.load(Path(nyu)/'nyu_dense.npz')
    ids = sorted({k.split('__')[0] for k in dense.files})
    ids = [i for i in ids if i in rows or i in e01]
    meta = {**rows, **e01}
    test = [i for i in ids if meta[i]['scene'] in test_scenes]
    train = [i for i in ids if meta[i]['scene'] not in test_scenes]
    return meta, dense, train, test


def depth_targets(depth):
    """Mean log depth per sub-patch cell (2x2 per 16 px patch of the 1024x768 input -> 96x128 cells), NaN if none."""
    d = np.asarray(depth, np.float32)
    H, W = DEPTH_HW[0] // 8, DEPTH_HW[1] // 8
    ys = np.linspace(0, d.shape[0], H + 1).round().astype(int); xs = np.linspace(0, d.shape[1], W + 1).round().astype(int)
    t = np.full((H, W), np.nan, np.float32)
    for i in range(H):
        for j in range(W):
            c = d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            c = c[c > 0]
            if c.size:
                t[i, j] = np.log(c).mean()
    return t


def cmd_depth(a):
    import torch
    dev = 'cuda'
    tower, ip = load_tower(a.shard, a.cfg, dev)
    meta, dense, train, test = nyu_sets(a.nyu)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    img = lambda i: Image.open(Path(a.nyu)/meta[i]['image']).convert('RGB').resize(DEPTH_HW[::-1], Image.BICUBIC)
    tg = {i: depth_targets(dense[f'{i}__depth']) for i in train + test}
    # the tower is frozen: its training-image features are computed once and kept on disk (fp16), then reused each epoch
    cache = np.lib.format.open_memmap(Path(a.out)/'depth_train_features.npy', 'w+', np.float16, (len(train), PH, PW, len(DEPTH_LAYERS) * 1152))
    for j in range(0, len(train), 4):
        cache[j:j + 4] = tower_features(tower, ip, [img(i) for i in train[j:j + 4]], DEPTH_LAYERS, dev).cpu().numpy().astype(np.float16)
    cache.flush()
    row = {i: j for j, i in enumerate(train)}
    torch.manual_seed(SEED)
    net = head(len(DEPTH_LAYERS) * 1152, 4).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(SEED)
    t0 = time.time()
    for ep in range(20):
        order = train[:]; rng.shuffle(order)
        tot = 0.0
        for b in range(0, len(order), 4):
            ids = order[b:b + 4]
            F = torch.as_tensor(np.stack([cache[row[i]] for i in ids]), device=dev).float()     # [B, PH, PW, D]
            pred = net(F).view(len(ids), PH, PW, 2, 2).permute(0, 1, 3, 2, 4).reshape(len(ids), 2 * PH, 2 * PW)
            y = torch.as_tensor(np.stack([tg[i] for i in ids]), device=dev)
            ok = torch.isfinite(y)
            loss = (pred[ok] - y[ok]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print('depth epoch', ep, round(tot / (len(order) / 4), 4), round(time.time() - t0), 's', flush=True)
    out = Path(a.out)
    del cache
    (out/'depth_train_features.npy').unlink()                          # temporary
    per_img = {}
    net.eval()
    with torch.no_grad():
        for i in test:
            F = tower_features(tower, ip, [img(i)], DEPTH_LAYERS, dev)
            p = net(F).view(1, PH, PW, 2, 2).permute(0, 1, 3, 2, 4).reshape(1, 1, 2 * PH, 2 * PW)
            full = torch.nn.functional.interpolate(p, size=(480, 640), mode='bilinear', align_corners=False)[0, 0].exp().cpu().numpy()
            per_img[i] = pixel_errors(full, dense[f'{i}__depth'])
    (out/'depth_head.json').write_text(json.dumps({'test_images': test, 'n_train_images': len(train), 'per_image': per_img}))
    print('saved depth head results', len(test), 'test images')


def pixel_errors(pred, depth):
    d = np.asarray(depth, np.float32); ok = (d > 0) & np.isfinite(pred) & (pred > 0)
    ar = np.abs(pred[ok] - d[ok]) / d[ok]; dl = np.maximum(pred[ok] / d[ok], d[ok] / pred[ok]) < 1.25
    return {'n': int(ok.sum()), 'abs_rel_sum': float(ar.sum()), 'delta_sum': float(dl.sum())}


def cmd_moge(a):
    import torch
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal').cuda().eval()
    meta, dense, train, test = nyu_sets(a.nyu)
    per = {}
    for i in test:
        x = torch.from_numpy(np.asarray(Image.open(Path(a.nyu)/meta[i]['image']).convert('RGB')).copy()).cuda().permute(2, 0, 1).float() / 255
        with torch.no_grad():
            d = model.infer(x)['depth'].float().cpu().numpy()
        per[i] = pixel_errors(d, dense[f'{i}__depth'])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/'depth_moge.json').write_text(json.dumps({'test_images': test, 'per_image': per}))
    print('saved MoGe-2 depth', len(test))


# ---------------------------------------------------------------------------- joints (COCO)
def crop_box(b, W, H):
    """True COCO box -> padded 3:4 (w:h) crop box, 25% larger."""
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    if w / h > CROP_HW[1] / CROP_HW[0]:
        h = w * CROP_HW[0] / CROP_HW[1]
    else:
        w = h * CROP_HW[1] / CROP_HW[0]
    w, h = w * 1.25, h * 1.25
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def cmd_data_joints(a):
    z = zipfile.ZipFile(a.coco)
    out = Path(a.out); (out/'images').mkdir(parents=True, exist_ok=True)
    sets = {}
    for split, n in (('train2017', 3000), ('val2017', 600)):
        d = json.loads(z.read(f'annotations/person_keypoints_{split}.json'))
        imgs = {i['id']: i for i in d['images']}
        cands = [p for p in d['annotations'] if not p['iscrowd'] and p['num_keypoints'] >= 10 and min(p['bbox'][2], p['bbox'][3]) >= 96]
        random.Random(SEED).shuffle(cands)
        sel = cands[:n]
        sets[split] = [{'ann_id': p['id'], 'image_id': p['image_id'], 'file': f"{split}_{p['image_id']:012d}.jpg",
                        'url': imgs[p['image_id']]['coco_url'], 'size': [imgs[p['image_id']]['width'], imgs[p['image_id']]['height']],
                        'bbox': p['bbox'], 'keypoints': p['keypoints'], 'licence': imgs[p['image_id']]['license']} for p in sel]

    def get(r):
        p = out/'images'/r['file']
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(r['url'], timeout=60).read())
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, [r for s in sets.values() for r in s]))
    (out/'people.json').write_text(json.dumps(sets))
    print({k: len(v) for k, v in sets.items()})


def load_crop(crops, r):
    im = Image.open(Path(crops)/'images'/r['file']).convert('RGB')
    W, H = im.size
    cb = crop_box(r['bbox'], W, H)
    c = im.crop(tuple(int(round(v)) for v in cb)).resize(CROP_HW[::-1], Image.BICUBIC)
    k = np.array(r['keypoints'], float).reshape(17, 3)
    uv = np.stack([(k[:, 0] - cb[0]) / (cb[2] - cb[0]) * CROP_HW[1], (k[:, 1] - cb[1]) / (cb[3] - cb[1]) * CROP_HW[0]], 1)
    return c, uv, k[:, 2], cb


def cmd_joints(a):
    import torch
    import torch.nn.functional as Fn
    dev = 'cuda'
    tower, ip = load_tower(a.shard, a.cfg, dev)
    S = json.loads((Path(a.crops)/'people.json').read_text())
    hp, wp = CROP_HW[0] // 16, CROP_HW[1] // 16
    torch.manual_seed(SEED)
    net = head(len(JOINT_LAYERS) * 1152, 17 * 3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(SEED)
    train = S['train2017']
    # the tower is frozen: training-crop features computed once, kept on disk (fp16) and reused each epoch
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(Path(a.out)/'joints_train_features.npy', 'w+', np.float16, (len(train), hp, wp, len(JOINT_LAYERS) * 1152))
    crops = {}
    for j in range(0, len(train), 32):
        bt = [load_crop(a.crops, r) for r in train[j:j + 32]]
        cache[j:j + len(bt)] = tower_features(tower, ip, [c for c, _, _, _ in bt], JOINT_LAYERS, dev).cpu().numpy().astype(np.float16)
        for r, (_, uv, v, _) in zip(train[j:j + 32], bt):
            crops[r['ann_id']] = (uv, v)
    cache.flush()
    row = {r['ann_id']: j for j, r in enumerate(train)}
    print('cached training features', round(time.time() - time.time()), flush=True)
    t0 = time.time()
    for ep in range(30):
        rng.shuffle(train)
        tot = 0.0
        for b in range(0, len(train), 32):
            batch = [(None, *crops[r['ann_id']], None) for r in train[b:b + 32]]
            F = torch.as_tensor(np.stack([cache[row[r['ann_id']]] for r in train[b:b + 32]]), device=dev).float()   # [B, hp, wp, D]
            o = net(F).view(len(batch), hp * wp, 17, 3)
            uv = torch.as_tensor(np.stack([u for _, u, _, _ in batch]), device=dev, dtype=torch.float32)
            vis = torch.as_tensor(np.stack([v for _, _, v, _ in batch]) == 2, device=dev)
            pu, pv = (uv[..., 0] / 16).clamp(0, wp - 1e-3), (uv[..., 1] / 16).clamp(0, hp - 1e-3)
            cell = (pv.long() * wp + pu.long())                                            # [B, 17]
            logits = o[..., 0].permute(0, 2, 1)                                             # [B, 17, P]
            ce = Fn.cross_entropy(logits[vis], cell[vis])
            off = torch.tanh(o[..., 1:]) * 0.5                                               # [B, P, 17, 2]
            sel = off.gather(1, cell[:, None, :, None].expand(-1, 1, -1, 2))[:, 0]           # [B, 17, 2]
            tgt = torch.stack([pu - pu.floor() - 0.5, pv - pv.floor() - 0.5], -1)
            l1 = (sel - tgt).abs().sum(-1)[vis].mean()
            loss = ce + l1
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        print('joints epoch', ep, round(tot / (len(train) / 32), 4), round(time.time() - t0), 's', flush=True)
    del cache
    (Path(a.out)/'joints_train_features.npy').unlink()                 # temporary
    net.eval()
    preds = {}
    with torch.no_grad():
        for r in S['val2017']:
            c, uv, v, cb = load_crop(a.crops, r)
            o = net(tower_features(tower, ip, [c], JOINT_LAYERS, dev)).view(hp * wp, 17, 3)
            cell = o[..., 0].argmax(0)                                                       # [17]
            off = torch.tanh(o[cell, torch.arange(17), 1:]) * 0.5
            px = ((cell % wp).float() + 0.5 + off[:, 0]) * 16; py = ((cell // wp).float() + 0.5 + off[:, 1]) * 16
            X = cb[0] + px.cpu().numpy() / CROP_HW[1] * (cb[2] - cb[0]); Y = cb[1] + py.cpu().numpy() / CROP_HW[0] * (cb[3] - cb[1])
            preds[str(r['ann_id'])] = np.stack([X, Y], 1).tolist()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/'joints_head.json').write_text(json.dumps(preds))
    print('saved joint head predictions', len(preds))


def cmd_vitpose(a):
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    S = json.loads((Path(a.crops)/'people.json').read_text())
    proc = AutoProcessor.from_pretrained('usyd-community/vitpose-plus-base')
    model = VitPoseForPoseEstimation.from_pretrained('usyd-community/vitpose-plus-base').cuda().eval()
    preds = {}
    for r in S['val2017']:
        im = Image.open(Path(a.crops)/'images'/r['file']).convert('RGB')
        inp = proc(im, boxes=[[r['bbox']]], return_tensors='pt').to('cuda')
        with torch.no_grad():
            o = model(**inp, dataset_index=torch.tensor([0], device='cuda'))
        res = proc.post_process_pose_estimation(o, boxes=[[r['bbox']]])[0][0]
        preds[str(r['ann_id'])] = res['keypoints'].cpu().numpy().tolist()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/'joints_vitpose.json').write_text(json.dumps(preds))
    print('saved ViTPose predictions', len(preds))


# ---------------------------------------------------------------------------- scoring
def boot_ci(values_by_unit, stat, B=2000):
    units = sorted(values_by_unit)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        pick = [units[j] for j in rng.integers(0, len(units), len(units))]
        vals.append(stat([values_by_unit[u] for u in pick]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def cmd_score(a):
    out = Path(a.out)
    R = {}
    if (out/'depth_head.json').exists() and (out/'depth_moge.json').exists():
        H = json.loads((out/'depth_head.json').read_text())['per_image']; M = json.loads((out/'depth_moge.json').read_text())['per_image']
        common = sorted(set(H) & set(M))
        ar = lambda L: sum(x['abs_rel_sum'] for x in L) / sum(x['n'] for x in L)
        dl = lambda L: sum(x['delta_sum'] for x in L) / sum(x['n'] for x in L)
        for name, D in (('head', H), ('moge2', M)):
            vb = {i: D[i] for i in common}
            R[f'depth_{name}'] = {'abs_rel': ar(list(vb.values())), 'abs_rel_ci': boot_ci(vb, ar), 'delta125': dl(list(vb.values())),
                                  'delta125_ci': boot_ci(vb, dl), 'n_images': len(common), 'n_pixels': sum(x['n'] for x in vb.values())}
        ratio = R['depth_head']['abs_rel'] / R['depth_moge2']['abs_rel']
        R['depth_pass'] = {'ratio_to_moge2': ratio, 'pass': bool(ratio <= 1.5)}
    if (out/'joints_head.json').exists() and (out/'joints_vitpose.json').exists():
        S = json.loads((Path(a.crops)/'people.json').read_text())['val2017']
        H = json.loads((out/'joints_head.json').read_text()); V = json.loads((out/'joints_vitpose.json').read_text())
        per = {'head': {}, 'vitpose': {}}
        for r in S:
            k = np.array(r['keypoints'], float).reshape(17, 3); vis = k[:, 2] == 2
            diag = float(np.hypot(r['bbox'][2], r['bbox'][3]))
            for name, P in (('head', H), ('vitpose', V)):
                d = np.linalg.norm(np.array(P[str(r['ann_id'])])[vis] - k[vis, :2], axis=1) / diag
                per[name][r['ann_id']] = d
        for name in ('head', 'vitpose'):
            for thr in (0.05, 0.10):
                f = lambda L, t=thr: float(np.mean(np.concatenate(L) < t))
                R[f'joints_{name}_pck{int(thr * 100)}'] = {'value': f(list(per[name].values())), 'ci': boot_ci(per[name], f), 'n_people': len(per[name]),
                                                           'n_keypoints': int(sum(len(x) for x in per[name].values()))}
        gap = R['joints_vitpose_pck5']['value'] - R['joints_head_pck5']['value']
        R['joints_pass'] = {'pck5_points_below_vitpose': gap * 100, 'pass': bool(gap * 100 <= 10)}
    (out/'results_e05.json').write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['data_joints', 'depth', 'joints', 'vitpose', 'moge', 'score'])
    ap.add_argument('--shard'); ap.add_argument('--cfg'); ap.add_argument('--nyu'); ap.add_argument('--crops')
    ap.add_argument('--coco'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    {'data_joints': cmd_data_joints, 'depth': cmd_depth, 'joints': cmd_joints, 'vitpose': cmd_vitpose, 'moge': cmd_moge,
     'score': cmd_score}[a.cmd](a)


if __name__ == '__main__':
    main()
