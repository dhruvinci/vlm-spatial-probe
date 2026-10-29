"""Run one vision-language model over the probe dataset (GPU): hidden states for the probes, and the model's answers.

Parts (each is skipped if its output exists, so a run can resume):
  feat      image + the object list ("Objects: the chair; the lamp.") + the question. Saved per object, bf16:
              vis    [n_obj, vision blocks, d_vis]   patches inside the object's box, mean-pooled (image order
                                                     restored: restore_grid, and Qwen2.5-VL's window order undone)
              lmimg  [n_obj, LM layers + 1, d]       merged image tokens inside the box, mean-pooled (index 0 = the
                                                     embedding layer, i.e. the vision tower's projected output)
              word   [n_obj, LM layers + 1, d]       the state at the last token of "the <name>"
            and, with --dense, every indoor image's merged tokens at a subset of layers, random-projected to 256 dims.
  noimg     the same text with no image: word states (the name-only control)
  mis       the same text with another image of the same domain (a derangement): word states
  ask       generated JSON answers: centre x / y (percent), distance from the camera (m), size (m), pair distances (m)
  ground    generated native boxes ("bbox_2d")
  yesno     next-token yes / no scores for relation questions and a naming check
Greedy decoding, no thinking (enable_thinking=False). Nothing here scores anything; scoring is local (probe.py).

  python src/extract.py --model-dir M --name NAME --data DATA --out OUT [--dense] [--parts feat,noimg,mis,ask,yesno,ground]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEED = 2604
DENSE_DIMS = 256
REL_CAP = 400                     # street (held out): at most this many questions per relation
NAMING_STREET = 200               # street images in the naming check

ASK_Q = ('For each object, estimate: its centre as x and y in percent of the image width and height (0 to 100, '
         'measured from the top-left corner); its distance from the camera in metres; and its size (longest dimension) '
         'in metres. Also estimate the distance in metres between each pair of objects. Reply with JSON only, in this '
         'form: {"objects": [{"name": <name>, "x": <number>, "y": <number>, "depth_m": <number>, "size_m": <number>}], '
         '"pairs": [{"a": <name>, "b": <name>, "distance_m": <number>}]}')
GROUND_Q = 'Output the bounding box of each listed object as JSON: [{"label": <name>, "bbox_2d": [x1, y1, x2, y2]}].'
REL_Q = {'closer': 'Is the {a} closer to the camera than the {b}? Answer yes or no.',
         'left_of': 'In the image, is the {a} to the left of the {b}? Answer yes or no.',
         'above': 'In the image, is the {a} higher up than the {b}? Answer yes or no.',
         'bigger': 'In reality, is the {a} bigger than the {b}? Answer yes or no.'}
NAMING_Q = 'Is there a {name} in this image? Answer yes or no.'
# E06: a second wording of the same questions (same JSON schema; relations asked the other way round, same truth)
ASK_Q2 = ('Look at the listed objects. For each one, say how far it is from the camera in metres, how big it is (its '
          'longest side, in metres), and where its centre is in the image as a percentage of the width and height (0 to '
          '100, from the top-left). Then give the distance in metres between every pair. Reply with JSON only, in this '
          'form: {"objects": [{"name": <name>, "x": <number>, "y": <number>, "depth_m": <number>, "size_m": <number>}], '
          '"pairs": [{"a": <name>, "b": <name>, "distance_m": <number>}]}')
REL_Q2 = {'closer': 'Is the {b} farther from the camera than the {a}? Answer yes or no.',
          'left_of': 'In the image, is the {b} to the right of the {a}? Answer yes or no.',
          'above': 'In the image, is the {b} lower down than the {a}? Answer yes or no.',
          'bigger': 'In reality, is the {b} smaller than the {a}? Answer yes or no.'}
COCO80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
          'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
          'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
          'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
          'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
NYU_VOCAB = ['chair', 'table', 'bed', 'sofa', 'lamp', 'door', 'window', 'picture', 'cabinet', 'television', 'sink', 'toilet',
             'pillow', 'desk', 'bookshelf', 'refrigerator', 'counter', 'mirror', 'towel', 'box', 'bag', 'monitor',
             'night stand', 'dresser', 'stool', 'bathtub', 'shelves', 'curtain', 'clock', 'plant', 'computer', 'keyboard',
             'microwave', 'oven', 'stove', 'bottle', 'cup', 'bowl', 'blinds', 'whiteboard', 'tv stand', 'coffee table',
             'ottoman', 'piano', 'laptop', 'printer', 'fan', 'vase', 'speaker', 'basket']
KITTI_VOCAB = ['car', 'van', 'truck', 'pedestrian', 'cyclist', 'tram']
VOCAB = {'people': COCO80, 'indoor': NYU_VOCAB, 'street': KITTI_VOCAB}


def u16(t):
    """bf16 tensor -> uint16 numpy (exact bits; restore locally with (u16.astype(uint32) << 16).view(float32))."""
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def restore_grid(tokens, height, width, merge=2):
    """Raw patches in grouped order -> [height, width, d] in image order (see src/layout_audit.py)."""
    return tokens.reshape(height // merge, width // merge, merge, merge, -1).permute(0, 2, 1, 3, 4).reshape(height, width, -1)


def names_line(objs):
    return 'Objects: ' + '; '.join(f"the {o['name']}" for o in objs) + '.'


def box_mask(box, h, w):
    """Cells of an h x w grid whose centres fall inside a fractional box; the cell under the box centre if none."""
    ys = (torch.arange(h) + 0.5) / h; xs = (torch.arange(w) + 0.5) / w
    m = ((ys[:, None] >= box[1]) & (ys[:, None] <= box[3]) & (xs[None] >= box[0]) & (xs[None] <= box[2]))
    if not m.any():
        cy = min(int((box[1] + box[3]) / 2 * h), h - 1); cx = min(int((box[0] + box[2]) / 2 * w), w - 1)
        m[cy, cx] = True
    return m.reshape(-1)


def fast_patch_embed(pe, checks):
    """The patch embedding is a Conv3d whose kernel equals its stride, i.e. one linear map per patch. On the rented
    Blackwell GPU (torch 2.9.1, CUDA 12.8) that Conv3d ran at about 45 s per batch, so it is replaced by the same
    weights as a matrix product (tests/check_extract.py checks the two agree)."""
    conv = pe.proj
    if not (isinstance(conv, torch.nn.Conv3d) and tuple(conv.kernel_size) == tuple(conv.stride)):
        return
    def forward(hidden_states):
        W = conv.weight.reshape(conv.out_channels, -1)
        y = hidden_states.reshape(-1, W.shape[1]).to(W.dtype) @ W.T
        return y + conv.bias if conv.bias is not None else y
    pe.forward = forward
    checks['patch_embed'] = 'Conv3d (kernel == stride) computed as the equivalent matrix product'


class Runner:
    def __init__(self, model_dir, dense):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.proc.tokenizer.padding_side = 'left'
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                 device_map='cuda' if torch.cuda.is_available() else 'cpu').eval()
        cfg = self.model.config
        self.kind = cfg.model_type                      # qwen3_5 | qwen2_5_vl
        self.visual = self.model.model.visual
        self.merge = self.visual.spatial_merge_size
        self.image_token_id = cfg.image_token_id
        tok = self.proc.tokenizer
        self.image_pad = tok.convert_ids_to_tokens(self.image_token_id)
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('yes', 'Yes', 'YES')})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('no', 'No', 'NO')})
        self.checks = {}
        fast_patch_embed(self.visual.patch_embed, self.checks)
        self.n_vis = len(self.visual.blocks)
        self.n_lm = len(self.model.model.language_model.layers) + 1
        self.d_vis = self.visual.blocks[0].norm1.weight.shape[0]
        self.d_lm = self.model.model.language_model.embed_tokens.weight.shape[1]
        self.dense = dense
        g = torch.Generator().manual_seed(SEED)
        self.dense_vis_layers = sorted(set(range(0, self.n_vis, 3)) | {self.n_vis - 1})
        self.dense_lm_layers = sorted(set(range(0, self.n_lm, 4)) | {self.n_lm - 1})
        self.R_vis = (torch.randn(self.d_vis, DENSE_DIMS, generator=g) / DENSE_DIMS ** 0.5)
        self.R_lm = (torch.randn(self.d_lm, DENSE_DIMS, generator=g) / DENSE_DIMS ** 0.5)

    def chat(self, text, image=True):
        content = ([{'type': 'image'}] if image else []) + [{'type': 'text', 'text': text}]
        return self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)

    def encode(self, texts, images):
        enc = self.proc(text=texts, images=images if images else None, return_tensors='pt', padding=True)
        return enc.to(self.model.device)

    def name_positions(self, text, objs, enc_row_ids, n_img_tokens):
        """Index in the processed (expanded, left-padded) row of the last token of each "the <name>"."""
        tok = self.proc.tokenizer
        t = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = t['input_ids'], t['offset_mapping']
        start = text.index('Objects: ')
        img_at = [j for j, i in enumerate(ids) if i == self.image_token_id]
        shift = (n_img_tokens - 1) if img_at else 0
        pad = len(enc_row_ids) - (len(ids) + shift)
        out = []
        for o in objs:
            s = text.index(f"the {o['name']}", start); e = s + len(f"the {o['name']}"); start = e
            j = next(j for j, (a, b) in enumerate(offs) if a <= e - 1 < b)
            pos = pad + j + (shift if img_at and j > img_at[0] else 0)
            assert enc_row_ids[pos] == ids[j], 'name token position mismatch'
            out.append(pos)
        return out

    # -------------------------------------------------------------- hidden states for one batch
    @torch.no_grad()
    def states(self, rows, texts, images, want_image_feats):
        enc = self.encode(texts, images)
        vis = []
        hooks = []
        if want_image_feats:
            hooks = [b.register_forward_hook(lambda m, i, o: vis.append(o if torch.is_tensor(o) else o[0])) for b in self.visual.blocks]
            if 'window_order' not in self.checks and self.kind == 'qwen2_5_vl':
                first = {}
                hooks.append(self.visual.patch_embed.register_forward_hook(lambda m, i, o: first.__setitem__('embed', o)))
                hooks.append(self.visual.blocks[0].register_forward_pre_hook(lambda m, a, kw=None: first.__setitem__('block0_in', a[0] if a else kw['hidden_states']), with_kwargs=False))
        try:
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        hs = out.hidden_states                                  # tuple (LM layers + 1) of [B, T, D]
        assert len(hs) == self.n_lm
        res = {'enc': enc, 'hs': hs}
        if want_image_feats:
            thw = enc['image_grid_thw']
            unperm = None
            if self.kind == 'qwen2_5_vl':
                from transformers.vision_utils import get_vision_window_index
                wi, _ = get_vision_window_index(thw, spatial_merge_size=self.merge, window_size=self.visual.window_size,
                                                patch_size=self.visual.patch_size)
                unperm = torch.argsort(wi.to(vis[0].device))
                if 'window_order' not in self.checks:
                    b0 = first['block0_in']; d = b0.shape[-1]
                    restored = b0.reshape(-1, self.merge ** 2, d)[unperm].reshape(-1, d)
                    self.checks['window_order'] = bool(torch.equal(restored, first['embed'].reshape(-1, d)))
                    assert self.checks['window_order'], 'Qwen2.5-VL window order not undone'
            else:
                self.checks.setdefault('window_order', 'not applicable (no window attention)')
            sizes = (thw.prod(-1)).tolist()
            per_layer = []
            for x in vis:
                if unperm is not None:
                    x = x.reshape(-1, self.merge ** 2, x.shape[-1])[unperm].reshape(-1, x.shape[-1])
                per_layer.append(torch.split(x, sizes))
            res['vis'] = [[restore_grid(per_layer[l][i], int(thw[i, 1]), int(thw[i, 2]), self.merge) for l in range(self.n_vis)]
                          for i in range(len(rows))]
            res['grid'] = [(int(thw[i, 1]), int(thw[i, 2])) for i in range(len(rows))]
        return res

    # -------------------------------------------------------------- yes / no scores
    @torch.no_grad()
    def yesno(self, texts, images):
        enc = self.encode(texts, images)
        logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        y = torch.logsumexp(lp[:, self.yes_ids], -1); n = torch.logsumexp(lp[:, self.no_ids], -1)
        top = lp.topk(5, -1)
        tok = self.proc.tokenizer
        return [{'logp_yes': float(y[i]), 'logp_no': float(n[i]),
                 'top5': [[tok.decode([int(t)]), round(float(v), 3)] for t, v in zip(top.indices[i], top.values[i])]}
                for i in range(len(texts))]

    @torch.no_grad()
    def generate(self, texts, images, max_new):
        enc = self.encode(texts, images)
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        return self.proc.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)


def load_image(data, row):
    return Image.open(data/row['image']).convert('RGB')


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def derangement(rows):
    """Each image's partner: another image of the same domain (seeded)."""
    rng = random.Random(SEED)
    part = {}
    for dom in sorted({r['domain'] for r in rows}):
        ids = [r['id'] for r in rows if r['domain'] == dom]
        if len(ids) < 2:
            continue
        perm = ids[:]
        while any(a == b for a, b in zip(ids, perm)):
            rng.shuffle(perm)
        part.update(zip(ids, perm))
    return part


def yesno_items(rows):
    """Relation questions (a/b order flipped at random so the truth is balanced) and the naming check."""
    rng = random.Random(SEED)
    items = []
    by_rel = {}
    for r in rows:
        if r['domain'] == 'figure':
            continue
        if r['split'] == 'train' and r['domain'] != 'two_people':
            continue                                            # test / held-out images (two_people: all, for n)
        for p in r['pairs']:
            for rel, truth in p['relations'].items():
                if r['domain'] == 'two_people' and rel == 'left_of':
                    continue                                    # the names say it
                a, b = p['a'], p['b']
                if rng.random() < 0.5:
                    a, b, truth = b, a, not truth
                by_rel.setdefault((r['domain'], rel), []).append(
                    {'kind': 'relation', 'relation': rel, 'id': r['id'], 'a': a, 'b': b, 'truth': bool(truth),
                     'text': REL_Q[rel].format(a=r['objects'][a]['name'], b=r['objects'][b]['name'])})
    for (dom, rel), its in sorted(by_rel.items()):
        if dom == 'street' and len(its) > REL_CAP:
            its = random.Random(SEED).sample(its, REL_CAP)
        items += its
    street = [r for r in rows if r['domain'] == 'street']
    keep_street = {r['id'] for r in random.Random(SEED).sample(street, min(NAMING_STREET, len(street)))}
    for r in rows:
        if r['domain'] not in VOCAB or r['split'] == 'train' or (r['domain'] == 'street' and r['id'] not in keep_street):
            continue
        named = [o['name'] for o in r['objects'] if o['name'] != 'person'] or [o['name'] for o in r['objects']]
        present = rng.choice(named)
        absent_pool = [c for c in VOCAB[r['domain']] if c not in set(r.get('classes', [])) | {o['name'] for o in r['objects']}]
        absent = rng.choice(absent_pool)
        for name, truth in ((present, True), (absent, False)):
            items.append({'kind': 'naming', 'id': r['id'], 'name': name, 'truth': truth, 'text': NAMING_Q.format(name=name)})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--parts', default='feat,noimg,mis,ask,yesno,ground')
    ap.add_argument('--dense', action='store_true')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--gen-batch', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='first N images of each domain only (smoke tests)')
    ap.add_argument('--ask-domains', default='', help='comma list: generate answers only for these domains (E03)')
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)/a.name
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((data/'dataset.json').read_text())
    if a.limit:
        doms = sorted({r['domain'] for r in rows})
        rows = [r for d in doms for r in [x for x in rows if x['domain'] == d][:a.limit]]
    t0 = time.time()
    R = Runner(a.model_dir, a.dense)
    meta_path = out/'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    import transformers
    meta.update({'model': a.name, 'model_dir': str(a.model_dir), 'kind': R.kind, 'n_vis_layers': R.n_vis, 'n_lm_layers': R.n_lm,
                 'd_vis': R.d_vis, 'd_lm': R.d_lm, 'transformers': transformers.__version__, 'torch': torch.__version__,
                 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu', 'load_s': round(time.time() - t0, 1),
                 'yes_ids': R.yes_ids, 'no_ids': R.no_ids})
    objs = [(r['id'], k) for r in rows for k in range(len(r['objects']))]
    oidx = {o: j for j, o in enumerate(objs)}
    (out/'objects.json').write_text(json.dumps(objs))
    byid = {r['id']: r for r in rows}
    # sort by image shape so batches pad little
    order = sorted(rows, key=lambda r: (r['size'][1] / r['size'][0], r['id']))
    parts = a.parts.split(',')
    timings = meta.setdefault('timings_s', {})

    def save_meta():
        meta['checks'] = {**meta.get('checks', {}), **{k: v for k, v in R.checks.items()}}
        meta_path.write_text(json.dumps(meta, indent=1))

    # ---------------------------------------------------------------- feat / noimg / mis
    partner = derangement(rows)
    for part in ('feat', 'noimg', 'mis'):
        if part not in parts or (out/f'{part}.done').exists():
            continue
        t = time.time()
        mm = lambda nm, L, d: np.lib.format.open_memmap(out/f'{part}_{nm}.npy', 'w+', np.uint16, (len(objs), L, d))
        word = mm('word', R.n_lm, R.d_lm)
        if part == 'feat':
            vis = mm('vis', R.n_vis, R.d_vis); lmimg = mm('lmimg', R.n_lm, R.d_lm)
            grids, dense = {}, {}
        for bi, chunk in enumerate(batches([r for r in order if part != 'mis' or r['id'] in partner], a.batch)):
            texts, images = [], []
            for r in chunk:
                txt = names_line(r['objects']) + '\n' + ASK_Q
                texts.append(R.chat(txt, image=part != 'noimg'))
                if part == 'feat':
                    images.append(load_image(data, r))
                elif part == 'mis':
                    images.append(load_image(data, byid[partner[r['id']]]))
            res = R.states(chunk, texts, images, want_image_feats=(part == 'feat'))
            ids = res['enc']['input_ids']
            for i, r in enumerate(chunk):
                n_img = int((ids[i] == R.image_token_id).sum())
                pos = R.name_positions(texts[i], r['objects'], ids[i].tolist(), n_img)
                js = [oidx[(r['id'], k)] for k in range(len(r['objects']))]
                word[js] = np.stack([u16(torch.stack([h[i, p] for h in res['hs']])) for p in pos])
                if part == 'feat':
                    h, w = res['grid'][i]; hm, wm = h // R.merge, w // R.merge
                    assert n_img == hm * wm, 'image token count != merged grid'
                    img_pos = (ids[i] == R.image_token_id).nonzero()[:, 0]
                    for k, j in enumerate(js):
                        box = r['objects'][k]['box']
                        mv = box_mask(box, h, w).to(img_pos.device)
                        vis[j] = u16(torch.stack([res['vis'][i][l].reshape(h * w, -1)[mv].float().mean(0) for l in range(R.n_vis)]))
                        ml = box_mask(box, hm, wm).to(img_pos.device)
                        lmimg[j] = u16(torch.stack([hh[i, img_pos[ml]].float().mean(0) for hh in res['hs']]))
                    grids[r['id']] = [h, w]
                    if R.dense and r['domain'] == 'indoor':
                        v = torch.stack([res['vis'][i][l].reshape(hm, R.merge, wm, R.merge, -1).float().mean((1, 3)).reshape(hm * wm, -1)
                                         for l in R.dense_vis_layers])                       # [Lv, cells, d_vis]
                        lm = torch.stack([res['hs'][l][i, img_pos].float() for l in R.dense_lm_layers])
                        dense[r['id']] = (u16(v @ R.R_vis.to(v.device)), u16(lm @ R.R_lm.to(lm.device)))
            del res
            if bi % 10 == 0:
                print(part, bi * a.batch, '/', len(order), round(time.time() - t), 's', flush=True)
        word.flush()
        if part == 'feat':
            vis.flush(); lmimg.flush()
            (out/'grids.json').write_text(json.dumps(grids))
            if dense:
                np.savez(out/'dense_indoor.npz', **{f'{k}__vis': v[0] for k, v in dense.items()}, **{f'{k}__lm': v[1] for k, v in dense.items()},
                         vis_layers=np.array(R.dense_vis_layers), lm_layers=np.array(R.dense_lm_layers))
        timings[part] = round(time.time() - t, 1)
        (out/f'{part}.done').write_text('ok')
        save_meta()

    # ---------------------------------------------------------------- generated answers
    ask_rows = [r for r in order if not a.ask_domains or r['domain'] in a.ask_domains.split(',')]
    for part, q, max_new in (('ask', ASK_Q, 384), ('ground', GROUND_Q, 256), ('ask2', ASK_Q2, 384)):
        if part not in parts or (out/f'{part}.jsonl').exists():
            continue
        t = time.time()
        recs = []
        for bi, chunk in enumerate(batches(ask_rows, a.gen_batch)):
            texts = [R.chat(names_line(r['objects']) + '\n' + q) for r in chunk]
            ans = R.generate(texts, [load_image(data, r) for r in chunk], max_new)
            recs += [{'id': r['id'], 'names': [o['name'] for o in r['objects']], 'answer': s} for r, s in zip(chunk, ans)]
            if bi % 5 == 0:
                print(part, bi * a.gen_batch, '/', len(order), round(time.time() - t), 's', flush=True)
        (out/f'{part}.jsonl').write_text('\n'.join(json.dumps(x) for x in recs))
        timings[part] = round(time.time() - t, 1)
        save_meta()

    # ---------------------------------------------------------------- yes / no
    for ynpart in ('yesno', 'yesno2'):
        if ynpart not in parts or (out/f'{ynpart}.jsonl').exists():
            continue
        t = time.time()
        items = yesno_items(rows)
        if ynpart == 'yesno2':                                  # same items, relations reworded (roles swapped, same truth)
            items = [x for x in items if x['kind'] == 'relation']
            for x in items:
                byid_ = byid[x['id']]
                x['text'] = REL_Q2[x['relation']].format(a=byid_['objects'][x['a']]['name'], b=byid_['objects'][x['b']]['name'])
        items.sort(key=lambda x: (byid[x['id']]['size'][1] / byid[x['id']]['size'][0], x['id']))
        recs = []
        for bi, chunk in enumerate(batches(items, a.batch)):
            sc = R.yesno([R.chat(x['text']) for x in chunk], [load_image(data, byid[x['id']]) for x in chunk])
            recs += [{**x, **s} for x, s in zip(chunk, sc)]
            if bi % 20 == 0:
                print(ynpart, bi * a.batch, '/', len(items), round(time.time() - t), 's', flush=True)
        (out/f'{ynpart}.jsonl').write_text('\n'.join(json.dumps(x) for x in recs))
        timings[ynpart] = round(time.time() - t, 1)
        save_meta()
    meta['total_s'] = round(time.time() - t0, 1)
    save_meta()
    print('done', a.name, meta['timings_s'], flush=True)


if __name__ == '__main__':
    main()
