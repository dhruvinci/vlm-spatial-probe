"""E04 extraction (GPU; experiments/E04-more-measures/PREREG.md): the measures E01 did not cover. Same model runner and
readouts as E01 (`extract.py`), with each measure's own question:
  feat      the image(s) + "Objects: the <name>." + the measure's question. Saved per object, bf16, every layer:
              vis    [n_obj, vision blocks, d_vis]   patches inside the object's box (motion: the last frame's box;
                                                     counts: the whole image), mean-pooled
              lmimg  [n_obj, LM layers + 1, d]       merged image tokens inside the box (motion: the last frame's)
              word   [n_obj, LM layers + 1, d]       the state at the last token of "the <name>" ("the cars" for counts)
  noimg     the same text with no image: word states
  mis       the same text with the image(s) of another item of the same source: word states
  ask       the measure's question answered as JSON (greedy, no thinking, 512 new tokens)
Motion items are four frames given as four images in one message, before the text (batched four items at a time).

  python src/extract_e04.py --model-dir M --name NAME --data DATA_E04 --out OUT [--parts feat,noimg,mis,ask]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from extract import Runner, batches, box_mask, derangement, names_line, u16

Q = {'orientation': ('For each object, give the direction it is facing as an angle in degrees, seen from above: 0 means it '
                     'faces the camera, 90 means it faces the right side of the image, 180 means it faces away from the '
                     'camera, 270 means it faces the left side of the image. For a person, use the direction their chest '
                     'faces, and also give the angle at each elbow and knee in degrees (180 = straight). Reply with JSON '
                     'only, in this form: {"objects": [{"name": <name>, "facing_deg": <number>, "left_elbow_deg": <number '
                     'or null>, "right_elbow_deg": <number or null>, "left_knee_deg": <number or null>, "right_knee_deg": '
                     '<number or null>}]}'),
     'surfaces': ('For each object, which way does its main flat surface face: "up", "towards the camera", or "sideways"? '
                  'Also give its three dimensions (length, width and height) in metres. Reply with JSON only, in this '
                  'form: {"objects": [{"name": <name>, "surface": "up" | "towards the camera" | "sideways", "dims_m": '
                  '[<number>, <number>, <number>]}]}'),
     'motion': ('These four images are frames from one video, in order, {dt} seconds apart. For each object, estimate how '
                'fast it is moving over the ground in metres per second, and the direction it is moving relative to the '
                'camera: "towards", "away", "left", "right", or "still". Reply with JSON only, in this form: {{"objects": '
                '[{{"name": <name>, "speed_mps": <number>, "direction": "towards" | "away" | "left" | "right" | '
                '"still"}}]}}'),
     'counts': ('How many of each listed object are in this image? Reply with JSON only, in this form: {"objects": '
                '[{"name": <name>, "count": <number>}]}')}
MAX_NEW = 512


def question(r):
    q = Q[r['measure']]
    if r['measure'] == 'motion':
        dt = r['objects'][0].get('dt_s', 0.9) / 3
        q = q.format(dt=f'{dt:.1f}')
    return q


def images_of(data, r):
    return [Image.open(data/p).convert('RGB') for p in (r['frames'] if 'frames' in r else [r['image']])]


def chat(R, text, n_images):
    content = [{'type': 'image'}] * n_images + [{'type': 'text', 'text': text}]
    return R.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)


def name_positions(R, text, objs, enc_row_ids, n_img_tokens):
    """As Runner.name_positions, for any number of images: each image placeholder expands to its own token count."""
    tok = R.proc.tokenizer
    t = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = t['input_ids'], t['offset_mapping']
    start = text.index('Objects: ')
    img_at = [j for j, i in enumerate(ids) if i == R.image_token_id]
    shift = n_img_tokens - len(img_at)
    pad = len(enc_row_ids) - (len(ids) + shift)
    out = []
    for o in objs:
        s = text.index(f"the {o['name']}", start); e = s + len(f"the {o['name']}"); start = e
        j = next(j for j, (a, b) in enumerate(offs) if a <= e - 1 < b)
        pos = pad + j + (shift if img_at and j > img_at[-1] else 0)
        assert enc_row_ids[pos] == ids[j], 'name token position mismatch'
        out.append(pos)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--parts', default='feat,noimg,mis,ask')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--gen-batch', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='first N items of each source only (smoke tests)')
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)/a.name
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((data/'dataset.json').read_text())
    if a.limit:
        doms = sorted({r['domain'] for r in rows})
        rows = [r for d in doms for r in [x for x in rows if x['domain'] == d][:a.limit]]
    t0 = time.time()
    R = Runner(a.model_dir, dense=False)
    meta_path = out/'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    import transformers
    meta.update({'model': a.name, 'kind': R.kind, 'n_vis_layers': R.n_vis, 'n_lm_layers': R.n_lm, 'd_vis': R.d_vis, 'd_lm': R.d_lm,
                 'transformers': transformers.__version__, 'torch': torch.__version__,
                 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu', 'load_s': round(time.time() - t0, 1)})
    objs = [(r['id'], k) for r in rows for k in range(len(r['objects']))]
    oidx = {o: j for j, o in enumerate(objs)}
    (out/'objects.json').write_text(json.dumps(objs))
    byid = {r['id']: r for r in rows}
    partner = derangement(rows)
    timings = meta.setdefault('timings_s', {})

    def save_meta():
        meta['checks'] = {**meta.get('checks', {}), **R.checks}
        meta_path.write_text(json.dumps(meta, indent=1))
    single = sorted([r for r in rows if 'frames' not in r], key=lambda r: (r['size'][1] / r['size'][0], r['id']))
    multi = [r for r in rows if 'frames' in r]
    for part in ('feat', 'noimg', 'mis'):
        if part not in a.parts.split(',') or (out/f'{part}.done').exists():
            continue
        t = time.time()
        mm = lambda nm, L, d: np.lib.format.open_memmap(out/f'{part}_{nm}.npy', 'w+', np.uint16, (len(objs), L, d))
        word = mm('word', R.n_lm, R.d_lm)
        if part == 'feat':
            vis = mm('vis', R.n_vis, R.d_vis); lmimg = mm('lmimg', R.n_lm, R.d_lm)
            grids = {}
        chunks = [c for c in batches([r for r in single if part != 'mis' or r['id'] in partner], a.batch)]
        chunks += [c for c in batches([r for r in multi if part != 'mis' or r['id'] in partner], max(1, a.batch // 4))]   # 4 frames each
        for bi, chunk in enumerate(chunks):
            texts, images, n_per = [], [], []
            for r in chunk:
                src = byid[partner[r['id']]] if part == 'mis' else r
                ims = [] if part == 'noimg' else images_of(data, src)
                texts.append(chat(R, names_line(r['objects']) + '\n' + question(r), len(ims)))
                images += ims; n_per.append(len(ims))
            feat_rows = [r for r, n in zip(chunk, n_per) for _ in range(max(n, 1))] if part == 'feat' else chunk
            res = R.states(feat_rows, texts, images, want_image_feats=(part == 'feat'))
            ids = res['enc']['input_ids']
            img_i = 0
            for i, r in enumerate(chunk):
                n_img = int((ids[i] == R.image_token_id).sum())
                pos = name_positions(R, texts[i], r['objects'], ids[i].tolist(), n_img)
                js = [oidx[(r['id'], k)] for k in range(len(r['objects']))]
                word[js] = np.stack([u16(torch.stack([h[i, p] for h in res['hs']])) for p in pos])
                if part == 'feat':
                    last = img_i + n_per[i] - 1                                                 # the last (or only) image
                    h, w = res['grid'][last]; hm, wm = h // R.merge, w // R.merge
                    img_pos = (ids[i] == R.image_token_id).nonzero()[:, 0][-hm * wm:]           # the last image's tokens
                    for k, j in enumerate(js):
                        box = r['objects'][k]['box']
                        mv = box_mask(box, h, w).to(img_pos.device)
                        vis[j] = u16(torch.stack([res['vis'][last][l].reshape(h * w, -1)[mv].float().mean(0) for l in range(R.n_vis)]))
                        ml = box_mask(box, hm, wm).to(img_pos.device)
                        lmimg[j] = u16(torch.stack([hh[i, img_pos[ml]].float().mean(0) for hh in res['hs']]))
                    grids[r['id']] = [h, w]
                img_i += n_per[i]
            del res
            if bi % 20 == 0:
                print(part, bi, '/', len(chunks), round(time.time() - t), 's', flush=True)
        word.flush()
        if part == 'feat':
            vis.flush(); lmimg.flush()
            (out/'grids.json').write_text(json.dumps(grids))
        timings[part] = round(time.time() - t, 1)
        (out/f'{part}.done').write_text('ok')
        save_meta()
    if 'ask' in a.parts.split(',') and not (out/'ask.jsonl').exists():
        t = time.time()
        recs = []
        chunks = [c for c in batches(single, a.gen_batch)] + [c for c in batches(multi, max(1, a.gen_batch // 4))]
        for bi, chunk in enumerate(chunks):
            texts, images = [], []
            for r in chunk:
                ims = images_of(data, r)
                texts.append(chat(R, names_line(r['objects']) + '\n' + question(r), len(ims))); images += ims
            ans = R.generate(texts, images, MAX_NEW)
            recs += [{'id': r['id'], 'measure': r['measure'], 'names': [o['name'] for o in r['objects']], 'answer': s} for r, s in zip(chunk, ans)]
            if bi % 10 == 0:
                print('ask', bi, '/', len(chunks), round(time.time() - t), 's', flush=True)
        (out/'ask.jsonl').write_text('\n'.join(json.dumps(x) for x in recs))
        timings['ask'] = round(time.time() - t, 1)
        save_meta()
    meta['total_s'] = round(time.time() - t0, 1)
    save_meta()
    print('done', a.name, meta['timings_s'], flush=True)


if __name__ == '__main__':
    main()
