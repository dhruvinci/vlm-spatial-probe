"""Checks on src/extract.py with a tiny random model (CPU): the rendered prompt, the name-token positions, and that
left-padded batches give the same states as single images (padding must not leak into the readouts).

  python tests/check_extract.py --model-dir TINY --data DATA
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import extract as X  # noqa: E402


def readouts(R, rows, data, part):
    texts = [R.chat(X.names_line(r['objects']) + '\n' + X.ASK_Q, image=part != 'noimg') for r in rows]
    imgs = [X.load_image(data, r) for r in rows] if part != 'noimg' else []
    res = R.states(rows, texts, imgs, want_image_feats=part == 'feat')
    ids = res['enc']['input_ids']
    out = []
    for i, r in enumerate(rows):
        n_img = int((ids[i] == R.image_token_id).sum())
        pos = R.name_positions(texts[i], r['objects'], ids[i].tolist(), n_img)
        word = torch.stack([torch.stack([h[i, p] for h in res['hs']]) for p in pos]).float()
        o = {'word': word, 'tokens': [R.proc.tokenizer.decode([int(ids[i, p])]) for p in pos]}
        if part == 'feat':
            h, w = res['grid'][i]
            o['vis_last'] = res['vis'][i][-1].float()
            img_pos = (ids[i] == R.image_token_id).nonzero()[:, 0]
            o['lm_img'] = torch.stack([hh[i, img_pos] for hh in res['hs']]).float()
        out.append(o)
    return out, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--data', required=True)
    a = ap.parse_args()
    data = Path(a.data)
    rows = json.loads((data/'dataset.json').read_text())
    pick = [next(r for r in rows if r['domain'] == d) for d in ('people', 'street', 'two_people', 'figure')]
    R = X.Runner(a.model_dir, dense=False)
    R.model.float()
    for part in ('feat', 'noimg'):
        batch, texts = readouts(R, pick, data, part)
        print(part, 'prompt:', repr(texts[0][:400]))
        for i, r in enumerate(pick):
            single, _ = readouts(R, [r], data, part)
            s, b = single[0], batch[i]
            rel = lambda u, v: float((u - v).abs().max() / v.abs().max().clamp_min(1e-6))
            msg = {'word': rel(b['word'], s['word'])}
            if part == 'feat':
                msg['vis_last'] = rel(b['vis_last'], s['vis_last']); msg['lm_img'] = rel(b['lm_img'], s['lm_img'])
            print(' ', r['id'], 'names ->', b['tokens'], 'batch vs single max rel diff', {k: f'{v:.2e}' for k, v in msg.items()})
            assert all(v < 1e-3 for v in msg.values()), 'batched states differ from single-image states'
    print('checks', R.checks, 'yes ids', R.yes_ids, 'no ids', R.no_ids, [R.proc.tokenizer.decode([t]) for t in R.yes_ids + R.no_ids])


if __name__ == '__main__':
    main()
