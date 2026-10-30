"""E04 probes (experiments/E04-more-measures/PREREG.md), E01's procedure on E04's targets: one ridge per readout, layer
and target on standardised features; alpha 10^0-10^5 by 5-fold cross-validation grouped like the split, on the training
split; layer families chosen on training only (V, L_img = last quarter, L_img_any, W, W_noimg, W_mis); controls on the
selected V and W layers (shuffled labels x 5, norm-matched random features). Writes CACHE/probe_e04_<model>.json.

Targets: facing (sin, cos); joints (elbow and knee angles, degrees); normal (unit 3-vector); surface_class (one-hot,
argmax); dims (log, sorted); speed (m/s); heading (sin, cos; moving objects); count (log(1 + count)).

  python src/probe_e04.py --data DATA_E04 --feats OUT/e04_<model> --out CACHE [--device cuda]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from probe import ALPHAS, KRidge, f32, gram

SEED, FOLDS = 2604, 5
CLASSES = ['up', 'towards', 'sideways']
TARGETS = {
    'facing': ('orientation', lambda o: [math.sin(math.radians(o['facing_deg'])), math.cos(math.radians(o['facing_deg']))] if o.get('facing_deg') is not None else None, 'angle2'),
    'joints': ('orientation', lambda o: [o[k] for k in ('elbow_l_deg', 'elbow_r_deg', 'knee_l_deg', 'knee_r_deg')] if o.get('elbow_l_deg') is not None else None, 'deg'),
    'normal': ('surfaces', lambda o: list(o['normal_cam']) if o.get('normal_cam') else None, 'normal'),
    'surface_class': ('surfaces', lambda o: [float(o['surface_class'] == c) for c in CLASSES] if o.get('surface_class') in CLASSES else None, 'class'),
    'dims': (None, lambda o: [math.log(max(d, 1e-3)) for d in sorted(o['dims_m'], reverse=True)] if o.get('dims_m') else None, 'logdims'),
    'speed': ('motion', lambda o: [o['speed_mps']] if o.get('speed_mps') is not None else None, 'abs'),
    'heading': ('motion', lambda o: [math.sin(math.radians(o['heading_deg'])), math.cos(math.radians(o['heading_deg']))] if o.get('moving') and o.get('heading_deg') is not None else None, 'angle2'),
    'count': ('counts', lambda o: [math.log1p(o['count'])] if o.get('count') is not None else None, 'count'),
}


def records(rows):
    recs = []
    for r in rows:
        for k, o in enumerate(r['objects']):
            recs.append({'id': r['id'], 'k': k, 'measure': r['measure'], 'domain': r['domain'], 'split': r['split'],
                         'group': r['group'], 'name': o['name'], 'o': o})
    return recs


def target(recs, t):
    meas, fn, _ = TARGETS[t]
    Y = [fn(r['o']) if (meas is None or r['measure'] == meas) else None for r in recs]
    dim = next((len(y) for y in Y if y is not None), None)
    if dim is None:
        return None
    ok = np.array([y is not None for y in Y])
    return np.array([y if y is not None else [np.nan] * dim for y in Y], float), ok


def err(kind, pred, Y):
    """Per-item error, lower is better (class: 1 - correct)."""
    pred, Y = np.asarray(pred, float), np.asarray(Y, float)
    if kind == 'angle2':
        a, b = np.arctan2(pred[:, 0], pred[:, 1]), np.arctan2(Y[:, 0], Y[:, 1])
        return np.degrees(np.abs(np.angle(np.exp(1j * (a - b)))))
    if kind == 'normal':
        p = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9)
        return np.degrees(np.arccos(np.clip((p * Y).sum(1), -1, 1)))
    if kind == 'class':
        return (pred.argmax(1) != Y.argmax(1)).astype(float)
    if kind == 'logdims':
        return (np.abs(np.exp(pred) - np.exp(Y)) / np.exp(Y)).mean(1)
    if kind == 'count':
        return np.abs(np.expm1(pred[:, 0]) - np.expm1(Y[:, 0]))
    return np.abs(pred - Y).mean(1)                                    # deg, abs


def cv_score(kind, pred, Y):
    e = err(kind, pred, Y)
    if kind == 'class':                                                # 1 - balanced accuracy
        c = Y.argmax(1)
        return float(1 - np.mean([1 - e[c == k].mean() for k in np.unique(c)]))
    return float(e.mean())


def probe_layer(K, recs, targets, fold, dev, shuffle=None):
    kr = KRidge(K, dev)
    res = {}
    for t, (Y, ok) in targets.items():
        kind = TARGETS[t][2]
        split = np.array([r['split'] for r in recs])
        tr = np.where(ok & (split == 'train'))[0]; te = np.where(ok & (split == 'test'))[0]
        Yt = Y.copy()
        if shuffle is not None:
            Yt[tr] = Yt[tr][shuffle(len(tr))]
        g = np.array([fold[recs[i]['group']] for i in tr])
        cv = np.zeros(len(ALPHAS))
        for f in range(FOLDS):
            a_tr, a_te = tr[g != f], tr[g == f]
            if len(a_te) == 0 or len(a_tr) < 5:
                continue
            for j, pr in enumerate(kr.fit_predict(a_tr, a_te, Yt, ALPHAS)):
                cv[j] += cv_score(kind, pr, Yt[a_te]) * len(a_te)
        j = int(np.argmin(cv))
        pred = kr.fit_predict(tr, te, Yt, [ALPHAS[j]])[0]
        res[t] = {'alpha': ALPHAS[j], 'cv': float(cv[j] / max(len(tr), 1)), 'test_idx': te.tolist(), 'test_pred': pred.tolist()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    rows = json.loads((Path(a.data)/'dataset.json').read_text())
    mdir = Path(a.feats)
    objs = json.loads((mdir/'objects.json').read_text())
    oidx = {tuple(o): j for j, o in enumerate(objs)}
    recs = records(rows)
    sel = np.array([oidx[(r['id'], r['k'])] for r in recs])
    groups = sorted({r['group'] for r in recs})
    fold = dict(zip(groups, np.random.default_rng(SEED).integers(0, FOLDS, len(groups))))
    targets = {t: v for t in TARGETS if (v := target(recs, t)) is not None}
    meta = json.loads((mdir/'meta.json').read_text())
    train_mask = np.array([r['split'] == 'train' for r in recs])
    readouts = {'vis': ('feat', 'vis'), 'lmimg': ('feat', 'lmimg'), 'word': ('feat', 'word'), 'word_noimg': ('noimg', 'word'),
                'word_mis': ('mis', 'word')}
    out = {'model': mdir.name, 'meta': meta, 'records': [{k: r[k] for k in ('id', 'k', 'measure', 'domain', 'split', 'group', 'name')} for r in recs],
           'targets': {t: {'n_train': int((ok & train_mask).sum()), 'n_test': int((ok & ~train_mask).sum())} for t, (_, ok) in targets.items()},
           'layers': {}, 'selected': {}, 'controls': {}}
    t0 = time.time()
    feats = {}
    for name, (part, nm) in readouts.items():
        p = mdir/f'{part}_{nm}.npy'
        if not p.exists():
            continue
        feats[name] = np.load(p, mmap_mode='r')
        out['layers'][name] = []
        for l in range(feats[name].shape[1]):
            K = gram(f32(feats[name][sel, l]), train_mask, a.device)
            r = probe_layer(K, recs, targets, fold, a.device)
            out['layers'][name].append({t: {'alpha': v['alpha'], 'cv': v['cv'], 'test': v['test_pred'], 'test_idx': v['test_idx']} for t, v in r.items()})
            print(mdir.name, name, l, {t: round(v['cv'], 3) for t, v in r.items()}, round(time.time() - t0), 's', flush=True)
    n_lm = meta['n_lm_layers']
    late = range(int(np.ceil(0.75 * (n_lm - 1))), n_lm)
    fam = {'V': ('vis', range(meta['n_vis_layers'])), 'L_img': ('lmimg', late), 'L_img_any': ('lmimg', range(n_lm)),
           'W': ('word', range(1, n_lm)), 'W_noimg': ('word_noimg', range(1, n_lm)), 'W_mis': ('word_mis', range(1, n_lm))}
    for f, (name, rng_l) in fam.items():
        if name not in out['layers']:
            continue
        out['selected'][f] = {}
        for t in targets:
            best = min(rng_l, key=lambda l: out['layers'][name][l][t]['cv'])
            L = out['layers'][name][best][t]
            out['selected'][f][t] = {'layer': int(best), 'readout': name, 'alpha': L['alpha'], 'cv': L['cv'], 'test_idx': L['test_idx'], 'test_pred': L['test']}
    for f in ('V', 'W'):
        if f not in out['selected']:
            continue
        out['controls'][f] = {}
        for t in targets:
            name, l = out['selected'][f][t]['readout'], out['selected'][f][t]['layer']
            X = f32(feats[name][sel, l])
            K = gram(X, train_mask, a.device)
            perms = []
            for s in range(5):
                g = np.random.default_rng(SEED + 1 + s)
                perms.append(probe_layer(K, recs, {t: targets[t]}, fold, a.device, shuffle=lambda n: g.permutation(n))[t]['test_pred'])
            Rn = np.random.default_rng(SEED + 100).standard_normal(X.shape).astype(np.float32)
            Rn *= (np.linalg.norm(X, axis=1) / np.linalg.norm(Rn, axis=1))[:, None]
            r = probe_layer(gram(Rn, train_mask, a.device), recs, {t: targets[t]}, fold, a.device)[t]
            out['controls'][f][t] = {'layer': l, 'shuffled_preds': perms, 'random': r['test_pred'], 'test_idx': r['test_idx']}
        print('controls', f, round(time.time() - t0), 's', flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/f'probe_e04_{mdir.name}.json').write_text(json.dumps(out))
    print('saved', Path(a.out)/f'probe_e04_{mdir.name}.json', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
