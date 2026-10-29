"""E02: leave-one-domain-out decoders on the cached hidden states (experiments/E02-decoder-transfer/PREREG.md).

For each held-out domain: every layer of each readout (vision blocks, image tokens in the language layers, the object
word) is scored by inner folds over the training domains only (leave one training domain out when there are two or
more, else 5 folds by image). The best layer and ridge alpha are refitted on all training domains; a small MLP is
trained on that layer; the head with the lower inner error is kept. Controls at the word's chosen layer: the word with
no image / a mismatched image, shuffled labels, norm-matched random features. Writes cache/e02_<model>.json.

  python src/transfer.py --cache CACHE --model Qwen3.8-27B [--device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe import ALPHAS, KRidge, f32, gram

SEED = 2604
GROUPS = {                                   # name: (domains with truth, targets fitted together)
    'geo': (('indoor', 'two_people', 'street'), ('depth', 'centre')),
    'size': (('indoor', 'street'), ('size',)),
    'xy': (('people', 'indoor', 'two_people', 'street'), ('xy',)),
    'orient': (('street',), ('orient',)),    # within street only: 5 folds by image (not a transfer test)
}


def records(rows, orient):
    recs = []
    for r in rows:
        for k, o in enumerate(r['objects']):
            ori = (orient.get(r['id']) or [None] * 99)[k] if r['domain'] == 'street' else None
            recs.append({'id': r['id'], 'k': k, 'domain': r['domain'], 'activity': r.get('activity'), 'name': o['name'],
                         'xy': o['xy'], 'size_img': r['size'], 'depth': o.get('depth_m'), 'size': o.get('size_m'),
                         'centre': o.get('centre_m'), 'alpha': ori['alpha'] if ori else None})
    return recs


def targets(recs, t):
    if t == 'depth':
        return np.array([[np.log(o['depth'])] if o['depth'] else [np.nan] for o in recs])
    if t == 'size':
        return np.array([[np.log(o['size'])] if o['size'] else [np.nan] for o in recs])
    if t == 'centre':
        return np.array([o['centre'] if o['centre'] else [np.nan] * 3 for o in recs], float)
    if t == 'xy':
        return np.array([o['xy'] for o in recs], float)
    return np.array([[np.sin(o['alpha']), np.cos(o['alpha'])] if o['alpha'] is not None else [np.nan] * 2 for o in recs])


def errors(t, pred, recs, idx):
    """Per-object error (lower is better)."""
    pred = np.asarray(pred, float)
    if t in ('depth', 'size'):
        y = np.array([recs[i][t] for i in idx], float)
        return np.abs(np.exp(pred[:, 0]) - y) / y
    if t == 'centre':
        y = np.array([recs[i]['centre'] for i in idx], float)
        return np.linalg.norm(pred - y, axis=1) / np.linalg.norm(y, axis=1)
    if t == 'xy':
        W = np.array([recs[i]['size_img'] for i in idx], float)
        d = (pred - np.array([recs[i]['xy'] for i in idx])) * W
        return np.hypot(d[:, 0], d[:, 1]) / np.hypot(W[:, 0], W[:, 1]) * 100
    a = np.array([recs[i]['alpha'] for i in idx])
    p = np.arctan2(pred[:, 0], pred[:, 1])
    return np.degrees(np.abs(np.angle(np.exp(1j * (p - a)))))


def folds_for(group, recs, ids_fold):
    doms, _ = GROUPS[group]
    if group == 'orient':
        return [(f'street_fold{f}', [d for d in doms], f) for f in range(5)]
    return [(h, [d for d in doms if d != h], None) for h in doms]


def split(group, recs, ok, held, train_doms, fold5, ids_fold):
    dom = np.array([o['domain'] for o in recs])
    if group == 'orient':
        f_of = np.array([ids_fold[o['id']] for o in recs])
        tr = np.where(ok & (dom == 'street') & (f_of != fold5))[0]; te = np.where(ok & (dom == 'street') & (f_of == fold5))[0]
        inner = [(tr[f_of[tr] != g], tr[f_of[tr] == g]) for g in range(5) if g != fold5]
        return tr, te, inner
    tr = np.where(ok & np.isin(dom, train_doms))[0]; te = np.where(ok & (dom == held))[0]
    if len(train_doms) >= 2:
        inner = [(tr[dom[tr] != d], tr[dom[tr] == d]) for d in train_doms]
    else:
        f_of = np.array([ids_fold[recs[i]['id']] for i in tr])
        inner = [(tr[f_of != g], tr[f_of == g]) for g in range(5)]
    return tr, te, inner


def mlp_fit_predict(Xtr, Ytr, Xte, dev, steps=400, seed=SEED):
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    m, s = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = (Xtr - m) / s; Xte = (Xte - m) / s
    Y = torch.as_tensor(Ytr, dtype=torch.float32, device=dev); ym, ys = Y.mean(0), Y.std(0).clamp_min(1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(Xtr) - (Y - ym) / ys) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return (net(Xte) * ys + ym).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    cache = Path(a.cache); dev = a.device
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure']
    orient = json.loads((cache/'e02_orientation.json').read_text())['objects']
    recs = records(rows, orient)
    mdir = cache/'out'/a.model
    oidx = {tuple(o): j for j, o in enumerate(json.loads((mdir/'objects.json').read_text()))}
    sel = np.array([oidx[(o['id'], o['k'])] for o in recs])
    meta = json.loads((mdir/'meta.json').read_text())
    n_lm = meta['n_lm_layers']
    ids = sorted({o['id'] for o in recs})
    ids_fold = dict(zip(ids, np.random.default_rng(SEED).integers(0, 5, len(ids))))
    feats = {n: np.load(mdir/f'{p}_{q}.npy', mmap_mode='r') for n, (p, q) in
             {'vis': ('feat', 'vis'), 'lmimg': ('feat', 'lmimg'), 'word': ('feat', 'word'), 'word_noimg': ('noimg', 'word'),
              'word_mis': ('mis', 'word')}.items()}
    fam = {'V': ('vis', list(range(meta['n_vis_layers']))), 'L_img': ('lmimg', list(range(int(np.ceil(0.75 * (n_lm - 1))), n_lm))),
           'L_img_any': ('lmimg', list(range(n_lm))), 'W': ('word', list(range(1, n_lm)))}
    t0 = time.time()
    # ---- every fold's split and inner folds
    plan = {}
    for g, (doms, tg) in GROUPS.items():
        Y = np.concatenate([targets(recs, t) for t in tg], 1)
        ok = ~np.isnan(Y).any(1)
        for held, train_doms, f5 in folds_for(g, recs, ids_fold):
            tr, te, inner = split(g, recs, ok, held, train_doms, f5, ids_fold)
            plan[(g, held)] = {'Y': Y, 'tr': tr, 'te': te, 'inner': inner, 'targets': tg,
                               'cols': np.cumsum([0] + [targets(recs[:1], t).shape[1] for t in tg])}
    # ---- inner scores for every (readout, layer, fold, target, alpha)
    inner_scores = {}
    for name in ('vis', 'lmimg', 'word'):
        for l in range(feats[name].shape[1]):
            X = f32(feats[name][sel, l])
            for key, P in plan.items():
                train_mask = np.zeros(len(recs), bool); train_mask[P['tr']] = True
                kr = KRidge(gram(X, train_mask, dev), dev)
                sc = {t: np.zeros(len(ALPHAS)) for t in P['targets']}; n = 0
                for itr, iva in P['inner']:
                    preds = kr.fit_predict(itr, iva, P['Y'], ALPHAS)
                    for j, pr in enumerate(preds):
                        for c, t in enumerate(P['targets']):
                            sc[t][j] += errors(t, pr[:, P['cols'][c]:P['cols'][c + 1]], recs, iva).sum()
                    n += len(iva)
                for t in P['targets']:
                    inner_scores[(name, l, key, t)] = (sc[t] / n).tolist()
            print(a.model, name, l, round(time.time() - t0), 's', flush=True)
    # ---- choose layer + alpha per family, refit, MLP, controls
    out = {'model': a.model, 'records': [{k: o[k] for k in ('id', 'k', 'domain', 'activity', 'name')} for o in recs], 'folds': {}}

    def ridge_fixed(X, P, t, alpha_from=None, shuffle=None):
        """Ridge at a fixed feature matrix: alpha by the same inner folds (unless given), refit on all training rows."""
        train_mask = np.zeros(len(recs), bool); train_mask[P['tr']] = True
        kr = KRidge(gram(X, train_mask, dev), dev)
        c = P['targets'].index(t); Y = P['Y'][:, P['cols'][c]:P['cols'][c + 1]].copy()
        if shuffle is not None:
            Y[P['tr']] = Y[P['tr']][shuffle(len(P['tr']))]
        if alpha_from is None:
            sc = np.zeros(len(ALPHAS))
            for itr, iva in P['inner']:
                for j, pr in enumerate(kr.fit_predict(itr, iva, Y, ALPHAS)):
                    sc[j] += errors(t, pr, recs, iva).sum()
            alpha_from = ALPHAS[int(np.argmin(sc))]
        return kr.fit_predict(P['tr'], P['te'], Y, [alpha_from])[0], alpha_from

    for key, P in plan.items():
        g, held = key
        F = {'test_idx': P['te'].tolist(), 'train_n': len(P['tr']), 'targets': {}}
        for t in P['targets']:
            c = P['targets'].index(t); Yt = P['Y'][:, P['cols'][c]:P['cols'][c + 1]]
            T = {}
            for f, (name, layers) in fam.items():
                best = min(((l, j) for l in layers for j in range(len(ALPHAS))), key=lambda lj: inner_scores[(name, lj[0], key, t)][lj[1]])
                l, j = best
                X = f32(feats[name][sel, l])
                rp, _ = ridge_fixed(X, P, t, alpha_from=ALPHAS[j])
                rec = {'layer': int(l), 'alpha': ALPHAS[j], 'inner_ridge': inner_scores[(name, l, key, t)][j], 'ridge': rp.tolist()}
                if f in ('V', 'L_img', 'W'):
                    sc = 0.0; n = 0
                    for itr, iva in P['inner']:
                        sc += errors(t, mlp_fit_predict(X[itr], Yt[itr], X[iva], dev), recs, iva).sum(); n += len(iva)
                    rec['inner_mlp'] = sc / n
                    rec['mlp'] = mlp_fit_predict(X[P['tr']], Yt[P['tr']], X[P['te']], dev).tolist()
                    rec['head'] = 'mlp' if rec['inner_mlp'] < rec['inner_ridge'] else 'ridge'
                T[f] = rec
            lw = T['W']['layer']
            ctrl = {}
            for cname in ('word_noimg', 'word_mis'):
                p_, al = ridge_fixed(f32(feats[cname][sel, lw]), P, t)
                ctrl[cname] = {'alpha': al, 'pred': p_.tolist()}
            Xw = f32(feats['word'][sel, lw])
            sh = []
            for s in range(5):
                gg = np.random.default_rng(SEED + 1 + s)
                sh.append(ridge_fixed(Xw, P, t, shuffle=lambda n: gg.permutation(n))[0].tolist())
            ctrl['shuffled'] = {'preds': sh}
            Rn = np.random.default_rng(SEED + 100).standard_normal(Xw.shape).astype(np.float32)
            Rn *= (np.linalg.norm(Xw, axis=1) / np.linalg.norm(Rn, axis=1))[:, None]
            ctrl['random'] = {'pred': ridge_fixed(Rn, P, t)[0].tolist()}
            T['controls'] = ctrl
            F['targets'][t] = T
            print(a.model, key, t, {f: (T[f]['layer'], T[f].get('head')) for f in fam}, round(time.time() - t0), 's', flush=True)
        out['folds'][f'{g}|{held}'] = F
    (cache/f'e02_{a.model}.json').write_text(json.dumps(out))
    print('saved', cache/f'e02_{a.model}.json', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
