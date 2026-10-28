"""Linear probes on the extracted features, layer by layer, with the pre-registered controls (E01 PREREG.md).

For each model, readout (vision tower; image tokens; object word with image / no image / mismatched image) and layer:
ridge regression on standardised features, alpha chosen by 5-fold cross-validation grouped by image on the training
split (kernel form on the GPU if there is one). Targets: x, y (fractions), log depth, log size, 3D centre (m).
Writes, per model, probe_<model>.json: every layer's cross-validated and test scores, and the test predictions of each
readout family's selected layer; the controls (shuffled labels, norm-matched random features) on the selected layers.
Scoring, intervals and gates are in analyse.py.

  python src/probe.py --cache CACHE --model Qwen3.8-27B [--device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SEED = 2604
ALPHAS = [10.0 ** k for k in range(0, 6)]
FOLDS = 5
TARGETS = {                                       # target: (training domains, kind)
    'xy': (('people', 'indoor', 'two_people'), 'xy'),
    'depth': (('indoor', 'two_people'), 'log'),
    'centre': (('indoor', 'two_people'), 'xyz'),
    'size': (('indoor',), 'log'),
}


def f32(u):
    """uint16 bf16 bits -> float32."""
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def table(rows):
    """One record per object: ids, domain, split, image, truth."""
    recs = []
    for r in rows:
        for k, o in enumerate(r['objects']):
            recs.append({'id': r['id'], 'k': k, 'domain': r['domain'], 'split': r['split'], 'name': o['name'],
                         'xy': o['xy'], 'box': o['box'], 'size_img': r['size'], 'depth': o.get('depth_m'),
                         'size': o.get('size_m'), 'centre': o.get('centre_m')})
    return recs


def target_matrix(recs, t):
    if t == 'xy':
        return np.array([o['xy'] for o in recs], float)
    if t == 'depth':
        return np.array([[np.log(o['depth'])] if o['depth'] else [np.nan] for o in recs], float)
    if t == 'size':
        return np.array([[np.log(o['size'])] if o['size'] else [np.nan] for o in recs], float)
    return np.array([o['centre'] if o['centre'] else [np.nan] * 3 for o in recs], float)


def cv_score(t, pred, y, recs_idx, recs):
    """Lower is better: centre error (% diagonal) for xy, abs-rel for log targets, mean 3D error (m) for centre."""
    if t == 'xy':
        W = np.array([recs[i]['size_img'] for i in recs_idx], float)
        d = (pred - y) * W
        return float(np.mean(np.hypot(d[:, 0], d[:, 1]) / np.hypot(W[:, 0], W[:, 1]) * 100))
    if t in ('depth', 'size'):
        return float(np.mean(np.abs(np.exp(pred[:, 0]) - np.exp(y[:, 0])) / np.exp(y[:, 0])))
    return float(np.mean(np.linalg.norm(pred - y, axis=1)))


class KRidge:
    """Kernel ridge on a fixed Gram matrix; intercept = training mean of y."""
    def __init__(self, K, dev):
        self.K = K
        self.dev = dev

    def fit_predict(self, tr, te, y, alphas):
        Ktt = self.K[np.ix_(tr, tr)] if isinstance(self.K, np.ndarray) else self.K[tr][:, tr]
        Ket = self.K[np.ix_(te, tr)] if isinstance(self.K, np.ndarray) else self.K[te][:, tr]
        try:
            lam, Q = torch.linalg.eigh(Ktt.double() if self.dev == 'cpu' else Ktt)
        except torch.linalg.LinAlgError:                       # ill-conditioned in float32 on the GPU: redo in float64
            lam, Q = torch.linalg.eigh(Ktt.double().cpu())
            Ket = Ket.cpu()
        lam = lam.clamp_min(0)
        mu = y[tr].mean(0)
        yt = torch.as_tensor(y[tr] - mu, dtype=Q.dtype, device=Q.device)
        Qy = Q.T @ yt
        out = []
        for a in alphas:
            beta = Q @ (Qy / (lam + a)[:, None])
            out.append((Ket.to(Q.dtype) @ beta).cpu().numpy() + mu)
        return out


def gram(X, train_mask, dev):
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    m = Xt[torch.as_tensor(train_mask, device=dev)].mean(0)
    s = Xt[torch.as_tensor(train_mask, device=dev)].std(0).clamp_min(1e-6)
    Z = (Xt - m) / s
    Z = Z / Z.shape[1] ** 0.5
    return Z @ Z.T


def probe_layer(X, recs, splits, dev, rng_groups, targets=TARGETS, shuffle=None):
    """All targets for one feature matrix. Returns {target: {'alpha', 'cv', 'test_idx', 'test_pred'}}. Targets with the
    same training domains share one eigendecomposition per fold."""
    train_all = np.array([o['split'] == 'train' for o in recs])
    K = gram(X, train_all, dev)
    kr = KRidge(K, dev)
    doms_of = np.array([o['domain'] for o in recs])
    split_of = np.array([o['split'] for o in recs])
    res = {}
    groups_by_doms = {}
    for t, (doms, _) in targets.items():
        groups_by_doms.setdefault(doms, []).append(t)
    for doms, ts in groups_by_doms.items():
        ys = {t: target_matrix(recs, t) for t in ts}
        ok = np.all([~np.isnan(ys[t]).any(1) for t in ts], 0)
        tr = np.where(train_all & ok & np.isin(doms_of, doms))[0]
        te = np.where(np.isin(split_of, ['test', 'heldout']) & ok)[0]
        Y = np.concatenate([ys[t] for t in ts], 1)
        cols = np.cumsum([0] + [ys[t].shape[1] for t in ts])
        if shuffle is not None:
            Y = Y.copy()
            Y[tr] = Y[tr][shuffle(len(tr))]
        groups = np.array([rng_groups[recs[i]['id']] for i in tr])
        cv = {t: np.zeros(len(ALPHAS)) for t in ts}
        for f in range(FOLDS):
            a_tr, a_te = tr[groups != f], tr[groups == f]
            for j, pr in enumerate(kr.fit_predict(a_tr, a_te, Y, ALPHAS)):
                for n, t in enumerate(ts):
                    cv[t][j] += cv_score(t, pr[:, cols[n]:cols[n + 1]], Y[a_te, cols[n]:cols[n + 1]], a_te, recs) * len(a_te)
        best = {t: int(np.argmin(cv[t])) for t in ts}
        preds = kr.fit_predict(tr, te, Y, [ALPHAS[j] for j in sorted(set(best.values()))])
        by_alpha = dict(zip(sorted(set(best.values())), preds))
        for n, t in enumerate(ts):
            j = best[t]
            res[t] = {'alpha': ALPHAS[j], 'cv': float(cv[t][j] / len(tr)), 'test_idx': te.tolist(),
                      'test_pred': by_alpha[j][:, cols[n]:cols[n + 1]].tolist()}
    return res


def load_readout(mdir, part, nm):
    p = mdir/f'{part}_{nm}.npy'
    return np.load(p, mmap_mode='r') if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    cache = Path(a.cache)
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure']
    mdir = cache/'out'/a.model
    objs = json.loads((mdir/'objects.json').read_text())
    oidx = {tuple(o): j for j, o in enumerate(objs)}
    recs = table(rows)
    sel = np.array([oidx[(o['id'], o['k'])] for o in recs])
    ids = sorted({o['id'] for o in recs})
    rng = np.random.default_rng(SEED)
    fold = dict(zip(ids, rng.integers(0, FOLDS, len(ids))))          # grouped by image
    meta = json.loads((mdir/'meta.json').read_text())
    readouts = {'vis': ('feat', 'vis'), 'lmimg': ('feat', 'lmimg'), 'word': ('feat', 'word'),
                'word_noimg': ('noimg', 'word'), 'word_mis': ('mis', 'word')}
    out = {'model': a.model, 'meta': meta, 'records': [{k: o[k] for k in ('id', 'k', 'domain', 'split', 'name')} for o in recs],
           'layers': {}, 'selected': {}, 'controls': {}}
    t0 = time.time()
    feats = {}
    for name, (part, nm) in readouts.items():
        arr = load_readout(mdir, part, nm)
        if arr is None:
            continue
        feats[name] = arr
        out['layers'][name] = []
        for l in range(arr.shape[1]):
            X = f32(arr[sel, l])
            r = probe_layer(X, recs, None, a.device, fold)
            out['layers'][name].append({t: {'alpha': v['alpha'], 'cv': v['cv']} for t, v in r.items()} | {'_pred': r})
            print(a.model, name, l, {t: round(v['cv'], 3) for t, v in r.items()}, round(time.time() - t0), 's', flush=True)
    # selected layers (cross-validated, training split only)
    n_lm = meta['n_lm_layers']
    late = range(int(np.ceil(0.75 * (n_lm - 1))), n_lm)
    fam = {'V': ('vis', range(meta['n_vis_layers'])), 'L_img': ('lmimg', late), 'L_img_any': ('lmimg', range(n_lm)),
           'W': ('word', range(1, n_lm)), 'W_noimg': ('word_noimg', range(1, n_lm)), 'W_mis': ('word_mis', range(1, n_lm))}
    for f, (name, rng_l) in fam.items():
        if name not in out['layers']:
            continue
        out['selected'][f] = {}
        for t in TARGETS:
            best = min(rng_l, key=lambda l: out['layers'][name][l][t]['cv'])
            p = out['layers'][name][best]['_pred'][t]
            out['selected'][f][t] = {'layer': int(best), 'readout': name, **p}
    # controls on V (and W): shuffled labels (5 permutations) and norm-matched random features
    for f in ('V', 'W'):
        if f not in out['selected']:
            continue
        out['controls'][f] = {}
        for t in TARGETS:
            name, l = out['selected'][f][t]['readout'], out['selected'][f][t]['layer']
            X = f32(feats[name][sel, l])
            perms = []
            for s in range(5):
                g = np.random.default_rng(SEED + 1 + s)
                r = probe_layer(X, recs, None, a.device, fold, targets={t: TARGETS[t]}, shuffle=lambda n: g.permutation(n))
                perms.append(r[t]['test_pred'])
            g = np.random.default_rng(SEED + 100)
            Rn = g.standard_normal(X.shape).astype(np.float32)
            Rn *= (np.linalg.norm(X, axis=1) / np.linalg.norm(Rn, axis=1))[:, None]
            r = probe_layer(Rn, recs, None, a.device, fold, targets={t: TARGETS[t]})
            out['controls'][f][t] = {'layer': l, 'shuffled_preds': perms, 'random': r[t]['test_pred'], 'test_idx': r[t]['test_idx']}
        print('controls', f, round(time.time() - t0), 's', flush=True)
    for name in out['layers']:                                         # keep per-layer test predictions (for curves)
        for rec in out['layers'][name]:
            rec['test'] = {t: rec['_pred'][t]['test_pred'] for t in TARGETS}
            rec['test_idx'] = {t: rec['_pred'][t]['test_idx'] for t in TARGETS}
            del rec['_pred']
    (cache/f'probe_{a.model}.json').write_text(json.dumps(out))
    print('saved', cache/f'probe_{a.model}.json', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
