"""E07: why the decoder does not transfer (experiments/E07-transfer-diagnosis/PREREG.md). Depth only, on E03's data and
cache, on the CPU.

For each held-out scene type, four arms on the object word, each with E03's heads (ridge with alpha by inner
leave-one-scene-type-out; the small MLP; the lower inner error kept):
  R   E03's decoder: log depth.
  A1  the training scene types' mean differences projected out of the features (fitted on the rows being trained on).
  A2  features and log depths centred per image; scored scale-invariantly.
  A3  target log(depth x angular size); depth = exp(prediction) / angular size (src/cameras_e07.py).
Layer: E03's choice for the model and fold (primary); re-chosen by E03's inner rule over every 4th language layer plus
E03's (exploratory). Controls: 2D cues (ridge on the box, centred per image) for A2; box + camera by name for A3.
Diagnostic: three-way scene-type classification before and after A1's erasure. Writes CACHE_E03/e07_<model>.json;
scored by src/analyse_e07.py.

  python src/diagnose_e07.py --cache CACHE --e03 CACHE_E03 --model Qwen3.8-27B [--threads 8]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe import ALPHAS, KRidge, f32, gram
from transfer import GROUPS_E03, SEED, mlp_fit_predict, records

DOMS = GROUPS_E03['geo'][0]
ARMS = ('R', 'A1', 'A2', 'A3')
DEV = 'cpu'


def centre(V, groups):
    """Subtract each image's mean (groups: the row indices of each image)."""
    V = np.array(V, np.float64 if V.dtype != np.float32 else np.float32)
    for g in groups:
        V[g] -= V[g].mean(0)
    return V


def erase(X, rows, dom):
    """Project out the mean differences of the scene types present in `rows` (k - 1 directions): afterwards those scene
    types have equal mean features (LEACE's linear guarantee, without its whitening)."""
    mu = X[rows].mean(0)
    Dm = np.stack([X[rows[dom[rows] == d]].mean(0) - mu for d in np.unique(dom[rows])], 1)
    U, sv, _ = np.linalg.svd(Dm.astype(np.float64), full_matrices=False)
    U = U[:, sv > sv.max() * 1e-6].astype(np.float32)
    return X - (X @ U) @ U.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--e03', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--cameras', default='', help='default: CACHE_E03/e07_cameras.json')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--smoke', default='', help='one held-out scene type at E03\'s layer only; writes e07_<model>.smoke.json')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cache, c3 = Path(a.cache), Path(a.e03)
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure' and r.get('activity') not in ('mma', 'grappling')]
    rows2 = json.loads((c3/'data/dataset.json').read_text())
    recs = records(rows, {}) + records(rows2, {})
    n1 = len(records(rows, {}))
    boxes = [o['box'] for r in rows + rows2 for o in r['objects']]
    cams = json.loads(Path(a.cameras or c3/'e07_cameras.json').read_text())
    ok = np.array([o['domain'] in DOMS and o['depth'] is not None and o['centre'] is not None for o in recs])
    idx = np.where(ok)[0]                                   # the depth objects, E01's first (E03's 'geo' rows)
    S = [recs[i] for i in idx]
    dom = np.array([o['domain'] for o in S]); names = [o['name'] for o in S]
    img = np.unique([o['id'] for o in S], return_inverse=True)[1]
    grp = [np.where(img == k)[0] for k in range(img.max() + 1)]
    D = np.array([o['depth'] for o in S], float)
    s_ang = np.array([cams[o['id']][o['k']] for o in S], float)
    W = np.array([o['size_img'] for o in S], float)
    B = np.array([boxes[i] for i in idx], float) * np.concatenate([W, W], 1)
    bw, bh = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
    cues = np.stack([np.log(np.sqrt(bw * bh)), (B[:, 0] + B[:, 2]) / 2 / W[:, 0], (B[:, 1] + B[:, 3]) / 2 / W[:, 1], np.log(bw / bh)], 1)
    Y = {'R': np.log(D)[:, None], 'A1': np.log(D)[:, None], 'A2': centre(np.log(D)[:, None], grp), 'A3': np.log(D * s_ang)[:, None]}

    def depth_err(arm, pred, rows_):
        """Per-object error on depth (lower is better): abs-rel, or scale-invariant abs-rel for A2."""
        p = np.asarray(pred, float)[:, 0]
        y = D[rows_]
        if arm == 'A2':
            q = np.exp(p); e = np.empty(len(y))
            for k in np.unique(img[rows_]):
                m = img[rows_] == k
                sc = np.median(y[m] / q[m]); e[m] = np.abs(q[m] * sc - y[m]) / y[m]
            return e
        z = np.exp(p) / s_ang[rows_] if arm == 'A3' else np.exp(p)
        return np.abs(z - y) / y

    mdir, mdir2 = cache/'out'/a.model, c3/'out'/f'e03_{a.model}'
    oidx = {tuple(o): j for j, o in enumerate(json.loads((mdir/'objects.json').read_text()))}
    oidx2 = {tuple(o): j for j, o in enumerate(json.loads((mdir2/'objects.json').read_text()))}
    first = idx < n1
    sel1 = np.array([oidx[(recs[i]['id'], recs[i]['k'])] for i in idx[first]])
    sel2 = np.array([oidx2[(recs[i]['id'], recs[i]['k'])] for i in idx[~first]])
    F1 = np.load(mdir/'feat_word.npy', mmap_mode='r'); F2 = np.load(mdir2/'feat_word.npy', mmap_mode='r')
    n_lm = json.loads((mdir/'meta.json').read_text())['n_lm_layers']

    def feat(l):
        return np.ascontiguousarray(np.concatenate([f32(F1[sel1, l]), f32(F2[sel2, l])]))

    e03 = json.loads((c3/f'e03_{a.model}.json').read_text())
    pos = {(o['id'], o['k']): j for j, o in enumerate(S)}
    plan = {}
    for held in DOMS:
        Fe = e03['folds'][f'geo|{held}']
        te = np.where(dom == held)[0]; tr = np.where(dom != held)[0]
        assert [pos[(e03['records'][i]['id'], e03['records'][i]['k'])] for i in Fe['test_idx']] == te.tolist()
        assert Fe['train_n'] == len(tr)
        inner = [(tr[dom[tr] != d], tr[dom[tr] == d]) for d in DOMS if d != held]
        plan[held] = {'tr': tr, 'te': te, 'inner': inner, 'e03_layer': Fe['targets']['depth']['W']['layer'],
                      'e03_alpha': Fe['targets']['depth']['W']['alpha']}
    if a.smoke:
        plan = {a.smoke: plan[a.smoke]}
    layers = sorted(set(range(1, n_lm, 4)) | {P['e03_layer'] for P in plan.values()}) if not a.smoke else [plan[a.smoke]['e03_layer']]
    t0 = time.time()
    log = lambda *x: print(a.model, *x, round(time.time() - t0), 's', flush=True)

    def mask(rows_):
        m = np.zeros(len(S), bool); m[rows_] = True
        return m

    def ridge_inner(arm, X, P):
        """Inner error per alpha (mean over every inner held-out object), ridge only."""
        sc = np.zeros(len(ALPHAS)); n = 0
        if arm == 'A1':
            for itr, iva in P['inner']:
                Xe = erase(X, itr, dom)
                for j, pr in enumerate(KRidge(gram(Xe, mask(P['tr']), DEV), DEV).fit_predict(itr, iva, Y[arm], ALPHAS)):
                    sc[j] += depth_err(arm, pr, iva).sum()
                n += len(iva)
            return sc / n
        Xa = centre(X, grp) if arm == 'A2' else X
        kr = KRidge(gram(Xa, mask(P['tr']), DEV), DEV)
        for itr, iva in P['inner']:
            for j, pr in enumerate(kr.fit_predict(itr, iva, Y[arm], ALPHAS)):
                sc[j] += depth_err(arm, pr, iva).sum()
            n += len(iva)
        return sc / n

    def fit(arm, X, P, alpha):
        """Final ridge at alpha, the MLP (inner error and final), and the head with the lower inner error."""
        def view(rows_fit):
            if arm == 'A1':
                return erase(X, rows_fit, dom)
            return centre(X, grp) if arm == 'A2' else X
        Xf = view(P['tr'])
        rp = KRidge(gram(Xf, mask(P['tr']), DEV), DEV).fit_predict(P['tr'], P['te'], Y[arm], [alpha])[0]
        sc = 0.0; n = 0
        for itr, iva in P['inner']:
            Xi = view(itr)
            sc += depth_err(arm, mlp_fit_predict(Xi[itr], Y[arm][itr], Xi[iva], DEV), iva).sum(); n += len(iva)
        mp = mlp_fit_predict(Xf[P['tr']], Y[arm][P['tr']], Xf[P['te']], DEV)
        return {'ridge': rp.tolist(), 'mlp': mp.tolist(), 'inner_mlp': sc / n}

    # ---- inner ridge sweep over the candidate layers (saved as it goes; resumable)
    sweep_file = c3/f"e07_{a.model}{'.smoke' if a.smoke else ''}.sweep.json"
    sweep = json.loads(sweep_file.read_text()) if sweep_file.exists() else {}
    for l in layers:
        if all(f'{arm}|{l}|{h}' in sweep for arm in ARMS for h in DOMS):
            continue
        X = feat(l)
        for held, P in plan.items():
            for arm in ARMS:
                if f'{arm}|{l}|{held}' not in sweep:
                    sweep[f'{arm}|{l}|{held}'] = ridge_inner(arm, X, P).tolist()
        sweep_file.write_text(json.dumps(sweep))
        log('sweep layer', l)
    # ---- fits: primary (E03's layer) and exploratory (re-chosen layer)
    out = {'model': a.model, 'objects': [{k: o[k] for k in ('id', 'k', 'domain', 'activity', 'name')} for o in S],
           'angular_size': s_ang.tolist(), 'layers_swept': layers, 'folds': {}}
    for held, P in plan.items():
        Fo = {'test_idx': P['te'].tolist(), 'train_n': len(P['tr']), 'e03_layer': P['e03_layer'], 'e03_alpha': P['e03_alpha'],
              'primary': {}, 'exploratory': {}}
        cand = sorted(set(range(1, n_lm, 4)) | {P['e03_layer']}) if not a.smoke else [P['e03_layer']]
        cache_fit = {}
        for arm in ARMS:
            for kind in ('primary', 'exploratory'):
                if kind == 'primary':
                    l = P['e03_layer']; j = int(np.argmin(sweep[f'{arm}|{l}|{held}']))
                else:
                    l, j = min(((l_, j_) for l_ in cand for j_ in range(len(ALPHAS))), key=lambda lj: sweep[f'{arm}|{lj[0]}|{held}'][lj[1]])
                if (l, j) not in cache_fit:
                    cache_fit[(l, j)] = fit(arm, feat(l), P, ALPHAS[j])
                r = dict(cache_fit[(l, j)])
                r.update({'layer': int(l), 'alpha': ALPHAS[j], 'inner_ridge': sweep[f'{arm}|{l}|{held}'][j]})
                r['head'] = 'mlp' if r['inner_mlp'] < r['inner_ridge'] else 'ridge'
                Fo[kind][arm] = r
                log(held, arm, kind, 'layer', l, 'alpha', ALPHAS[j], r['head'])
            cache_fit = {}
        # controls
        Zc = centre(cues, grp)
        kr = KRidge(gram(Zc, mask(P['tr']), DEV), DEV)
        sc = np.zeros(len(ALPHAS))
        for itr, iva in P['inner']:
            for j, pr in enumerate(kr.fit_predict(itr, iva, Y['A2'], ALPHAS)):
                sc[j] += depth_err('A2', pr, iva).sum()
        al = ALPHAS[int(np.argmin(sc))]
        by = {}
        for i in P['tr']:
            by.setdefault(names[i], []).append(Y['A3'][i, 0])
        g = Y['A3'][P['tr'], 0].mean()
        Fo['controls'] = {'cues2d': {'alpha': al, 'pred': kr.fit_predict(P['tr'], P['te'], Y['A2'], [al])[0].tolist()},
                          'name_box_camera': {'pred': [[float(np.mean(by[names[i]])) if names[i] in by else float(g)] for i in P['te']],
                                              'names_unseen': int(sum(names[i] not in by for i in P['te']))}}
        # diagnostic: scene-type classification on the training rows, before and after erasure (E03's layer)
        X = feat(P['e03_layer'])
        ids = sorted(set(img[P['tr']].tolist()))
        fold_of = dict(zip(ids, np.random.default_rng(SEED).integers(0, 5, len(ids))))
        f_of = np.array([fold_of[k] for k in img[P['tr']]])
        train_doms = [d for d in DOMS if d != held]
        onehot = np.array([[float(d == t) for t in train_doms] for d in dom])
        acc = {}
        for name_ in ('before', 'after'):
            hits = np.zeros(len(ALPHAS)); n = 0
            for f in range(5):
                ctr, cte = P['tr'][f_of != f], P['tr'][f_of == f]
                Xd = erase(X, ctr, dom) if name_ == 'after' else X
                for j, pr in enumerate(KRidge(gram(Xd, mask(ctr), DEV), DEV).fit_predict(ctr, cte, onehot, ALPHAS)):
                    hits[j] += (pr.argmax(1) == onehot[cte].argmax(1)).sum()
                n += len(cte)
            acc[name_] = {'best_alpha_accuracy': float(hits.max() / n), 'alpha': ALPHAS[int(hits.argmax())],
                          'chance_majority': float(np.bincount(onehot[P['tr']].argmax(1)).max() / len(P['tr']))}
        Fo['scene_type_classification'] = acc
        log(held, 'controls + diagnostic', acc)
        out['folds'][held] = Fo
        (c3/f"e07_{a.model}{'.smoke' if a.smoke else ''}.json").write_text(json.dumps(out))
    log('saved', c3/f"e07_{a.model}{'.smoke' if a.smoke else ''}.json")


if __name__ == '__main__':
    main()
