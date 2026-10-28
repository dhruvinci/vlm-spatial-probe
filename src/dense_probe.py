"""Exploratory (not pre-registered): per-patch depth on NYU indoor images, layer by layer.

Every merged image token of every indoor image, at a subset of layers (every 3rd vision block, every 4th language
layer, random-projected to 256 dims on the GPU by extract.py --dense). Target: the log of the mean Kinect depth inside
the token's cell. Ridge (primal, 256 dims), alpha by 5-fold cross-validation grouped by scene on the training scenes;
scored on the test scenes as abs-rel. Baselines: the training mean, and the cell's position alone (row and column,
one-hot: the patch-index control).

  python src/dense_probe.py --cache CACHE --model Qwen3.8-27B
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEED = 2604
ALPHAS = [10.0 ** k for k in range(-1, 5)]


def f32(u):
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def cell_depth(depth, hm, wm):
    """Mean valid depth per cell of an hm x wm grid over the depth map (NaN if no valid pixel)."""
    H, W = depth.shape
    ys = np.linspace(0, H, hm + 1).round().astype(int); xs = np.linspace(0, W, wm + 1).round().astype(int)
    out = np.full((hm, wm), np.nan)
    for i in range(hm):
        for j in range(wm):
            d = depth[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            d = d[d > 0]
            if d.size:
                out[i, j] = d.mean()
    return out.reshape(-1)


def ridge_fit(X, y, a):
    m, s = X.mean(0), X.std(0) + 1e-6
    Z = (X - m) / s
    w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - m) / s) @ w + y.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    mdir = cache/'out'/a.model
    rows = {r['id']: r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] == 'indoor'}
    grids = json.loads((mdir/'grids.json').read_text())
    dense = np.load(mdir/'dense_indoor.npz')
    gt = np.load(cache/'data/nyu_dense.npz')
    ids = sorted(rows, key=lambda k: int(k.split('_')[1]))
    scenes = sorted({rows[k]['scene'] for k in ids})
    fold = dict(zip(scenes, np.random.default_rng(SEED).integers(0, 5, len(scenes))))
    Y, img, pos = [], [], []
    for k in ids:
        hm, wm = grids[k][0] // 2, grids[k][1] // 2
        y = cell_depth(gt[f'{k}__depth'].astype(np.float32), hm, wm)
        Y.append(y); img += [k] * len(y)
        rr, cc = np.divmod(np.arange(hm * wm), wm)
        pos.append(np.stack([rr, cc], 1))
    Y = np.concatenate(Y); img = np.array(img); pos = np.concatenate(pos)
    ok = np.isfinite(Y) & (Y > 0)
    split = np.array([rows[k]['split'] for k in img])
    tr, te = ok & (split == 'train'), ok & (split == 'test')
    g = np.array([fold[rows[k]['scene']] for k in img])
    y = np.log(Y)

    def evaluate(X):
        cv = []
        for al in ALPHAS:
            err = []
            for f in range(5):
                a_tr, a_te = tr & (g != f), tr & (g == f)
                p = ridge_fit(X[a_tr], y[a_tr], al)(X[a_te])
                err.append(np.abs(np.exp(p) - Y[a_te]) / Y[a_te])
            cv.append(np.mean(np.concatenate(err)))
        al = ALPHAS[int(np.argmin(cv))]
        p = ridge_fit(X[tr], y[tr], al)(X[te])
        e = np.abs(np.exp(p) - Y[te]) / Y[te]
        return e, al, p

    test_imgs = sorted(set(img[te]))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(test_imgs), (2000, len(test_imgs)))
    ix = {k: j for j, k in enumerate(test_imgs)}
    jj = np.array([ix[k] for k in img[te]])

    def summ(e):
        s = np.bincount(jj, e, len(test_imgs)); c = np.bincount(jj, minlength=len(test_imgs)).astype(float)
        W = np.stack([np.bincount(d, minlength=len(test_imgs)) for d in draws]).astype(float)
        b = (W @ s) / (W @ c)
        return {'abs_rel': float(e.mean()), 'ci': np.percentile(b, [2.5, 97.5]).tolist(), 'n_cells': int(len(e)), 'n_images': len(test_imgs)}

    out = {'model': a.model, 'exploratory': True, 'baselines': {}, 'vis': [], 'lm': []}
    out['baselines']['train_mean'] = summ(np.abs(np.exp(y[tr].mean()) - Y[te]) / Y[te])
    onehot = np.zeros((len(Y), pos[:, 0].max() + 1 + pos[:, 1].max() + 1), np.float32)
    onehot[np.arange(len(Y)), pos[:, 0]] = 1; onehot[np.arange(len(Y)), pos[:, 0].max() + 1 + pos[:, 1]] = 1
    e, al, _ = evaluate(onehot)
    out['baselines']['position_only'] = summ(e)
    example = test_imgs[0]
    out['example'] = {'id': example, 'grid': [grids[example][0] // 2, grids[example][1] // 2],
                      'truth': np.where(np.isfinite(Y[img == example]), Y[img == example], -1).tolist(), 'pred': {}}
    for fam, key, layers in (('vis', 'vis', dense['vis_layers']), ('lm', 'lm', dense['lm_layers'])):
        for li, layer in enumerate(layers):
            X = np.concatenate([f32(dense[f'{k}__{key}'][li]) for k in ids])
            e, al, p = evaluate(X)
            out[fam].append({'layer': int(layer), 'alpha': al, **summ(e)})
            full = np.full(len(Y), np.nan); full[te] = np.exp(p)
            out['example']['pred'][f'{fam}{int(layer)}'] = np.nan_to_num(full[img == example], nan=-1).tolist()
            print(fam, layer, round(out[fam][-1]['abs_rel'], 3), flush=True)
    (cache/f'dense_{a.model}.json').write_text(json.dumps(out))
    print('saved', cache/f'dense_{a.model}.json')


if __name__ == '__main__':
    main()
