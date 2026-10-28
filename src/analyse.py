"""Score everything against the truth, with 95% bootstrap intervals over images, and apply the pre-registered gates
(experiments/E01-spatial-ask-and-probe/PREREG.md). Reads the cache (extract.py outputs, probe_<model>.json from
probe.py, cv_moge.json / cv_rfdetr.json from cv_reference.py); writes experiments/E01-.../data/results.json.

Every number is reported twice for the two-person domain: with all of its frames (the pre-registered set) and without
the MMA and grappling sequences ("calm").

  python src/analyse.py --cache CACHE --out experiments/E01-spatial-ask-and-probe/data/results.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

SEED = 2604
B = 2000
MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
TRAIN_DOMAINS = {'xy': ('people', 'indoor', 'two_people'), 'depth': ('indoor', 'two_people'),
                 'centre': ('indoor', 'two_people'), 'size': ('indoor',)}
COMBAT = ('mma', 'grappling')


# ---------------------------------------------------------------------------- records
def records(rows):
    recs = []
    for r in rows:
        for k, o in enumerate(r['objects']):
            recs.append({'id': r['id'], 'k': k, 'domain': r['domain'], 'split': r['split'], 'name': o['name'],
                         'activity': r.get('activity'), 'xy': o['xy'], 'W': r['size'][0], 'H': r['size'][1],
                         'depth': o.get('depth_m'), 'size': o.get('size_m'), 'centre': o.get('centre_m')})
    return recs


def subset_mask(recs, idx, dataset):
    """'all' = the pre-registered test set; 'calm' = without the MMA and grappling frames."""
    if dataset == 'all':
        return np.ones(len(idx), bool)
    return np.array([recs[i]['activity'] not in COMBAT for i in idx])


# ---------------------------------------------------------------------------- per-item errors
def err_xy(recs, idx, pred):
    if len(idx) == 0:
        return np.zeros(0)
    W = np.array([[recs[i]['W'], recs[i]['H']] for i in idx], float)
    y = np.array([recs[i]['xy'] for i in idx], float)
    d = (np.asarray(pred, float) - y) * W
    return np.hypot(d[:, 0], d[:, 1]) / np.hypot(W[:, 0], W[:, 1]) * 100


def err_absrel(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return np.abs(pred - true) / true


def delta125(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return (np.maximum(pred / true, true / pred) < 1.25).astype(float)


# ---------------------------------------------------------------------------- bootstrap over images
class Boot:
    """Resampling weights over images, shared by every metric so differences are paired."""
    def __init__(self, ids, seed=SEED):
        self.ids = sorted(set(ids))
        self.pos = {k: j for j, k in enumerate(self.ids)}
        rng = np.random.default_rng(seed)
        n = len(self.ids)
        draws = rng.integers(0, n, (B, n))
        self.W = np.zeros((B, n), np.float32)
        for b in range(B):
            self.W[b] = np.bincount(draws[b], minlength=n)

    def sums(self, item_ids, values):
        j = np.array([self.pos[i] for i in item_ids], dtype=int)
        s = np.bincount(j, weights=values, minlength=len(self.ids)); c = np.bincount(j, minlength=len(self.ids)).astype(float)
        return s, c

    def mean(self, item_ids, values):
        if len(item_ids) == 0:
            return float('nan'), np.full(B, np.nan)
        s, c = self.sums(item_ids, np.asarray(values, float))
        return float(s.sum() / c.sum()), (self.W @ s) / np.maximum(self.W @ c, 1e-9)

    def balanced_acc(self, item_ids, correct, truth):
        correct, truth = np.asarray(correct, bool), np.asarray(truth, bool)
        tp, p = self.sums(item_ids, (correct & truth).astype(float))[0], self.sums(item_ids, truth.astype(float))[0]
        tn, n = self.sums(item_ids, (correct & ~truth).astype(float))[0], self.sums(item_ids, (~truth).astype(float))[0]
        point = 0.5 * (tp.sum() / max(p.sum(), 1) + tn.sum() / max(n.sum(), 1))
        boot = 0.5 * ((self.W @ tp) / np.maximum(self.W @ p, 1e-9) + (self.W @ tn) / np.maximum(self.W @ n, 1e-9))
        return float(point), boot


def ci(boot):
    if not np.isfinite(boot).any():
        return [float('nan'), float('nan')]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return [float(lo), float(hi)]


def stat(point, boot, n):
    return {'value': point, 'ci': ci(boot), 'n': int(n)}


# ---------------------------------------------------------------------------- name-only prior
def prior_pred(recs, target, test_idx):
    doms = TRAIN_DOMAINS[target]
    tr = [o for o in recs if o['split'] == 'train' and o['domain'] in doms and val(o, target) is not None]
    by = {}
    for o in tr:
        by.setdefault(o['name'], []).append(val(o, target))
    g = np.mean([val(o, target) for o in tr], 0)
    means = {k: np.mean(v, 0) for k, v in by.items()}
    return np.array([means.get(recs[i]['name'], g) for i in test_idx])


def val(o, target):
    if target == 'xy':
        return np.array(o['xy'], float)
    if target == 'depth':
        return np.log(o['depth']) if o['depth'] else None
    if target == 'size':
        return np.log(o['size']) if o['size'] else None
    return np.array(o['centre'], float) if o['centre'] else None


def score(recs, target, idx, pred):
    """Per-item error for a target: % diagonal (xy), abs-rel (depth, size; pred in log), 3D error in m (centre)."""
    pred = np.asarray(pred, float)
    if target == 'xy':
        return err_xy(recs, idx, pred)
    if target in ('depth', 'size'):
        return err_absrel([recs[i][target] for i in idx], np.exp(pred.reshape(-1)))
    return np.linalg.norm(pred - np.array([recs[i]['centre'] for i in idx], float), axis=1)


# ---------------------------------------------------------------------------- answer parsing
def norm_name(s):
    s = str(s).lower().strip()
    s = re.sub(r'^the\s+', '', s)
    return re.sub(r'[^a-z0-9 \']', '', s).strip()


def number(v):
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    m = re.search(r'-?\d+(\.\d+)?', str(v)) if v is not None else None
    return float(m.group()) if m else None


def json_blocks(text):
    t = text.strip()
    t = re.sub(r'^```(json)?', '', t).strip(); t = re.sub(r'```$', '', t).strip()
    for cand in (t, t[t.find('{'):t.rfind('}') + 1] if '{' in t else '', t[t.find('['):t.rfind(']') + 1] if '[' in t else ''):
        try:
            return json.loads(cand), True
        except Exception:
            continue
    out = []
    for blk in re.findall(r'\{[^{}]*\}', t):
        try:
            out.append(json.loads(blk))
        except Exception:
            pass
    return out, False


def match(names, cand):
    """Index of the object a model's label refers to."""
    c = norm_name(cand)
    ns = [norm_name(n) for n in names]
    if c in ns:
        return ns.index(c)
    hits = [j for j, n in enumerate(ns) if n in c or c in n]
    return hits[0] if len(hits) == 1 else None


def parse_ask(rec):
    names = rec['names']
    d, whole = json_blocks(rec['answer'])
    objs = d.get('objects', []) if isinstance(d, dict) else [x for x in d if isinstance(x, dict) and 'name' in x]
    pairs = d.get('pairs', []) if isinstance(d, dict) else [x for x in d if isinstance(x, dict) and 'a' in x]
    res = [{} for _ in names]
    for j, o in enumerate(objs):
        if not isinstance(o, dict):
            continue
        k = match(names, o.get('name', '')) if 'name' in o else None
        if k is None and len(objs) == len(names):
            k = j
        if k is None:
            continue
        x, y, dep, sz = (number(o.get(f)) for f in ('x', 'y', 'depth_m', 'size_m'))
        if x is not None and y is not None:
            res[k]['xy'] = [x / 100, y / 100]
        if dep is not None and dep > 0:
            res[k]['depth'] = dep
        if sz is not None and sz > 0:
            res[k]['size'] = sz
    pd = {}
    for p in pairs:
        if not isinstance(p, dict):
            continue
        a, b, dist = match(names, p.get('a', '')), match(names, p.get('b', '')), number(p.get('distance_m'))
        if a is not None and b is not None and a != b and dist is not None and dist > 0:
            pd[(min(a, b), max(a, b))] = dist
    return res, pd, whole


def parse_ground(rec, kind, grid):
    names = rec['names']
    d, _ = json_blocks(rec['answer'])
    items = d if isinstance(d, list) else d.get('objects', []) if isinstance(d, dict) else []
    res = [None] * len(names)
    for j, o in enumerate(items):
        if not isinstance(o, dict) or 'bbox_2d' not in o:
            continue
        k = match(names, o.get('label', '')) if 'label' in o else (j if len(items) == len(names) else None)
        bb = o['bbox_2d']
        if k is None or not isinstance(bb, list) or len(bb) != 4 or res[k] is not None:
            continue
        bb = [number(v) for v in bb]
        if any(v is None for v in bb):
            continue
        if kind == 'qwen3_5':
            sx = sy = 1000.0                                   # Qwen3-family boxes: relative, 0-1000
        else:
            sy, sx = grid[0] * 14.0, grid[1] * 14.0            # Qwen2.5-VL boxes: pixels of the resized input
        res[k] = [(bb[0] + bb[2]) / 2 / sx, (bb[1] + bb[3]) / 2 / sy]
    return res


# ---------------------------------------------------------------------------- main analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    a = ap.parse_args()
    MODELS[:] = a.models.split(',')
    cache = Path(a.cache)
    rows_all = json.loads((cache/'data/dataset.json').read_text())
    rows = [r for r in rows_all if r['domain'] != 'figure']
    byid = {r['id']: r for r in rows_all}
    recs = records(rows)
    load_relations(rows, recs)
    rindex = {(o['id'], o['k']): i for i, o in enumerate(recs)}
    boot = Boot([o['id'] for o in recs])
    moge = json.loads((cache/'cv_moge.json').read_text()) if (cache/'cv_moge.json').exists() else {}
    rfd = json.loads((cache/'cv_rfdetr.json').read_text()) if (cache/'cv_rfdetr.json').exists() else {}
    R = {'models': {}, 'notes': []}

    def m(idx, errs):
        pt, bt = boot.mean([recs[i]['id'] for i in idx], errs)
        return pt, bt

    def test_idx(target, dataset, domains=None):
        doms = domains or TRAIN_DOMAINS[target]
        idx = [i for i, o in enumerate(recs) if o['split'] == 'test' and o['domain'] in doms and val(o, target) is not None]
        return [i for i, keep in zip(idx, subset_mask(recs, idx, dataset)) if keep]

    for model in MODELS:
        mdir = cache/'out'/model
        if not (mdir/'meta.json').exists():
            R['models'][model] = {'status': 'not run'}
            continue
        meta = json.loads((mdir/'meta.json').read_text())
        res = {'status': 'run', 'meta': {k: meta.get(k) for k in ('kind', 'n_vis_layers', 'n_lm_layers', 'd_vis', 'd_lm', 'gpu',
                                                                 'transformers', 'torch', 'timings_s', 'total_s', 'checks')}}
        pj = cache/f'probe_{model}.json'
        probe = json.loads(pj.read_text()) if pj.exists() else None
        # ---------------- Q2: probes, gates, recovery
        if probe:
            res['probe'] = {}
            for dataset in ('all', 'calm'):
                out = {}
                for target in ('depth', 'xy', 'size', 'centre'):
                    T = test_idx(target, dataset)
                    sel = probe['selected']

                    preds = {'P': prior_pred(recs, target, T)}
                    for fam in ('V', 'L_img', 'L_img_any', 'W', 'W_noimg', 'W_mis'):
                        if fam in sel:
                            s_ = sel[fam][target]
                            pos_ = {i: j for j, i in enumerate(s_['test_idx'])}
                            preds[fam] = [s_['test_pred'][pos_[i]] for i in T]
                    e = {k: score(recs, target, T, v) for k, v in preds.items()}
                    for f in ('V', 'W'):
                        c = probe['controls'].get(f, {}).get(target)
                        if c:
                            pos = {i: j for j, i in enumerate(c['test_idx'])}
                            sh = np.mean([score(recs, target, T, [p[pos[i]] for i in T]) for p in c['shuffled_preds']], 0)
                            e[f'{f}_shuffled'] = sh
                            e[f'{f}_random'] = score(recs, target, T, [c['random'][pos[i]] for i in T])
                    pts = {k: m(T, v) for k, v in e.items()}
                    tgt = {'n_objects': len(T), 'n_images': len({recs[i]['id'] for i in T}),
                           'layers': {f: sel[f][target]['layer'] for f in sel},
                           'error': {k: stat(p, b, len(T)) for k, (p, b) in pts.items()}}
                    if target == 'depth':
                        tgt['delta125'] = {k: stat(*m(T, delta125([recs[i]['depth'] for i in T], np.exp(np.ravel(v)))), len(T))
                                           for k, v in preds.items()}
                    Pp, Pb = pts['P']; Vp, Vb = pts['V']
                    g0 = {}
                    for c in ('V_shuffled', 'V_random'):
                        if c in pts:
                            diff = Pb - pts[c][1]
                            g0[c] = {'P_minus_control': stat(Pp - pts[c][0], diff, len(T)), 'pass': bool(ci(diff)[1] <= 0 or ci(diff)[0] <= 0)}
                    gain = Pb - Vb
                    g1 = {'P_minus_V': stat(Pp - Vp, gain, len(T)), 'relative_gain': float((Pp - Vp) / Pp),
                          'pass': bool((Pp - Vp) / Pp >= 0.20 and ci(gain)[0] > 0)}
                    rec_ = {}
                    for x in ('L_img', 'W', 'W_noimg', 'W_mis', 'L_img_any'):
                        if x in pts:
                            Xp, Xb = pts[x]
                            Rb = (Pb - Xb) / np.where(np.abs(Pb - Vb) < 1e-12, np.nan, Pb - Vb)
                            rec_[x] = {'R': float((Pp - Xp) / (Pp - Vp)), 'ci': [float(v) for v in np.nanpercentile(Rb, [2.5, 97.5])],
                                       'relative_gain_over_P': float((Pp - Xp) / Pp), 'P_minus_X_ci': ci(Pb - Xb)}
                    tgt['G0'] = g0; tgt['G1'] = g1; tgt['recovery'] = rec_
                    if target in ('depth', 'xy', 'size'):
                        tgt['verdict'] = verdict(g0, g1, rec_, target)
                    out[target] = tgt
                out['relations'] = probe_relations(recs, probe, boot, dataset)
                out['distance'] = probe_distance(recs, rows, probe, boot, dataset, prior_pred)
                res['probe'][dataset] = out
            res['curves'] = curves(recs, probe, boot)
        # ---------------- Q1/Q3: asked answers
        res['ask'] = ask_scores(model, mdir, meta, recs, rows, byid, rindex, boot, probe, moge, rfd)
        res['yesno'] = yesno_scores(mdir, recs, rows, byid, boot)
        R['models'][model] = res
    R['paired'] = paired(R, recs, rows, cache, boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=1, default=float))
    print('saved', a.out)


def verdict(g0, g1, rec, target):
    if not all(v['pass'] for v in g0.values()):
        return 'controls failed (G0): not interpreted'
    if not g1['pass']:
        return 'G1 failed: the tower does not beat the name prior by 20%'
    keys = ['W'] if target == 'xy' else ['L_img', 'W']
    fal = [k for k in keys if k in rec and rec[k]['R'] >= 0.8 and rec[k]['relative_gain_over_P'] >= 0.2 and rec[k]['P_minus_X_ci'][0] > 0]
    if fal:
        return 'falsified (' + ', '.join(fal) + ' keeps >= 0.8 of the tower gain)'
    if all(k in rec and rec[k]['R'] < 0.5 and rec[k]['ci'][1] < 0.8 for k in keys):
        return 'supported'
    return 'partial'


def probe_relations(recs, probe, boot, dataset):
    """Relations derived from the selected probes' per-object predictions, on test pairs."""
    sel = probe['selected']
    out = {}
    for fam in ('V', 'L_img', 'W'):
        if fam not in sel:
            continue
        preds = {}
        for t in ('xy', 'depth', 'size'):
            s = sel[fam][t]
            preds[t] = {i: p for i, p in zip(s['test_idx'], s['test_pred'])}
        items = relation_items(recs, dataset)
        corr, truth, ids, rel = [], [], [], []
        for (ia, ib, r_, t_, rid) in items:
            t = {'closer': 'depth', 'left_of': 'xy', 'above': 'xy', 'bigger': 'size'}[r_]
            if ia not in preds[t] or ib not in preds[t]:
                continue
            pa, pb = np.ravel(preds[t][ia]), np.ravel(preds[t][ib])
            ans = pa[1] < pb[1] if r_ == 'above' else pa[0] > pb[0] if r_ == 'bigger' else pa[0] < pb[0]
            corr.append(ans == t_); truth.append(t_); ids.append(rid); rel.append(r_)
        out[fam] = {}
        for r_ in ('closer', 'left_of', 'above', 'bigger', 'pooled'):
            k = [j for j in range(len(rel)) if r_ == 'pooled' or rel[j] == r_]
            if not k:
                continue
            p, b = boot.balanced_acc([ids[j] for j in k], [corr[j] for j in k], [truth[j] for j in k])
            out[fam][r_] = stat(p, b, len(k))
    return out


_REL = []


def load_relations(rows, recs):
    """(index a, index b, relation, truth, image id) for every pair (two_people: no left_of; the names give it)."""
    pos = {(o['id'], o['k']): i for i, o in enumerate(recs)}
    _REL.clear()
    for r in rows:
        for p in r.get('pairs', []):
            for rel, truth in p['relations'].items():
                if not (r['domain'] == 'two_people' and rel == 'left_of'):
                    _REL.append((pos[(r['id'], p['a'])], pos[(r['id'], p['b'])], rel, bool(truth), r['id']))


def relation_items(recs, dataset, splits=('test',)):
    return [x for x in _REL if recs[x[0]]['split'] in splits and (dataset == 'all' or recs[x[0]]['activity'] not in COMBAT)]


def probe_distance(recs, rows, probe, boot, dataset, prior_fn):
    sel = probe['selected']
    pos = {(o['id'], o['k']): i for i, o in enumerate(recs)}
    pairs = []
    for r in rows:
        if r['split'] != 'test' or r['domain'] not in TRAIN_DOMAINS['centre'] or (dataset == 'calm' and r.get('activity') in COMBAT):
            continue
        for p in r['pairs']:
            if 'distance_m' in p:
                pairs.append((pos[(r['id'], p['a'])], pos[(r['id'], p['b'])], p['distance_m'], r['id']))
    if not pairs:
        return {}
    idx = sorted({i for a, b, _, _ in pairs for i in (a, b)})
    pr = dict(zip(idx, prior_fn(recs, 'centre', idx)))
    out = {}
    cands = {'P': pr}
    for fam in ('V', 'L_img', 'W'):
        if fam in sel:
            s = sel[fam]['centre']
            cands[fam] = {i: np.array(p) for i, p in zip(s['test_idx'], s['test_pred'])}
    for k, c in cands.items():
        e = [abs(np.linalg.norm(np.subtract(c[a], c[b])) - d) / d for a, b, d, _ in pairs]
        p, b = boot.mean([x[3] for x in pairs], e)
        out[k] = stat(p, b, len(pairs))
    return out


def curves(recs, probe, boot):
    """Every layer's test error (pooled test of the training domains), per readout and target."""
    out = {}
    for name, layers in probe['layers'].items():
        out[name] = {}
        for t in ('depth', 'xy', 'size', 'centre'):
            T = [i for i, o in enumerate(recs) if o['split'] == 'test' and o['domain'] in TRAIN_DOMAINS[t] and val(o, t) is not None]
            vals = []
            for L in layers:
                pos = {i: j for j, i in enumerate(L['test_idx'][t])}
                e = score(recs, t, T, [L['test'][t][pos[i]] for i in T])
                pt, bt = boot.mean([recs[i]['id'] for i in T], e)
                vals.append([pt] + ci(bt))
            out[name][t] = vals
    return out


def ask_scores(model, mdir, meta, recs, rows, byid, rindex, boot, probe, moge, rfd):
    """Asked numbers (JSON) and native boxes, against truth, the prior, the selected probes and the CV references,
    on the same objects."""
    if not (mdir/'ask.jsonl').exists():
        return {'status': 'not run'}
    ask = [json.loads(l) for l in (mdir/'ask.jsonl').read_text().splitlines()]
    grid = json.loads((mdir/'grids.json').read_text()) if (mdir/'grids.json').exists() else {}
    parsed, pdist, whole = {}, {}, 0
    for rec in ask:
        p, pd, w = parse_ask(rec)
        parsed[rec['id']] = p; pdist[rec['id']] = pd; whole += w
    out = {'n_answers': len(ask), 'valid_json': whole, 'raw_figure': next((x['answer'] for x in ask if byid[x['id']]['domain'] == 'figure'), None),
           'parsed_figure': next(([*parsed[x['id']]] for x in ask if byid[x['id']]['domain'] == 'figure'), None)}
    ground = {}
    if (mdir/'ground.jsonl').exists():
        for rec in (json.loads(l) for l in (mdir/'ground.jsonl').read_text().splitlines()):
            ground[rec['id']] = parse_ground(rec, meta['kind'], grid.get(rec['id'], [0, 0]))
        out['ground_figure'] = next((ground[k] for k in ground if byid[k]['domain'] == 'figure'), None)
    sel = probe['selected'] if probe else {}
    for dataset in ('all', 'calm'):
        D = {}
        for target, field in (('depth', 'depth'), ('xy', 'xy'), ('size', 'size')):
            for split, doms in (('test', TRAIN_DOMAINS[target]), ('heldout', ('street',)), ('test_people', ('people',)),
                                ('test_indoor', ('indoor',)), ('test_two_people', ('two_people',))):
                if (split == 'test_people' and target != 'xy') or (split in ('test_indoor', 'test_two_people') and target != 'depth'):
                    continue
                idx = [i for i, o in enumerate(recs) if o['split'] == ('heldout' if split == 'heldout' else 'test') and o['domain'] in doms
                       and val(o, target) is not None and (dataset == 'all' or o['activity'] not in COMBAT)]
                if not idx:
                    continue
                have = [i for i in idx if field in parsed.get(recs[i]['id'], [{}] * 9)[recs[i]['k']]]
                cov = len(have) / len(idx)
                if not have:
                    D[f'{target}_{split}'] = {'coverage': 0.0, 'n_asked': len(idx)}
                    continue
                e = {}
                if target == 'xy':
                    xy = np.array([parsed[recs[i]['id']][recs[i]['k']]['xy'] for i in have], float)
                    e['ask'] = err_xy(recs, have, xy)
                    # exploratory (after the run): answers above 100 are read as the 0-1000 grounding scale
                    out_rng = (xy > 1.0).any(1)
                    if meta['kind'] == 'qwen3_5':           # the Qwen3 family's grounding scale is 0-1000
                        e['ask_rescaled_exploratory'] = err_xy(recs, have, np.where(out_rng[:, None], xy / 10, xy))
                else:
                    e['ask'] = err_absrel([recs[i][target] for i in have], [parsed[recs[i]['id']][recs[i]['k']][field] for i in have])
                e['P'] = score(recs, target, have, prior_pred(recs, target, have))
                for fam in ('V', 'L_img', 'W'):
                    if fam in sel:
                        s = sel[fam][target]
                        pos = {i: j for j, i in enumerate(s['test_idx'])}
                        if all(i in pos for i in have):
                            e[fam] = score(recs, target, have, [s['test_pred'][pos[i]] for i in have])
                if target == 'depth' and moge:
                    mv = [moge[recs[i]['id']]['objects'][recs[i]['k']]['depth_m'] for i in have]
                    if all(v for v in mv):
                        e['MoGe-2'] = err_absrel([recs[i]['depth'] for i in have], mv)
                key = f'{target}_{split}'
                D[key] = {'coverage': cov, 'n_asked': len(idx), **{k: stat(*boot.mean([recs[i]['id'] for i in have], v), len(have)) for k, v in e.items()}}
                if target == 'xy':
                    D[key]['share_above_100'] = float(out_rng.mean())
                    D[key]['median_ask'] = float(np.median(e['ask']))
                ids_ = [recs[i]['id'] for i in have]
                pp, pb = boot.mean(ids_, e['P']); ap_, ab = boot.mean(ids_, e['ask'])
                D[key]['P_minus_ask'] = stat(pp - ap_, pb - ab, len(have))
                D[key]['ask_gain_over_P'] = float((pp - ap_) / pp) if pp else None
                D[key]['ask_is_measurement'] = bool(D[key]['ask_gain_over_P'] is not None and D[key]['ask_gain_over_P'] >= 0.2
                                                    and D[key]['P_minus_ask']['ci'][0] > 0)   # pre-registered rule (Q1)
                if target == 'depth':
                    D[key]['delta125_ask'] = stat(*boot.mean([recs[i]['id'] for i in have], delta125([recs[i]['depth'] for i in have],
                                              [parsed[recs[i]['id']][recs[i]['k']]['depth'] for i in have])), len(have))
                    D[key]['median_ratio_ask'] = float(np.median([parsed[recs[i]['id']][recs[i]['k']]['depth'] / recs[i]['depth'] for i in have])) if have else None
                if target == 'xy' and ground:
                    gidx = [i for i in idx if ground.get(recs[i]['id']) and ground[recs[i]['id']][recs[i]['k']] is not None]
                    if gidx:
                        D[key]['ground'] = stat(*boot.mean([recs[i]['id'] for i in gidx], err_xy(recs, gidx, [ground[recs[i]['id']][recs[i]['k']] for i in gidx])), len(gidx))
                        D[key]['ground_coverage'] = len(gidx) / len(idx)
                    if rfd:
                        ridx = [i for i in idx if recs[i]['id'] in rfd and rfd[recs[i]['id']]['objects'][recs[i]['k']]['box']]
                        if ridx:
                            bx = [rfd[recs[i]['id']]['objects'][recs[i]['k']]['box'] for i in ridx]
                            D[key]['RF-DETR'] = stat(*boot.mean([recs[i]['id'] for i in ridx], err_xy(recs, ridx, [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in bx])), len(ridx))
        # pair distances (asked) on test pairs of depth domains and held-out street
        for split, doms in (('test', TRAIN_DOMAINS['centre']), ('heldout', ('street',))):
            items = [(r['id'], p) for r in rows if r['split'] == split and r['domain'] in doms and (dataset == 'all' or r.get('activity') not in COMBAT)
                     for p in r['pairs'] if 'distance_m' in p]
            have = [(rid, p) for rid, p in items if (min(p['a'], p['b']), max(p['a'], p['b'])) in pdist.get(rid, {})]
            if have:
                e = err_absrel([p['distance_m'] for _, p in have], [pdist[rid][(min(p['a'], p['b']), max(p['a'], p['b']))] for rid, p in have])
                D[f'distance_{split}'] = {'coverage': len(have) / len(items), 'n_asked': len(items), 'ask': stat(*boot.mean([rid for rid, _ in have], e), len(have))}
        out[dataset] = D
    return out


def yesno_scores(mdir, recs, rows, byid, boot):
    if not (mdir/'yesno.jsonl').exists():
        return {'status': 'not run'}
    items = [json.loads(l) for l in (mdir/'yesno.jsonl').read_text().splitlines()]
    out = {}
    for dataset in ('all', 'calm'):
        D = {}
        keep = [x for x in items if dataset == 'all' or byid[x['id']].get('activity') not in COMBAT]
        rel = [x for x in keep if x['kind'] == 'relation']
        for dom in ('people', 'indoor', 'two_people', 'street', 'pooled_test', 'pooled'):
            for r_ in ('closer', 'left_of', 'above', 'bigger', 'all'):
                k = [x for x in rel if (dom == 'pooled' or (dom == 'pooled_test' and byid[x['id']]['domain'] != 'street') or byid[x['id']]['domain'] == dom)
                     and (r_ == 'all' or x['relation'] == r_)]
                if not k:
                    continue
                p, b = boot.balanced_acc([x['id'] for x in k], [(x['logp_yes'] > x['logp_no']) == x['truth'] for x in k], [x['truth'] for x in k])
                D[f'{dom}/{r_}'] = {**stat(p, b, len(k)), 'yes_rate': float(np.mean([x['logp_yes'] > x['logp_no'] for x in k]))}
        nam = [x for x in keep if x['kind'] == 'naming']
        for dom in ('people', 'indoor', 'street', 'pooled'):
            k = [x for x in nam if dom == 'pooled' or byid[x['id']]['domain'] == dom]
            if k:
                p, b = boot.mean([x['id'] for x in k], [float((x['logp_yes'] > x['logp_no']) == x['truth']) for x in k])
                D[f'naming/{dom}'] = {**stat(p, b, len(k)), 'yes_rate': float(np.mean([x['logp_yes'] > x['logp_no'] for x in k]))}
        out[dataset] = D
    out['_items'] = {'relation': sum(x['kind'] == 'relation' for x in items), 'naming': sum(x['kind'] == 'naming' for x in items)}
    return out


def paired(R, recs, rows, cache, boot):
    """Q3: paired differences between models on the same items (asked depth / size / distance abs-rel, 2D error,
    relation balanced accuracy, naming accuracy)."""
    pairs = [('SpatialAxiom-9B', 'Qwen3.5-9B'), ('SpatialAxiom-9B', 'Qwen3.8-27B'), ('SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B'),
             ('SpaceQwen2.5-VL-3B', 'Qwen3.8-27B'), ('Qwen3.5-9B', 'Qwen3.8-27B')]
    out = {}
    byid = {r['id']: r for r in rows}
    asks, yn = {}, {}
    for mname in MODELS:
        d = cache/'out'/mname
        if (d/'ask.jsonl').exists():
            asks[mname] = {x['id']: parse_ask(x) for x in (json.loads(l) for l in (d/'ask.jsonl').read_text().splitlines())}
        if (d/'yesno.jsonl').exists():
            yn[mname] = [json.loads(l) for l in (d/'yesno.jsonl').read_text().splitlines()]
    for A, Bm in pairs:
        if A not in asks or Bm not in asks:
            continue
        res = {}
        for dataset in ('all', 'calm'):
            D = {}
            for target in ('depth', 'size', 'xy'):
                for split, doms in (('test', TRAIN_DOMAINS[target]), ('heldout', ('street',))):
                    idx = [i for i, o in enumerate(recs) if o['split'] == split and o['domain'] in doms and val(o, target) is not None
                           and (dataset == 'all' or o['activity'] not in COMBAT)]
                    have = [i for i in idx if all(o['id'] in asks[mm] and target in asks[mm][o['id']][0][o['k']]
                                                  for mm in (A, Bm) for o in [recs[i]])]
                    if not have:
                        continue
                    errs = {}
                    for mm in (A, Bm):
                        v = [asks[mm][recs[i]['id']][0][recs[i]['k']][target] for i in have]
                        errs[mm] = err_xy(recs, have, v) if target == 'xy' else err_absrel([recs[i][target] for i in have], v)
                    ids = [recs[i]['id'] for i in have]
                    pa, ba = boot.mean(ids, errs[A]); pb, bb = boot.mean(ids, errs[Bm])
                    D[f'{target}_{split}'] = {A: stat(pa, ba, len(have)), Bm: stat(pb, bb, len(have)), 'diff_A_minus_B': stat(pa - pb, ba - bb, len(have))}
                    if target == 'xy' and all(json.loads((cache/'out'/mm/'meta.json').read_text())['kind'] == 'qwen3_5' for mm in (A, Bm)):
                        # exploratory: answers above 100 read as the Qwen3 family's 0-1000 grounding scale
                        re = {}
                        for mm in (A, Bm):
                            v = np.array([asks[mm][recs[i]['id']][0][recs[i]['k']]['xy'] for i in have], float)
                            re[mm] = err_xy(recs, have, np.where((v > 1).any(1)[:, None], v / 10, v))
                        pa, ba = boot.mean(ids, re[A]); pb, bb = boot.mean(ids, re[Bm])
                        D[f'xy_rescaled_exploratory_{split}'] = {A: stat(pa, ba, len(have)), Bm: stat(pb, bb, len(have)), 'diff_A_minus_B': stat(pa - pb, ba - bb, len(have))}
            if A in yn and Bm in yn:
                for kind in ('relation', 'naming'):
                    ka = {(x['id'], x.get('relation', x.get('name')), x.get('a'), x.get('b')): x for x in yn[A] if x['kind'] == kind
                          and (dataset == 'all' or byid[x['id']].get('activity') not in COMBAT)}
                    kb = {(x['id'], x.get('relation', x.get('name')), x.get('a'), x.get('b')): x for x in yn[Bm] if x['kind'] == kind}
                    keys = [k for k in ka if k in kb]
                    if not keys:
                        continue
                    ids = [k[0] for k in keys]
                    truth = [ka[k]['truth'] for k in keys]
                    ca = [(ka[k]['logp_yes'] > ka[k]['logp_no']) == ka[k]['truth'] for k in keys]
                    cb = [(kb[k]['logp_yes'] > kb[k]['logp_no']) == kb[k]['truth'] for k in keys]
                    if kind == 'relation':
                        pa, ba = boot.balanced_acc(ids, ca, truth); pb, bb = boot.balanced_acc(ids, cb, truth)
                    else:
                        pa, ba = boot.mean(ids, np.array(ca, float)); pb, bb = boot.mean(ids, np.array(cb, float))
                    D[kind] = {A: stat(pa, ba, len(keys)), Bm: stat(pb, bb, len(keys)), 'diff_A_minus_B': stat(pa - pb, ba - bb, len(keys))}
            res[dataset] = D
        out[f'{A} vs {Bm}'] = res
    return out


if __name__ == '__main__':
    main()
