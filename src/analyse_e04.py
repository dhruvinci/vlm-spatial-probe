"""E04 scoring (experiments/E04-more-measures/PREREG.md): per model and target, the probes (V, L_img, W and the
controls), the name-only prior P, the spoken answers and the specialists, on the test split, with 95% bootstrap
intervals over images or clips (2,000 resamples, paired); the pre-registered gates G0 / G1, recovery R, the outcome,
attribution, Q1 and Q3. Writes RESULTS.json and tables_e04.md next to it.

  python src/analyse_e04.py --data DATA_E04 --probes DIR(probe_e04_<model>.json) --answers OUT(e04_<model>/ask.jsonl)
      --cv DIR(cv_e04_normals.json, cv_e04_counts.json) --out experiments/E04-more-measures/data/results_e04.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyse import Boot, ci, json_blocks, match, stat
from probe_e04 import CLASSES, TARGETS, err, records, target

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
PRIMARY = ['facing', 'normal', 'speed', 'count']
FAMS = ['V', 'L_img', 'L_img_any', 'W', 'W_noimg', 'W_mis']


def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def fr(v):
    return 'n/a' if v is None or v != v else f'{v:.2f}'


def heading_class(deg, moving):
    if not moving:
        return 'still'
    d = deg % 360
    return 'towards' if d < 45 or d >= 315 else 'right' if d < 135 else 'away' if d < 225 else 'left'


def parse(rec):
    """Spoken values per object of one answer: facing, joints, surface class, dims, speed, direction, count."""
    d, _ = json_blocks(rec['answer'])
    items = d.get('objects', []) if isinstance(d, dict) else [x for x in d if isinstance(x, dict)]
    res = [{} for _ in rec['names']]
    for j, o in enumerate(items):
        if not isinstance(o, dict):
            continue
        k = match(rec['names'], o.get('name', '')) if 'name' in o else None
        if k is None and len(items) == len(rec['names']):
            k = j
        if k is None:
            continue
        v = res[k]
        if number(o.get('facing_deg')) is not None:
            v['facing'] = number(o['facing_deg']) % 360
        js = [number(o.get(f)) for f in ('left_elbow_deg', 'right_elbow_deg', 'left_knee_deg', 'right_knee_deg')]
        if all(x is not None for x in js):
            v['joints'] = js
        s = str(o.get('surface', '')).lower()
        if s:
            v['surface_class'] = 'towards' if 'towards' in s else 'up' if s.startswith('up') else 'sideways' if 'side' in s else None
        dm = o.get('dims_m')
        if isinstance(dm, list) and len(dm) == 3 and all(number(x) and number(x) > 0 for x in dm):
            v['dims'] = sorted((number(x) for x in dm), reverse=True)
        if number(o.get('speed_mps')) is not None and number(o['speed_mps']) >= 0:
            v['speed'] = number(o['speed_mps'])
        if str(o.get('direction', '')).lower() in ('towards', 'away', 'left', 'right', 'still'):
            v['direction'] = str(o['direction']).lower()
        if number(o.get('count')) is not None and number(o['count']) >= 0:
            v['count'] = number(o['count'])
    return res


def lenient_count(rec):
    """Exploratory: a count from answers that ignore the asked form: {"<class>": n}, {"<class>": [{"count": n}]}, or
    {"<class>": [one box per object]} (len of the list; a truncated list undercounts)."""
    d, _ = json_blocks(rec['answer'])
    if not isinstance(d, dict) or len(d) != 1:
        return None
    val = list(d.values())[0]
    if number(val) is not None:
        return number(val)
    if isinstance(val, list) and val and all(isinstance(e, dict) for e in val):
        cs = [number(e.get('count')) for e in val if number(e.get('count')) is not None]
        if len(cs) == 1:
            return cs[0]
        if all('bbox_2d' in e for e in val):
            return float(len(val))
    return None


def spoken_err(t, v, o):
    """Per-item error of a spoken value against the object's truth (None if not answered)."""
    if t == 'facing' and 'facing' in v:
        return abs((v['facing'] - o['facing_deg'] + 180) % 360 - 180)
    if t == 'joints' and 'joints' in v and o.get('elbow_l_deg') is not None:
        return float(np.mean(np.abs(np.subtract(v['joints'], [o[k] for k in ('elbow_l_deg', 'elbow_r_deg', 'knee_l_deg', 'knee_r_deg')]))))
    if t == 'surface_class' and v.get('surface_class'):
        return float(v['surface_class'] != o['surface_class'])
    if t == 'dims' and 'dims' in v and o.get('dims_m'):
        y = sorted(o['dims_m'], reverse=True)
        return float(np.mean(np.abs(np.subtract(v['dims'], y)) / np.array(y)))
    if t == 'speed' and 'speed' in v:
        return abs(v['speed'] - o['speed_mps'])
    if t == 'count' and 'count' in v:
        return abs(v['count'] - o['count'])
    return None


def name_prior(recs, Y, tr, te, kind):
    by = {}
    for i in tr:
        by.setdefault(recs[i]['name'], []).append(Y[i])
    g = np.mean(Y[tr], 0)
    P = np.array([np.mean(by[recs[i]['name']], 0) if recs[i]['name'] in by else g for i in te])
    if kind == 'class':
        P = np.eye(len(CLASSES))[P.argmax(1)]
    return P


def bal_err(boot, ids, e, Y):
    """1 - balanced accuracy, with its bootstrap draws (paired)."""
    c = Y.argmax(1)
    correct = e == 0
    p, b = None, []
    pts, bs = [], []
    for k in np.unique(c):
        m = c == k
        pk, bk = boot.mean([ids[i] for i in np.where(m)[0]], correct[m].astype(float))
        pts.append(pk); bs.append(bk)
    return 1 - float(np.mean(pts)), 1 - np.mean(bs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--probes', required=True)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--cv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    a = ap.parse_args()
    rows = json.loads((Path(a.data)/'dataset.json').read_text())
    recs = records(rows)
    split = np.array([r['split'] for r in recs])
    cvn = json.loads((Path(a.cv)/'cv_e04_normals.json').read_text()) if (Path(a.cv)/'cv_e04_normals.json').exists() else {}
    cvc = json.loads((Path(a.cv)/'cv_e04_counts.json').read_text()) if (Path(a.cv)/'cv_e04_counts.json').exists() else {}
    R_ = {'models': {}}
    lines = ['# E04 tables (printed by src/analyse_e04.py from results_e04.json)', '',
             'Test split. Errors: lower is better (angles in degrees; speed in m/s; count: absolute error; dims: abs-rel; surface '
             'class: 1 - balanced accuracy). 95% intervals: bootstrap over images or clips, paired. P = name-only prior.', '']
    spoken_all = {}
    for model in a.models.split(','):
        pf = Path(a.probes)/f'probe_e04_e04_{model}.json'
        if not pf.exists():
            pf = Path(a.probes)/f'probe_e04_{model}.json'
        if not pf.exists():
            R_['models'][model] = {'status': 'not run'}
            continue
        Pj = json.loads(pf.read_text())
        af = Path(a.answers)/f'e04_{model}'/'ask.jsonl'
        raw = {x['id']: x for x in (json.loads(l) for l in af.read_text().splitlines())} if af.exists() else {}
        asks = {k: parse(x) for k, x in raw.items()}
        M = {}
        lines += [f'## {model}', '', '| target | n | P | V | L_img | W | W no image | W mismatched | shuffled | random | spoken (coverage) | specialist | G0 | G1 | R L_img | R W | outcome | attribution |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for t in TARGETS:
            tgt = target(recs, t)
            if tgt is None or t not in Pj['selected'].get('V', {}):
                continue
            Y, ok = tgt
            kind = TARGETS[t][2]
            tr = np.where(ok & (split == 'train'))[0]; te = np.where(ok & (split == 'test'))[0]
            ids = [recs[i]['id'] for i in te]
            boot = Boot(ids)
            pos = {i: j for j, i in enumerate(te)}

            def pred_of(sel):
                p = dict(zip(sel['test_idx'], sel['test_pred']))
                return np.array([p[i] for i in te], float)
            preds = {'P': name_prior(recs, Y, tr, te, kind)}
            for f in FAMS:
                if f in Pj['selected'] and t in Pj['selected'][f]:
                    preds[f] = pred_of(Pj['selected'][f][t])
            E = {k: err(kind, v, Y[te]) for k, v in preds.items()}
            c = Pj['controls'].get('V', {}).get(t)
            if c:
                pc = lambda arr: np.array([dict(zip(c['test_idx'], arr))[i] for i in te], float)
                shuf = [err(kind, pc(s), Y[te]) for s in c['shuffled_preds']]
                E['shuffled'] = np.mean(shuf, 0)
                E['random'] = err(kind, pc(c['random']), Y[te])
            agg = (lambda e: bal_err(boot, ids, e, Y[te])) if kind == 'class' else (lambda e: boot.mean(ids, e))
            S = {k: agg(v) for k, v in E.items()}
            if c and kind == 'class':                                  # each permutation scored, then averaged
                per = [agg(e_) for e_ in shuf]
                S['shuffled'] = (float(np.mean([q_[0] for q_ in per])), np.mean([q_[1] for q_ in per], 0))
            res = {'n_test': int(len(te)), 'n_train': int(len(tr)), 'layers': {f: Pj['selected'][f][t]['layer'] for f in FAMS if f in Pj['selected'] and t in Pj['selected'][f]},
                   'error': {k: stat(p, b, len(te)) for k, (p, b) in S.items()}}
            Pp, Pb = S['P']; Vp, Vb = S['V']
            g0 = {k: {'P_minus_control': stat(Pp - S[k][0], Pb - S[k][1], len(te)), 'pass': bool(ci(Pb - S[k][1])[0] <= 0)}
                  for k in ('shuffled', 'random') if k in S}
            g1 = {'P_minus_V': stat(Pp - Vp, Pb - Vb, len(te)), 'relative_gain': float((Pp - Vp) / Pp) if Pp else None,
                  'pass': bool(Pp and (Pp - Vp) / Pp >= 0.20 and ci(Pb - Vb)[0] > 0)}
            rec_ = {}
            for x in ('L_img', 'W', 'W_noimg', 'W_mis', 'L_img_any'):
                if x in S:
                    Xp, Xb = S[x]
                    Rb = (Pb - Xb) / np.where(np.abs(Pb - Vb) < 1e-12, np.nan, Pb - Vb)
                    rec_[x] = {'R': float((Pp - Xp) / (Pp - Vp)) if Pp != Vp else None, 'ci': [float(v) for v in np.nanpercentile(Rb, [2.5, 97.5])],
                               'relative_gain_over_P': float((Pp - Xp) / Pp) if Pp else None, 'P_minus_X_ci': ci(Pb - Xb)}
            ctrl_ok = all(v['pass'] for v in g0.values())
            if not ctrl_ok:
                outcome = 'controls failed (G0): not interpreted'
            elif not g1['pass']:
                outcome = 'premise fails (G1): the tower does not carry the measure'
            else:
                strong = [x for x in ('L_img', 'W') if rec_.get(x, {}).get('R') is not None and rec_[x]['R'] >= 0.8
                          and rec_[x]['relative_gain_over_P'] >= 0.2 and rec_[x]['P_minus_X_ci'][0] > 0]
                weak = all(rec_.get(x, {}).get('R') is not None and rec_[x]['R'] < 0.5 and rec_[x]['ci'][1] < 0.8 for x in ('L_img', 'W'))
                outcome = 'falsified' if strong else 'supported' if weak else 'partial'
            attribution = {x: rec_[x]['R'] for x in ('W_noimg', 'W_mis') if x in rec_}
            attributed = all(v is not None and v < 0.3 for v in attribution.values()) if attribution else None
            # Q1: spoken
            sp = {}
            for j, i in enumerate(te):
                r = recs[i]; v = (asks.get(r['id']) or [{}] * 99)[r['k']] if r['id'] in asks else {}
                e = spoken_err(t, v, r['o'])
                if e is not None:
                    sp[j] = e
            q1 = {'coverage': len(sp) / max(len(te), 1)}
            if sp and kind != 'class':
                js = sorted(sp); sid = [ids[j] for j in js]
                p1, b1 = boot.mean(sid, [sp[j] for j in js]); p2, b2 = boot.mean(sid, E['P'][js]); p3, b3 = boot.mean(sid, E['W'][js])
                q1.update({'spoken': stat(p1, b1, len(js)), 'P_on_same': stat(p2, b2, len(js)), 'W_on_same': stat(p3, b3, len(js)),
                           'P_minus_spoken': stat(p2 - p1, b2 - b1, len(js)),
                           'measurement': bool(p2 and (p2 - p1) / p2 >= 0.2 and ci(b2 - b1)[0] > 0 and len(js) >= 30)})
                spoken_all.setdefault(t, {})[model] = {ids[j]: sp[j] for j in js} | {'_idx': {ids[j] + f'#{recs[te[j]]["k"]}': sp[j] for j in js}}
            elif sp:                                                   # class: 1 - balanced accuracy, as the probes
                js = sorted(sp); sid = [ids[j] for j in js]
                p1, b1 = bal_err(boot, sid, np.array([sp[j] for j in js]), Y[te][js])
                p2, b2 = bal_err(boot, sid, E['P'][js], Y[te][js])
                spoken_all.setdefault(t, {})[model] = {'_idx': {ids[j] + f'#{recs[te[j]]["k"]}': sp[j] for j in js},
                                                       '_cls': {ids[j] + f'#{recs[te[j]]["k"]}': int(Y[te[j]].argmax()) for j in js}}
                q1.update({'spoken_error_rate': float(np.mean([sp[j] for j in js])), 'n': len(js),
                           'spoken': stat(p1, b1, len(js)), 'P_on_same': stat(p2, b2, len(js)), 'P_minus_spoken': stat(p2 - p1, b2 - b1, len(js)),
                           'measurement': bool(p2 and (p2 - p1) / p2 >= 0.2 and ci(b2 - b1)[0] > 0 and len(js) >= 30)})
            if t == 'count' and sp is not None:                        # exploratory: lenient parse of off-form answers
                le = {}
                for j, i in enumerate(te):
                    r = recs[i]
                    if j in sp:
                        le[j] = sp[j]
                    elif r['id'] in raw and (n_ := lenient_count(raw[r['id']])) is not None and n_ >= 0:
                        le[j] = abs(n_ - r['o']['count'])
                if le:
                    js = sorted(le); sid = [ids[j] for j in js]
                    p1, b1 = boot.mean(sid, [le[j] for j in js])
                    q1['exploratory_lenient_parse'] = {'coverage': len(le) / len(te), 'spoken': stat(p1, b1, len(js))}
            if t == 'joints' and sp:                                     # exploratory: read the answer as flexion (180 - x)
                fl = {}
                for j, i in enumerate(te):
                    r = recs[i]; v = (asks.get(r['id']) or [{}] * 99)[r['k']] if r['id'] in asks else {}
                    if 'joints' in v and r['o'].get('elbow_l_deg') is not None:
                        fl[j] = float(np.mean(np.abs(np.subtract([180 - x for x in v['joints']], [r['o'][k] for k in ('elbow_l_deg', 'elbow_r_deg', 'knee_l_deg', 'knee_r_deg')]))))
                js = sorted(fl); sid = [ids[j] for j in js]
                p1, b1 = boot.mean(sid, [fl[j] for j in js])
                q1['exploratory_read_as_flexion'] = stat(p1, b1, len(js))
            # specialists
            spec = {}
            if t == 'normal' and cvn:
                se = {j: float(np.degrees(np.arccos(np.clip(np.dot(cvn[recs[i]['id']]['objects'][recs[i]['k']]['normal_cam'], Y[i]), -1, 1))))
                      for j, i in enumerate(te) if recs[i]['id'] in cvn and cvn[recs[i]['id']]['objects'][recs[i]['k']]['normal_cam']}
            elif t == 'count' and cvc:
                se = {j: abs(cvc[recs[i]['id']]['objects'][recs[i]['k']]['count'] - recs[i]['o']['count']) for j, i in enumerate(te) if recs[i]['id'] in cvc}
            else:
                se = {}
            if se:
                js = sorted(se); sid = [ids[j] for j in js]
                p1, b1 = boot.mean(sid, [se[j] for j in js]); p2, b2 = boot.mean(sid, E['W'][js])
                spec = {'specialist': stat(p1, b1, len(js)), 'W_on_same': stat(p2, b2, len(js)), 'W_over_specialist': float(p2 / p1) if p1 else None}
                both = [j for j in js if j in sp]                        # the spoken answer against the specialist, paired
                if both:
                    bid = [ids[j] for j in both]
                    p3, b3 = boot.mean(bid, [sp[j] for j in both]); p4, b4 = boot.mean(bid, [se[j] for j in both])
                    spec.update({'spoken_on_same': stat(p3, b3, len(both)), 'specialist_on_same_as_spoken': stat(p4, b4, len(both)),
                                 'spoken_minus_specialist': stat(p3 - p4, b3 - b4, len(both))})
            # per source
            per_src = {}
            for dom in sorted({recs[i]['domain'] for i in te}):
                js = [j for j, i in enumerate(te) if recs[i]['domain'] == dom]
                if len(js) < 2:
                    continue
                sid = [ids[j] for j in js]
                per_src[dom] = {k: stat(*boot.mean(sid, E[k][js]), len(js)) for k in ('P', 'V', 'L_img', 'W') if k in E and kind != 'class'}
            res.update({'G0': g0, 'G1': g1, 'recovery': rec_, 'outcome': outcome, 'attribution_R': attribution,
                        'attributed_to_image': attributed, 'Q1': q1, 'specialist': spec, 'per_source': per_src})
            M[t] = res
            f = lambda s: f"{s['value']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]" if s and s.get('value') == s.get('value') and s.get('value') is not None else 'n/a'
            e = res['error']
            spk = f"{f(q1.get('spoken'))} ({q1['coverage']:.0%})" if 'spoken' in q1 else (f"{q1.get('spoken_error_rate', float('nan')):.3f} ({q1['coverage']:.0%})" if q1.get('n') else 'n/a')
            lines.append(f"| {t} | {len(te)} | {f(e['P'])} | {f(e['V'])} | {f(e.get('L_img'))} | {f(e.get('W'))} | {f(e.get('W_noimg'))} | {f(e.get('W_mis'))} | "
                         f"{f(e.get('shuffled'))} | {f(e.get('random'))} | {spk} | {f(spec.get('specialist'))} | "
                         f"{'pass' if ctrl_ok else 'fail'} | {'pass' if g1['pass'] else 'fail'} | "
                         f"{fr(rec_.get('L_img', {}).get('R'))} | {fr(rec_.get('W', {}).get('R'))} | {outcome} | "
                         f"{'yes' if attributed else 'no' if attributed is not None else 'n/a'} |")
        # exploratory (not pre-registered): the spoken direction (towards / away / left / right / still) against the
        # truth's class (heading_class), on the motion test items; baseline = the training split's most common class
        tr_cls = [heading_class(r['o'].get('heading_deg') or 0, r['o']['moving']) for r in recs
                  if r['measure'] == 'motion' and r['split'] == 'train' and (r['o'].get('heading_deg') is not None or not r['o']['moving'])]
        base = max(set(tr_cls), key=tr_cls.count) if tr_cls else None
        dr = []
        for r in recs:
            if r['measure'] != 'motion' or r['split'] != 'test' or (r['o']['moving'] and r['o'].get('heading_deg') is None):
                continue
            tc = heading_class(r['o'].get('heading_deg') or 0, r['o']['moving'])
            v = asks[r['id']][r['k']] if r['id'] in asks else {}
            dr.append((r['id'], tc, v.get('direction')))
        if dr and base:
            ans = [x for x in dr if x[2]]
            boot = Boot([x[0] for x in ans])
            pa, ba = boot.mean([x[0] for x in ans], [float(x[1] == x[2]) for x in ans])
            pb, bb = boot.mean([x[0] for x in ans], [float(x[1] == base) for x in ans])
            M['direction_exploratory'] = {'n': len(dr), 'coverage': len(ans) / len(dr), 'baseline_class': base,
                                          'spoken_accuracy': stat(pa, ba, len(ans)), 'baseline_accuracy': stat(pb, bb, len(ans)),
                                          'spoken_minus_baseline': stat(pa - pb, ba - bb, len(ans))}
            g = lambda s: f"{s['value']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
            lines += ['', f"Exploratory (not pre-registered): spoken motion direction, 5 classes, n = {len(ans)} answered of {len(dr)}: "
                      f"accuracy {g(M['direction_exploratory']['spoken_accuracy'])}; always '{base}' (the training split's most "
                      f"common class) {g(M['direction_exploratory']['baseline_accuracy'])}."]
        g = lambda s: f"{s['value']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
        for t, T in M.items():
            sp_ = T.get('specialist', {}) if isinstance(T, dict) else {}
            if 'spoken_minus_specialist' in sp_:
                lines.append(f"Spoken against the specialist ({t}, paired, n = {sp_['spoken_on_same']['n']}): spoken "
                             f"{g(sp_['spoken_on_same'])}, specialist {g(sp_['specialist_on_same_as_spoken'])}, difference "
                             f"{g(sp_['spoken_minus_specialist'])}.")
        if 'count' in M and 'exploratory_lenient_parse' in M['count']['Q1']:
            x = M['count']['Q1']['exploratory_lenient_parse']
            lines.append(f"Exploratory (not pre-registered): spoken count with off-form answers also parsed ({{class: n}}, "
                         f"{{class: [{{count}}]}}, a list of boxes): coverage {x['coverage']:.0%}, error {g(x['spoken'])}.")
        if 'joints' in M and 'exploratory_read_as_flexion' in M['joints']['Q1']:
            lines.append(f"Exploratory (not pre-registered): spoken joint angles read as flexion (180 - answer): "
                         f"{g(M['joints']['Q1']['exploratory_read_as_flexion'])}.")
        lines += ['', f'Per source ({model}; the probes are fitted on all sources of a measure together):', '',
                  '| target | source | n | P | V | L_img | W |', '|---|---|---|---|---|---|---|']
        for t, T in M.items():
            for dom, ps in T.get('per_source', {}).items():
                if ps:
                    lines.append(f"| {t} | {dom} | {ps['P']['n']} | " + ' | '.join(f"{ps[k]['value']:.3f}" if k in ps else 'n/a' for k in ('P', 'V', 'L_img', 'W')) + ' |')
        R_['models'][model] = {'status': 'run', 'targets': M}
        lines.append('')
    # Q3: SpatialAxiom-9B against Qwen3.5-9B, asked, paired on the items both answered
    q3 = {}
    for t, by in spoken_all.items():
        if 'SpatialAxiom-9B' in by and 'Qwen3.5-9B' in by:
            A, B = by['SpatialAxiom-9B']['_idx'], by['Qwen3.5-9B']['_idx']
            common = sorted(set(A) & set(B))
            if common:
                ids = [c.split('#')[0] for c in common]
                boot = Boot(ids)
                if '_cls' in by['SpatialAxiom-9B']:                     # class: 1 - balanced accuracy on the common items
                    Yc = np.eye(len(CLASSES))[[by['SpatialAxiom-9B']['_cls'][c] for c in common]]
                    pa, ba = bal_err(boot, ids, np.array([A[c] for c in common]), Yc)
                    pb, bb = bal_err(boot, ids, np.array([B[c] for c in common]), Yc)
                else:
                    pa, ba = boot.mean(ids, [A[c] for c in common]); pb, bb = boot.mean(ids, [B[c] for c in common])
                q3[t] = {'SpatialAxiom-9B': stat(pa, ba, len(common)), 'Qwen3.5-9B': stat(pb, bb, len(common)),
                         'difference': stat(pa - pb, ba - bb, len(common)), 'counts': bool(ci(ba - bb)[1] < 0 or ci(ba - bb)[0] > 0)}
    R_['Q3'] = q3
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R_, indent=1, default=float))
    lines += ['## Q3: SpatialAxiom-9B against Qwen3.5-9B, asked (paired)', '', '| target | SpatialAxiom-9B | Qwen3.5-9B | difference | counts |', '|---|---|---|---|---|']
    for t, v in q3.items():
        g = lambda s: f"{s['value']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
        lines.append(f"| {t} | {g(v['SpatialAxiom-9B'])} | {g(v['Qwen3.5-9B'])} | {g(v['difference'])} | {v['counts']} |")
    (Path(a.out).parent/'tables_e04.md').write_text('\n'.join(lines) + '\n')
    print('saved', a.out)


if __name__ == '__main__':
    main()
