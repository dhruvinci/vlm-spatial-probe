"""E02 scoring: every held-out fold on its own, against the spoken answer, the name prior and the specialist, on the
same objects; 95% bootstrap intervals over images; the pre-registered pass bar (experiments/E02-decoder-transfer/PREREG.md).

  python src/analyse_e02.py --cache CACHE --out experiments/E02-decoder-transfer/data/results_e02.json
  (also writes data/tables_e02.md, or tables_e03.md with --e03, next to it)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyse import COMBAT, Boot, ci, parse_ask, parse_ground, stat
from transfer import GROUPS, GROUPS_E03, errors, records

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
MEASURES = [('depth', 'geo', 'abs-rel'), ('centre', 'geo', 'relative 3D error'), ('distance', 'geo', 'abs-rel'),
            ('size', 'size', 'abs-rel'), ('xy', 'xy', '% of diagonal'), ('orient', 'orient', 'mean angle error (deg)')]


def name_prior(recs, tr, t, te):
    """Training-fold mean per object name (log space for depth/size; circular via sin/cos for orientation)."""
    from transfer import targets
    Y = targets(recs, t)
    by = {}
    for i in tr:
        by.setdefault(recs[i]['name'], []).append(Y[i])
    g = np.mean([Y[i] for i in tr], 0)
    return np.array([np.mean(by[recs[i]['name']], 0) if recs[i]['name'] in by else g for i in te])


def pick(T, f):
    r = T[f]
    return np.array(r[r.get('head', 'ridge')], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--e03', default='', help='E03 cache root: score e03_<model>.json (calm frames; more domains)')
    a = ap.parse_args()
    MODELS[:] = a.models.split(',')
    cache = Path(a.cache)
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure']
    orient = json.loads((cache/'e02_orientation.json').read_text())['objects']
    groups, measures, tag = GROUPS, MEASURES, 'e02'
    if a.e03:
        c3 = Path(a.e03)
        rows = [r for r in rows if r.get('activity') not in COMBAT]
        rows2 = json.loads((c3/'data/dataset.json').read_text())
        groups, measures, tag = GROUPS_E03, [m for m in MEASURES if m[0] != 'orient'], 'e03'
    recs = records(rows, orient) + (records(rows2, {}) if a.e03 else [])
    rows = rows + (rows2 if a.e03 else [])
    byid = {r['id']: r for r in rows}
    pos = {(o['id'], o['k']): i for i, o in enumerate(recs)}
    moge = json.loads((cache/'cv_moge.json').read_text())
    rfd = {**json.loads((cache/'cv_rfdetr.json').read_text()), **json.loads((cache/'cv_rfdetr_e02.json').read_text())}
    if a.e03:
        moge.update(json.loads((c3/'cv_moge.json').read_text()))
        rfd.update(json.loads((c3/'cv_rfdetr.json').read_text())); rfd.update(json.loads((c3/'cv_rfdetr_e02.json').read_text()))
    R = {'models': {}}
    lines = [f'# {tag.upper()} tables (printed by src/analyse_e02.py from results_{tag}.json)', '',
             'Errors: lower is better. Intervals: 95% bootstrap over the held-out images. "calm" = without MMA and grappling frames.',
             'Pass = the decoder (object word, chosen head) beats the spoken answer (interval of spoken minus decoder above 0) '
             'and is within 1.5x the specialist\'s error on the same objects.', '']
    for model in MODELS:
        p = (Path(a.e03) if a.e03 else cache)/f'{tag}_{model}.json'
        if not p.exists():
            R['models'][model] = {'status': 'not run'}
            continue
        E = json.loads(p.read_text())
        mdir = cache/'out'/model
        meta = json.loads((mdir/'meta.json').read_text())
        asks = {x['id']: parse_ask(x) for x in (json.loads(l) for l in (mdir/'ask.jsonl').read_text().splitlines())}
        if a.e03 and (Path(a.e03)/'out'/f'e03_{model}'/'ask.jsonl').exists():
            asks.update({x['id']: parse_ask(x) for x in (json.loads(l) for l in (Path(a.e03)/'out'/f'e03_{model}'/'ask.jsonl').read_text().splitlines())})
        grids = json.loads((mdir/'grids.json').read_text())
        ground = {x['id']: parse_ground(x, meta['kind'], grids.get(x['id'], [0, 0])) for x in
                  (json.loads(l) for l in (mdir/'ground.jsonl').read_text().splitlines())}
        M = {}
        lines += [f'## {model}', '']
        for measure, g, unit in measures:
            fold_keys = [k for k in E['folds'] if k.startswith(g + '|')]
            for fk in fold_keys:
                held = fk.split('|')[1]
                F = E['folds'][fk]
                te_all = np.array(F['test_idx'])
                t = 'centre' if measure == 'distance' else measure
                T = F['targets'][t]
                train_doms = [d for d in groups[g][0] if d != held] if g != 'orient' else ['street']
                if g == 'orient':
                    tr = [i for i, o in enumerate(recs) if o['alpha'] is not None and i not in set(te_all.tolist())]
                else:
                    tr = [i for i, o in enumerate(recs) if o['domain'] in train_doms and (o[t if t != 'xy' else 'xy'] is not None)]
                for ds in ('all', 'calm'):
                    keep = np.array([ds == 'all' or recs[i]['activity'] not in COMBAT for i in te_all])
                    te = te_all[keep]
                    if not len(te):
                        continue
                    sub = lambda arr: np.asarray(arr, float)[keep]
                    preds = {'decoder (word)': sub(pick(T, 'W')), 'word, ridge': sub(T['W']['ridge']), 'word, MLP': sub(T['W']['mlp']),
                             'vision tower': sub(pick(T, 'V')), 'late image tokens': sub(pick(T, 'L_img')),
                             'image tokens, any layer': sub(T['L_img_any']['ridge']),
                             'word, no image': sub(T['controls']['word_noimg']['pred']),
                             **({'word, mismatched image': sub(T['controls']['word_mis']['pred'])} if 'word_mis' in T['controls'] else {}),
                             'random features': sub(T['controls']['random']['pred']),
                             'name prior': name_prior(recs, tr, t, te)}
                    sh = [sub(s) for s in T['controls']['shuffled']['preds']]
                    boot = Boot([recs[i]['id'] for i in te])
                    ids = [recs[i]['id'] for i in te]
                    res = {'held_out': held, 'n_objects': int(len(te)), 'n_images': len(set(ids)), 'head': T['W'].get('head'),
                           'layers': {f: T[f]['layer'] for f in ('V', 'L_img', 'L_img_any', 'W')}, 'error': {}}
                    if measure == 'distance':
                        pidx = {i: j for j, i in enumerate(te)}
                        pairs = [(pos[(r['id'], q['a'])], pos[(r['id'], q['b'])], q['distance_m'], r['id'], q) for r in rows
                                 if r['domain'] == held for q in r['pairs'] if 'distance_m' in q]
                        pairs = [x for x in pairs if x[0] in pidx and x[1] in pidx]
                        if not pairs:
                            continue
                        pid = [x[3] for x in pairs]
                        boot = Boot(pid)
                        dist = lambda P_: np.array([abs(np.linalg.norm(P_[pidx[a_]] - P_[pidx[b_]]) - d) / d for a_, b_, d, _, _ in pairs])
                        E_ = {k: dist(v) for k, v in preds.items()}
                        E_['shuffled labels'] = np.mean([dist(s) for s in sh], 0)
                        res['n_objects'] = len(pairs); res['n_images'] = len(set(pid))
                        res['error'] = {k: stat(*boot.mean(pid, v), len(pairs)) for k, v in E_.items()}
                        dec = E_['decoder (word)']
                        sp = [(j, asks[x[3]][1].get((min(x[4]['a'], x[4]['b']), max(x[4]['a'], x[4]['b']))))   # images with no
                              for j, x in enumerate(pairs) if x[3] in asks]                                        # spoken answer: skipped
                        sp = [(j, v) for j, v in sp if v]
                        spoken = {j: abs(v - pairs[j][2]) / pairs[j][2] for j, v in sp}
                        mg = {}
                        for j, (a_, b_, d, rid, q) in enumerate(pairs):
                            ca, cb = moge[rid]['objects'][q['a']]['centre_m'], moge[rid]['objects'][q['b']]['centre_m']
                            if ca and cb:
                                mg[j] = abs(np.linalg.norm(np.subtract(ca, cb)) - d) / d
                    else:
                        E_ = {k: errors(t, v, recs, te) for k, v in preds.items()}
                        E_['shuffled labels'] = np.mean([errors(t, s, recs, te) for s in sh], 0)
                        res['error'] = {k: stat(*boot.mean(ids, v), len(te)) for k, v in E_.items()}
                        dec = E_['decoder (word)']
                        spoken, mg, nat = {}, {}, {}
                        for j, i in enumerate(te):
                            o = recs[i]; ans = asks.get(o['id'], ([{}] * 9, {}, 0))[0]
                            ans = ans[o['k']] if o['k'] < len(ans) else {}
                            if t in ('depth', 'size') and t in ans:
                                spoken[j] = abs(ans[t] - o[t]) / o[t]
                            if t == 'xy' and 'xy' in ans:
                                spoken[j] = errors('xy', np.array([ans['xy']]), recs, [i])[0]
                            if t == 'xy' and ground.get(o['id']) and ground[o['id']][o['k']] is not None:
                                nat[j] = errors('xy', np.array([ground[o['id']][o['k']]]), recs, [i])[0]
                            if t == 'depth' and moge[o['id']]['objects'][o['k']]['depth_m']:
                                mg[j] = abs(moge[o['id']]['objects'][o['k']]['depth_m'] - o['depth']) / o['depth']
                            if t == 'centre' and moge[o['id']]['objects'][o['k']]['centre_m']:
                                c = np.array(moge[o['id']]['objects'][o['k']]['centre_m']); y = np.array(o['centre'])
                                mg[j] = np.linalg.norm(c - y) / np.linalg.norm(y)
                            if t == 'xy' and o['id'] in rfd and rfd[o['id']]['objects'][o['k']].get('box'):
                                b = rfd[o['id']]['objects'][o['k']]['box']
                                mg[j] = errors('xy', np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]), recs, [i])[0]
                        if t == 'depth':                                  # scale-invariant abs-rel (images with >= 2 objects)
                            res['scale_invariant'] = si_scores(recs, te, preds, asks, moge, boot)
                        if nat:
                            js = sorted(nat)
                            res['native_box'] = stat(*boot.mean([ids[j] for j in js], [nat[j] for j in js]), len(js))
                    pid_list = [pairs[j][3] for j in range(len(pairs))] if measure == 'distance' else ids
                    # pass bar
                    bar = {}
                    if spoken:
                        js = sorted(spoken)
                        sp_e = np.array([spoken[j] for j in js]); de = dec[js]
                        sid = [pid_list[j] for j in js]
                        pa, ba = boot.mean(sid, sp_e); pd_, bd = boot.mean(sid, de)
                        bar['spoken'] = stat(pa, ba, len(js)); bar['decoder_on_spoken_objects'] = stat(pd_, bd, len(js))
                        bar['spoken_minus_decoder'] = stat(pa - pd_, ba - bd, len(js))
                        bar['beats_spoken'] = bool(ci(ba - bd)[0] > 0) if len(js) >= 30 else None
                    if mg:
                        js = sorted(mg)
                        sid = [pid_list[j] for j in js]
                        ps, bs = boot.mean(sid, [mg[j] for j in js]); pd_, bd = boot.mean(sid, dec[js])
                        bar['specialist'] = stat(ps, bs, len(js)); bar['decoder_on_specialist_objects'] = stat(pd_, bd, len(js))
                        bar['ratio_to_specialist'] = float(pd_ / ps) if ps else None
                        bar['within_1_5x'] = bool(pd_ <= 1.5 * ps) if len(js) >= 30 else None
                    if 'beats_spoken' in bar and 'within_1_5x' in bar and None not in (bar['beats_spoken'], bar['within_1_5x']):
                        bar['pass'] = bar['beats_spoken'] and bar['within_1_5x']
                    elif 'beats_spoken' in bar and 'specialist' not in bar and bar['beats_spoken'] is not None:
                        bar['pass_spoken_half_only'] = bar['beats_spoken']
                    res['bar'] = bar
                    M.setdefault(f'{measure}|{held}', {})[ds] = res
        R['models'][model] = {'status': 'run', 'folds': M}
        lines += table_lines(model, M)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=1, default=float))
    (Path(a.out).parent/f'tables_{tag}.md').write_text('\n'.join(lines) + '\n')
    print('saved', a.out)


def si_scores(recs, te, preds, asks, moge, boot):
    """Scale-invariant abs-rel: per image, predictions rescaled by median(true/pred); images with >= 2 objects."""
    by = {}
    for j, i in enumerate(te):
        by.setdefault(recs[i]['id'], []).append(j)
    multi = [j for js in by.values() if len(js) >= 2 for j in js]
    if not multi:
        return {}
    y = np.array([recs[te[j]]['depth'] for j in multi])
    out = {}
    cands = {k: np.exp(np.asarray(v, float)[multi, 0]) for k, v in preds.items() if k in ('decoder (word)', 'vision tower', 'name prior')}
    sp = [asks.get(recs[te[j]]['id'], ([{}] * 9,))[0] for j in multi]
    sp = [(s[recs[te[j]]['k']] if recs[te[j]]['k'] < len(s) else {}).get('depth') for s, j in zip(sp, multi)]
    mg = [moge[recs[te[j]]['id']]['objects'][recs[te[j]]['k']]['depth_m'] for j in multi]
    ids = [recs[te[j]]['id'] for j in multi]
    for name, p in list(cands.items()) + [('spoken', np.array([v if v else np.nan for v in sp], float)), ('MoGe-2', np.array([v if v else np.nan for v in mg], float))]:
        ok = np.isfinite(p)
        e = np.full(len(p), np.nan)
        for k in set(ids):
            m = np.array([x == k for x in ids]) & ok
            if m.sum() >= 2:
                s = np.median(y[m] / p[m]); e[m] = np.abs(p[m] * s - y[m]) / y[m]
        keep = np.isfinite(e)
        out[name] = stat(*boot.mean([ids[q] for q in np.where(keep)[0]], e[keep]), int(keep.sum()))
    return out


def table_lines(model, M):
    f = lambda s, d=3: f"{s['value']:.{d}f} [{s['ci'][0]:.{d}f}, {s['ci'][1]:.{d}f}]" if s and 'value' in s and s['value'] == s['value'] else 'n/a'
    out = []
    for ds in ('calm', 'all'):
        out += [f'### {model}, {"without MMA and grappling (paper-facing)" if ds == "calm" else "all frames"}', '',
                '| measure | held out | n | decoder: word (head, layer) | spoken | specialist | name prior | vision tower | late image tokens | word, no image | word, mismatched | shuffled | random | spoken minus decoder | decoder / specialist | pass |',
                '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for key, v in M.items():
            if ds not in v:
                continue
            r = v[ds]; e = r['error']; b = r['bar']; d = 2 if key.startswith(('xy', 'orient')) else 3
            verdict = ('PASS' if b.get('pass') else 'fail') if 'pass' in b else (
                ('beats spoken (no specialist)' if b['pass_spoken_half_only'] else 'fail (no specialist)') if 'pass_spoken_half_only' in b else 'n/a')
            out.append(f"| {key.split('|')[0]} | {r['held_out']} | {r['n_objects']} | {f(e['decoder (word)'], d)} ({r['head']}, {r['layers']['W']}) | "
                       f"{f(b.get('spoken'), d)} | {f(b.get('specialist'), d)} | {f(e['name prior'], d)} | {f(e['vision tower'], d)} | "
                       f"{f(e['late image tokens'], d)} | {f(e['word, no image'], d)} | {f(e.get('word, mismatched image'), d)} | "
                       f"{f(e['shuffled labels'], d)} | {f(e['random features'], d)} | {f(b.get('spoken_minus_decoder'), d)} | "
                       f"{b['ratio_to_specialist']:.2f} | {verdict} |" if b.get('ratio_to_specialist') is not None else
                       f"| {key.split('|')[0]} | {r['held_out']} | {r['n_objects']} | {f(e['decoder (word)'], d)} ({r['head']}, {r['layers']['W']}) | "
                       f"{f(b.get('spoken'), d)} | {f(b.get('specialist'), d)} | {f(e['name prior'], d)} | {f(e['vision tower'], d)} | "
                       f"{f(e['late image tokens'], d)} | {f(e['word, no image'], d)} | {f(e.get('word, mismatched image'), d)} | "
                       f"{f(e['shuffled labels'], d)} | {f(e['random features'], d)} | {f(b.get('spoken_minus_decoder'), d)} | n/a | {verdict} |")
        out.append('')
        extra = []
        for key, v in M.items():
            if ds in v and 'scale_invariant' in v[ds]:
                si = v[ds]['scale_invariant']
                extra.append(f"- depth, held out {v[ds]['held_out']}: scale-invariant abs-rel " + '; '.join(f"{k} {f(s)}" for k, s in si.items()))
            if ds in v and 'native_box' in v[ds]:
                extra.append(f"- 2D, held out {v[ds]['held_out']}: native box {f(v[ds]['native_box'], 2)}")
        out += extra + ['']
    return out


if __name__ == '__main__':
    main()
