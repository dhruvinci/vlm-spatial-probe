"""E07 scoring (experiments/E07-transfer-diagnosis/PREREG.md): each arm against the refitted E03 decoder (R), E03's pass
bar against the spoken answer and MoGe-2, and each arm against its control; paired 95% bootstrap over images.

  python src/analyse_e07.py --cache CACHE --e03 CACHE_E03 --out experiments/E07-transfer-diagnosis/data/results_e07.json
  (also writes data/tables_e07.md next to it)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyse import Boot, ci, parse_ask, stat

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B']
DOMS = ['indoor', 'indoor_synth', 'two_people', 'street']
ARMS = {'R': 'R, E03 decoder', 'A1': 'A1, scene type erased', 'A2': 'A2, within the image', 'A3': 'A3, camera given'}
HYP = {'A1': 'H1, scene-type shortcut', 'A2': 'H2, absolute scale (relative geometry transfers)', 'A3': 'H3, missing camera'}


def si(z, y, groups):
    """Scale-invariant abs-rel per object: each image's predictions rescaled by the median of true over predicted."""
    e = np.full(len(y), np.nan)
    for g in groups:
        sc = np.median(y[g] / z[g]); e[g] = np.abs(z[g] * sc - y[g]) / y[g]
    return e


def spoken(asks, o):
    """The spoken depth of one object, or nan."""
    L = asks[o['id']][0] if o['id'] in asks else []
    return L[o['k']].get('depth', np.nan) if o['k'] < len(L) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--e03', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache, c3 = Path(a.cache), Path(a.e03)
    truth = {}
    for p in (cache/'data/dataset.json', c3/'data/dataset.json'):
        for r in json.loads(p.read_text()):
            for k, o in enumerate(r['objects']):
                truth[(r['id'], k)] = o.get('depth_m')
    moge = {**json.loads((cache/'cv_moge.json').read_text()), **json.loads((c3/'cv_moge.json').read_text())}
    e03 = json.loads((Path(a.out).parents[2]/'E03-many-domain-decoder/data/results_e03.json').read_text())
    R_ = {'models': {}}
    f = lambda s, d=3: f"{s['value']:.{d}f} [{s['ci'][0]:.{d}f}, {s['ci'][1]:.{d}f}]" if s and s['value'] == s['value'] else 'n/a'
    lines = ['# E07 tables (printed by src/analyse_e07.py from results_e07.json)', '',
             'Depth. Errors: lower is better (abs-rel; scale-invariant abs-rel for A2 and its comparisons). Intervals: 95% '
             'bootstrap over the held-out images (2,000 resamples), paired. "helps" = the interval of R minus the arm is above 0.', '']
    for model in MODELS:
        p = c3/f'e07_{model}.json'
        if not p.exists():
            R_['models'][model] = {'status': 'not run'}
            continue
        E = json.loads(p.read_text())
        asks = {}
        for q in (cache/'out'/model/'ask.jsonl', c3/'out'/f'e03_{model}'/'ask.jsonl'):
            if q.exists():
                asks.update({x['id']: parse_ask(x) for x in (json.loads(l) for l in q.read_text().splitlines())})
        objs = E['objects']; s_all = np.array(E['angular_size'])
        M = {}
        for held in DOMS:
            F = E['folds'][held]
            te = np.array(F['test_idx'])
            O = [objs[i] for i in te]
            ids = [o['id'] for o in O]
            y = np.array([truth[(o['id'], o['k'])] for o in O], float)
            s = s_all[te]
            groups = list({k: [j for j, x in enumerate(ids) if x == k] for k in dict.fromkeys(ids)}.values())
            sp = np.array([spoken(asks, o) for o in O], float)
            mg = np.array([moge[o['id']]['objects'][o['k']]['depth_m'] or np.nan for o in O], float)
            boot = Boot(ids)
            # images where every object has a spoken depth (or a MoGe-2 depth): the scale-invariant comparisons
            full_sp = np.concatenate([g for g in groups if np.isfinite(sp[g]).all()] or [np.array([], int)]).astype(int)
            full_mg = np.concatenate([g for g in groups if np.isfinite(mg[g]).all()] or [np.array([], int)]).astype(int)
            e_sp, e_mg = np.abs(sp - y) / y, np.abs(mg - y) / y
            si_sp, si_mg = si(np.where(np.isfinite(sp), sp, 1.0), y, groups), si(np.where(np.isfinite(mg), mg, 1.0), y, groups)
            res = {'n_objects': len(te), 'n_images': len(groups), 'e03_layer': F['e03_layer'],
                   'scene_type_classification': F['scene_type_classification'], 'kinds': {}}
            for kind in ('primary', 'exploratory'):
                K = F[kind]
                pr = {arm: np.array(K[arm][K[arm]['head']], float)[:, 0] for arm in ARMS}
                err = {'R': np.abs(np.exp(pr['R']) - y) / y, 'A1': np.abs(np.exp(pr['A1']) - y) / y,
                       'A3': np.abs(np.exp(pr['A3']) / s - y) / y, 'A2': si(np.exp(pr['A2']), y, groups),
                       'R_si': si(np.exp(pr['R']), y, groups)}
                ctl = {'cues2d': si(np.exp(np.array(F['controls']['cues2d']['pred'], float)[:, 0]), y, groups),
                       'name_box_camera': np.abs(np.exp(np.array(F['controls']['name_box_camera']['pred'], float)[:, 0]) / s - y) / y}
                st = lambda e, js=None: stat(*boot.mean([ids[j] for j in (range(len(ids)) if js is None else js)],
                                                       e if js is None else e[js]), len(e) if js is None else len(js))
                out = {'arms': {}, 'controls': {k: st(v) for k, v in ctl.items()}, 'R_scale_invariant': st(err['R_si'])}
                for arm in ARMS:
                    e = err[arm]; ref = err['R_si'] if arm == 'A2' else err['R']
                    A = {'layer': K[arm]['layer'], 'alpha': K[arm]['alpha'], 'head': K[arm]['head'], 'error': st(e)}
                    if arm != 'R':
                        pa, ba = boot.mean(ids, ref); pb, bb = boot.mean(ids, e)
                        d = stat(pa - pb, ba - bb, len(e)); A['R_minus_arm'] = d
                        A['vs_R'] = 'helps' if d['ci'][0] > 0 else 'hurts' if d['ci'][1] < 0 else 'no difference'
                    # E03's bar
                    js = full_sp if arm == 'A2' else np.where(np.isfinite(sp))[0]
                    e_s = si_sp if arm == 'A2' else e_sp
                    bar = {}
                    if len(js):
                        sid = [ids[j] for j in js]
                        p1, b1 = boot.mean(sid, e_s[js]); p2, b2 = boot.mean(sid, e[js])
                        bar.update({'spoken': stat(p1, b1, len(js)), 'arm_on_spoken_objects': stat(p2, b2, len(js)),
                                    'spoken_minus_arm': stat(p1 - p2, b1 - b2, len(js)),
                                    'beats_spoken': bool(ci(b1 - b2)[0] > 0) if len(js) >= 30 else None})
                    jm = full_mg if arm == 'A2' else np.where(np.isfinite(mg))[0]
                    e_m = si_mg if arm == 'A2' else e_mg
                    if len(jm):
                        mid = [ids[j] for j in jm]
                        p1, b1 = boot.mean(mid, e_m[jm]); p2, b2 = boot.mean(mid, e[jm])
                        bar.update({'specialist': stat(p1, b1, len(jm)), 'arm_on_specialist_objects': stat(p2, b2, len(jm)),
                                    'ratio_to_specialist': float(p2 / p1), 'within_1_5x': bool(p2 <= 1.5 * p1) if len(jm) >= 30 else None})
                    if bar.get('beats_spoken') is not None and bar.get('within_1_5x') is not None:
                        bar['pass'] = bar['beats_spoken'] and bar['within_1_5x']
                    A['bar'] = bar
                    if arm in ('A2', 'A3'):
                        c = ctl['cues2d' if arm == 'A2' else 'name_box_camera']
                        pa, ba = boot.mean(ids, c); pb, bb = boot.mean(ids, e)
                        A['control_minus_arm'] = stat(pa - pb, ba - bb, len(e))
                        A['beats_control'] = bool(ci(ba - bb)[0] > 0)
                    out['arms'][arm] = A
                res['kinds'][kind] = out
            pub = e03['models'][model]['folds'][f'depth|{held}']['calm']['error']['decoder (word)']['value']
            r_ = res['kinds']['primary']['arms']['R']['error']['value']
            res['reproduces_e03'] = {'published': pub, 'refit': r_, 'within_5pct': bool(abs(r_ - pub) <= 0.05 * pub)}
            M[held] = res
        R_['models'][model] = {'status': 'run', 'folds': M}
    # reading (pre-registered): over the 8 folds, primary layers
    summary = {}
    for kind in ('primary', 'exploratory'):
        S = {}
        for arm in ('A1', 'A2', 'A3'):
            v = [R_['models'][m]['folds'][h]['kinds'][kind]['arms'][arm] for m in MODELS if R_['models'][m].get('status') == 'run' for h in DOMS]
            helps = sum(x['vs_R'] == 'helps' for x in v); hurts = sum(x['vs_R'] == 'hurts' for x in v)
            verdict = 'supported' if helps >= 4 and hurts == 0 else 'not supported' if helps <= 1 else 'mixed'
            passes = [(m, h) for m in MODELS if R_['models'][m].get('status') == 'run' for h in DOMS
                      if R_['models'][m]['folds'][h]['kinds'][kind]['arms'][arm]['bar'].get('pass')
                      and not R_['models'][m]['folds'][h]['kinds'][kind]['arms']['R']['bar'].get('pass')]
            S[arm] = {'hypothesis': HYP[arm], 'folds': len(v), 'helps': helps, 'hurts': hurts, 'verdict': verdict,
                      'passes_where_R_fails': passes,
                      'helps_and_beats_control': sum(x['vs_R'] == 'helps' and x.get('beats_control', True) for x in v)}
        summary[kind] = S
    R_['reading'] = summary
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R_, indent=1, default=float))
    # tables
    for kind in ('primary', 'exploratory'):
        lines += [f"## {'Primary: E03' if kind == 'primary' else 'Exploratory: re-chosen'} layers", '',
                  '| model | held out | n | arm (layer, head) | error | spoken | MoGe-2 | R minus arm | vs R | spoken minus arm | arm / MoGe-2 | pass | control | control minus arm |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for model in MODELS:
            if R_['models'][model].get('status') != 'run':
                continue
            for held in DOMS:
                r = R_['models'][model]['folds'][held]; K = r['kinds'][kind]
                for arm, A in K['arms'].items():
                    b = A['bar']
                    ctl = ('2D cues ' + f(K['controls']['cues2d'])) if arm == 'A2' else ('box + camera ' + f(K['controls']['name_box_camera'])) if arm == 'A3' else ''
                    lines.append(f"| {model} | {held} | {r['n_objects']} | {ARMS[arm]} ({A['layer']}, {A['head']}) | {f(A['error'])} | "
                                 f"{f(b.get('spoken'))} | {f(b.get('specialist'))} | {f(A.get('R_minus_arm'))} | {A.get('vs_R', '')} | "
                                 f"{f(b.get('spoken_minus_arm'))} | {b.get('ratio_to_specialist', float('nan')):.2f} | "
                                 f"{('PASS' if b['pass'] else 'fail') if 'pass' in b else 'n/a'} | {ctl} | {f(A.get('control_minus_arm'))} |")
                lines.append(f"| {model} | {held} | {r['n_objects']} | R, scale-invariant | {f(K['R_scale_invariant'])} | | | | | | | | | |")
        lines += ['', '**Reading** (pre-registered; 8 folds):', '']
        for arm, S in summary[kind].items():
            lines.append(f"- {ARMS[arm]} ({S['hypothesis']}): helps in {S['helps']}, hurts in {S['hurts']} -> **{S['verdict']}**; "
                         f"passes E03's bar where R fails: {S['passes_where_R_fails'] or 'none'}; helps and beats its control: {S['helps_and_beats_control']}")
        lines.append('')
    lines += ['## Checks', '', '| model | held out | R refit | E03 published | within 5% | scene type named, before erasure | after | majority share |', '|---|---|---|---|---|---|---|---|']
    for model in MODELS:
        if R_['models'][model].get('status') != 'run':
            continue
        for held in DOMS:
            r = R_['models'][model]['folds'][held]; c = r['scene_type_classification']; q = r['reproduces_e03']
            lines.append(f"| {model} | {held} | {q['refit']:.4f} | {q['published']:.4f} | {q['within_5pct']} | {c['before']['best_alpha_accuracy']:.3f} | "
                         f"{c['after']['best_alpha_accuracy']:.3f} | {c['after']['chance_majority']:.3f} |")
    (Path(a.out).parent/'tables_e07.md').write_text('\n'.join(lines) + '\n')
    print('saved', a.out)


if __name__ == '__main__':
    main()
