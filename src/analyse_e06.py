"""E06 (experiments/E06-prompt-robustness/PREREG.md): the spoken answers under a second wording, against E01's wording,
paired on the same objects and items; the two pre-registered robustness criteria.

  python src/analyse_e06.py --cache CACHE --e06 CACHE_E03 --out experiments/E06-prompt-robustness/data/results_e06.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyse import COMBAT, TRAIN_DOMAINS, Boot, ci, err_absrel, parse_ask, prior_pred, records, stat, val

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--e06', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache, c6 = Path(a.cache), Path(a.e06)
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure']
    byid = {r['id']: r for r in rows}
    recs = records(rows)
    boot = Boot([o['id'] for o in recs])
    asks = {}
    R = {'models': {}, 'criteria': {}}
    for m in MODELS:
        f1, f2 = cache/'out'/m/'ask.jsonl', c6/'out'/f'e06_{m}'/'ask2.jsonl'
        if not f2.exists():
            R['models'][m] = {'status': 'not run'}
            continue
        p1 = {x['id']: parse_ask(x) for x in (json.loads(l) for l in f1.read_text().splitlines())}
        p2 = {x['id']: parse_ask(x) for x in (json.loads(l) for l in f2.read_text().splitlines())}
        asks[m] = (p1, p2)
        M = {}
        for ds in ('all', 'calm'):
            D = {}
            for target in ('depth', 'size'):
                for split, doms in (('test', TRAIN_DOMAINS[target]), ('heldout', ('street',))):
                    idx = [i for i, o in enumerate(recs) if o['split'] == split and o['domain'] in doms and val(o, target) is not None
                           and (ds == 'all' or o['activity'] not in COMBAT)]
                    have = [i for i in idx if all(target in P[recs[i]['id']][0][recs[i]['k']] for P in (p1, p2) if recs[i]['id'] in P)
                            and recs[i]['id'] in p1 and recs[i]['id'] in p2]
                    if not have:
                        continue
                    ids = [recs[i]['id'] for i in have]
                    e1 = err_absrel([recs[i][target] for i in have], [p1[recs[i]['id']][0][recs[i]['k']][target] for i in have])
                    e2 = err_absrel([recs[i][target] for i in have], [p2[recs[i]['id']][0][recs[i]['k']][target] for i in have])
                    eP = err_absrel([recs[i][target] for i in have], np.exp(prior_pred(recs, target, have).reshape(-1)))
                    a1, b1 = boot.mean(ids, e1); a2, b2 = boot.mean(ids, e2); aP, bP = boot.mean(ids, eP)
                    D[f'{target}_{split}'] = {'n': len(have), 'coverage_of_asked': len(have) / len(idx), 'wording_1': stat(a1, b1, len(have)),
                                              'wording_2': stat(a2, b2, len(have)), 'name_prior': stat(aP, bP, len(have)),
                                              'wording_2_minus_1': stat(a2 - a1, b2 - b1, len(have)),
                                              'prior_minus_wording_2': stat(aP - a2, bP - b2, len(have)),
                                              'wording_2_gain_over_prior': float((aP - a2) / aP),
                                              'wording_2_is_measurement': bool((aP - a2) / aP >= 0.2 and ci(bP - b2)[0] > 0)}
            y1 = {(x['id'], x['relation'], x['a'], x['b']): x for x in (json.loads(l) for l in (cache/'out'/m/'yesno.jsonl').read_text().splitlines()) if x['kind'] == 'relation'}
            y2 = {(x['id'], x['relation'], x['a'], x['b']): x for x in (json.loads(l) for l in (c6/'out'/f'e06_{m}'/'yesno2.jsonl').read_text().splitlines())}
            keys = [k for k in y1 if k in y2 and (ds == 'all' or byid[k[0]].get('activity') not in COMBAT)]
            for scope, kk in (('pooled', keys), ('pooled_test', [k for k in keys if byid[k[0]]['domain'] != 'street'])):
                ids = [k[0] for k in kk]; truth = [y1[k]['truth'] for k in kk]
                c1 = [(y1[k]['logp_yes'] > y1[k]['logp_no']) == y1[k]['truth'] for k in kk]
                c2 = [(y2[k]['logp_yes'] > y2[k]['logp_no']) == y2[k]['truth'] for k in kk]
                q1, w1 = boot.balanced_acc(ids, c1, truth); q2, w2 = boot.balanced_acc(ids, c2, truth)
                D[f'relations_{scope}'] = {'n': len(kk), 'wording_1': stat(q1, w1, len(kk)), 'wording_2': stat(q2, w2, len(kk)),
                                           'wording_2_minus_1': stat(q2 - q1, w2 - w1, len(kk)),
                                           'yes_rate_1': float(np.mean([y1[k]['logp_yes'] > y1[k]['logp_no'] for k in kk])),
                                           'yes_rate_2': float(np.mean([y2[k]['logp_yes'] > y2[k]['logp_no'] for k in kk]))}
            M[ds] = D
        R['models'][m] = {'status': 'run', **M}
    # criterion 1: Qwen3.8-27B's asked depth is still "qualitative" under wording 2
    for ds in ('all', 'calm'):
        q = R['models'].get('Qwen3.8-27B', {}).get(ds, {}).get('depth_test')
        if q:
            R['criteria'][f'1_qwen38_depth_still_qualitative_{ds}'] = {'gain_over_prior': q['wording_2_gain_over_prior'],
                                                                       'prior_minus_wording_2': q['prior_minus_wording_2'],
                                                                       'holds': not q['wording_2_is_measurement']}
    # criterion 2: SpatialAxiom-9B still beats Qwen3.5-9B on asked depth under wording 2 (paired, same objects)
    if 'SpatialAxiom-9B' in asks and 'Qwen3.5-9B' in asks:
        for ds in ('all', 'calm'):
            idx = [i for i, o in enumerate(recs) if o['split'] == 'test' and o['domain'] in TRAIN_DOMAINS['depth'] and o['depth']
                   and (ds == 'all' or o['activity'] not in COMBAT)]
            have = [i for i in idx if all(recs[i]['id'] in asks[m][1] and 'depth' in asks[m][1][recs[i]['id']][0][recs[i]['k']]
                                          for m in ('SpatialAxiom-9B', 'Qwen3.5-9B'))]
            ids = [recs[i]['id'] for i in have]
            e = {m: err_absrel([recs[i]['depth'] for i in have], [asks[m][1][recs[i]['id']][0][recs[i]['k']]['depth'] for i in have])
                 for m in ('SpatialAxiom-9B', 'Qwen3.5-9B')}
            pa, ba = boot.mean(ids, e['SpatialAxiom-9B']); pb, bb = boot.mean(ids, e['Qwen3.5-9B'])
            R['criteria'][f'2_spatialaxiom_beats_base_{ds}'] = {'SpatialAxiom-9B': stat(pa, ba, len(have)), 'Qwen3.5-9B': stat(pb, bb, len(have)),
                                                                 'difference': stat(pa - pb, ba - bb, len(have)), 'holds': bool(ci(ba - bb)[1] < 0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(json.dumps(R['criteria'], indent=1))


if __name__ == '__main__':
    main()
