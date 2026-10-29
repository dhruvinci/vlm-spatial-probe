"""E03's "does variety help?" (pre-registered): on the held-out objects both E02 and E03 tested (E01's indoor and street
objects, calm frames), the E03 decoder's error minus the E02 decoder's, paired, 95% bootstrap over images.

  python src/compare_e02_e03.py --cache CACHE --e03 CACHE_E03 --out experiments/E03-many-domain-decoder/data/variety.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyse import COMBAT, Boot, ci, stat
from transfer import errors, records


def preds(E, recs, measure, held):
    g = 'size' if measure == 'size' else 'geo'
    F = E['folds'].get(f'{g}|{held}')
    if not F:
        return {}
    T = F['targets'][measure]['W']
    P = np.array(T[T.get('head', 'ridge')], float)
    return {(recs[i]['id'], recs[i]['k']): P[j] for j, i in enumerate(F['test_idx'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--e03', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache, c3 = Path(a.cache), Path(a.e03)
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure']
    orient = json.loads((cache/'e02_orientation.json').read_text())['objects']
    recs2 = records(rows, orient)
    calm = [r for r in rows if r.get('activity') not in COMBAT]
    recs3 = records(calm, orient) + records(json.loads((c3/'data/dataset.json').read_text()), {})
    look = {(o['id'], o['k']): o for o in recs2}
    out = {}
    for model in ('Qwen3.8-27B', 'SpatialAxiom-9B'):
        if not (c3/f'e03_{model}.json').exists():
            continue
        E2 = json.loads((cache/f'e02_{model}.json').read_text()); E3 = json.loads((c3/f'e03_{model}.json').read_text())
        for measure in ('depth', 'centre', 'size'):
            for held in ('indoor', 'street'):
                p2, p3 = preds(E2, recs2, measure, held), preds(E3, recs3, measure, held)
                keys = sorted(k for k in p2 if k in p3 and look[k]['activity'] not in COMBAT)
                if len(keys) < 30:
                    continue
                objs = [look[k] for k in keys]
                tmp = objs                                              # errors() reads truth from the record list
                e2 = errors(measure, np.array([p2[k] for k in keys]), tmp, list(range(len(tmp))))
                e3 = errors(measure, np.array([p3[k] for k in keys]), tmp, list(range(len(tmp))))
                ids = [k[0] for k in keys]
                boot = Boot(ids)
                a2, b2 = boot.mean(ids, e2); a3, b3 = boot.mean(ids, e3)
                out[f'{model}|{measure}|{held}'] = {'n': len(keys), 'e02_decoder': stat(a2, b2, len(keys)), 'e03_decoder': stat(a3, b3, len(keys)),
                                                    'e03_minus_e02': stat(a3 - a2, b3 - b2, len(keys)), 'variety_helps': bool(ci(b3 - b2)[1] < 0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, round(v['e02_decoder']['value'], 3), '->', round(v['e03_decoder']['value'], 3), v['e03_minus_e02']['ci'], v['variety_helps'])


if __name__ == '__main__':
    main()
