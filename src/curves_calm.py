"""E01 layer curves without the MMA and grappling frames ("calm"). Same probes as results.json's `curves` (fitted on the
pre-registered training split, per-layer test predictions in probe_<model>.json), scored on the calm test objects only.
As a check, the same code recomputes the all-frame curves and they must equal results.json's.

  python src/curves_calm.py --cache CACHE --results experiments/E01-spatial-ask-and-probe/data/results.json \
      --out experiments/E01-spatial-ask-and-probe/data/curves_calm.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyse import COMBAT, MODELS, TRAIN_DOMAINS, Boot, ci, records, score, val


def curves(recs, probe, boot, calm):
    """As analyse.curves, with the test objects optionally restricted to calm frames."""
    out = {}
    for name, layers in probe['layers'].items():
        out[name] = {}
        for t in ('depth', 'xy', 'size', 'centre'):
            T = [i for i, o in enumerate(recs) if o['split'] == 'test' and o['domain'] in TRAIN_DOMAINS[t] and val(o, t) is not None
                 and not (calm and o['activity'] in COMBAT)]
            vals = []
            for L in layers:
                pos = {i: j for j, i in enumerate(L['test_idx'][t])}
                e = score(recs, t, T, [L['test'][t][pos[i]] for i in T])
                pt, bt = boot.mean([recs[i]['id'] for i in T], e)
                vals.append([pt] + ci(bt))
            out[name][t] = vals
        out[name]['n_test'] = {t: len([i for i, o in enumerate(recs) if o['split'] == 'test' and o['domain'] in TRAIN_DOMAINS[t]
                                        and val(o, t) is not None and not (calm and o['activity'] in COMBAT)]) for t in ('depth', 'xy', 'size', 'centre')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    rows = [r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] != 'figure']
    recs = records(rows)
    boot = Boot([o['id'] for o in recs])
    R = json.loads(Path(a.results).read_text())
    out = {'note': 'Per layer: [error, 95% CI low, high] on the calm test objects (no MMA or grappling frames); same shape '
                   'and probes as results.json models.<model>.curves. Errors: abs-rel (depth, size), % of the image diagonal '
                   '(xy), metres (centre). size is trained and tested on indoor only, so its calm and all-frame curves match.',
           'models': {}}
    for m in MODELS:
        probe = json.loads((cache/f'probe_{m}.json').read_text())
        allc = curves(recs, probe, boot, calm=False)
        ref = R['models'][m]['curves']
        worst = max(abs(x - y) for n in ref for t in ref[n] for row, row2 in zip(ref[n][t], allc[n][t]) for x, y in zip(row, row2))
        assert worst < 1e-9, (m, worst)
        out['models'][m] = curves(recs, probe, boot, calm=True)
        print(m, 'all-frame curves reproduce results.json (max diff', worst, ')', flush=True)
        del probe
    Path(a.out).write_text(json.dumps(out, indent=0))
    print('saved', a.out)


if __name__ == '__main__':
    main()
