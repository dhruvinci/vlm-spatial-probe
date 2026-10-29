"""E06 tables: experiments/E06-prompt-robustness/data/tables_e06.md from results_e06.json.

  python src/report_e06.py --results experiments/E06-prompt-robustness/data/results_e06.json
"""
import argparse
import json
from pathlib import Path


def f(s, d=3):
    return f"{s['value']:.{d}f} [{s['ci'][0]:.{d}f}, {s['ci'][1]:.{d}f}]" if s else 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    L = ['# E06 tables (printed by src/report_e06.py from results_e06.json)', '',
         'Wording 1 = E01; wording 2 = the second wording. abs-rel: lower is better; balanced accuracy: higher is better.', '',
         '## Pre-registered criteria', '']
    for k, v in R['criteria'].items():
        L.append(f"- {k}: **{'holds' if v['holds'] else 'FAILS'}** " + '; '.join(
            f"{kk} {f(vv) if isinstance(vv, dict) else round(vv, 3)}" for kk, vv in v.items() if kk != 'holds'))
    for ds in ('calm', 'all'):
        L += ['', f'## Per model, {"without MMA and grappling (paper-facing)" if ds == "calm" else "all frames"}', '',
              '| model | measure | n | wording 1 | wording 2 | 2 minus 1 | name prior |', '|---|---|---|---|---|---|---|']
        for m, r in R['models'].items():
            if r.get('status') != 'run':
                continue
            for k, v in r[ds].items():
                L.append(f"| {m} | {k} | {v['n']} | {f(v['wording_1'])} | {f(v['wording_2'])} | {f(v['wording_2_minus_1'])} | {f(v.get('name_prior'))} |")
    Path(a.results).with_name('tables_e06.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
