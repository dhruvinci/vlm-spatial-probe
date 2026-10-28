"""Markdown tables for RESULTS.md, printed straight from results.json (so the text never retypes a number).

  python src/report_tables.py --results experiments/E01-spatial-ask-and-probe/data/results.json > experiments/E01-spatial-ask-and-probe/data/tables.md
"""
import argparse
import json

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
FAM = {'P': 'name prior', 'V': 'vision tower', 'L_img': 'image tokens, last quarter of language layers',
       'L_img_any': 'image tokens, any language layer', 'W': 'object word, with the image',
       'W_noimg': 'object word, no image', 'W_mis': 'object word, mismatched image',
       'V_shuffled': 'control: shuffled labels (tower layer)', 'V_random': 'control: norm-matched random features'}


def ci(s, d=3):
    if not s or 'value' not in s:
        return 'n/a'
    return f"{s['value']:.{d}f} [{s['ci'][0]:.{d}f}, {s['ci'][1]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    a = ap.parse_args()
    R = json.loads(open(a.results).read())
    out = []
    p = out.append
    for model in MODELS:
        m = R['models'].get(model, {})
        if m.get('status') != 'run':
            p(f'\n## {model}: not run\n')
            continue
        p(f'\n## {model}\n')
        for ds, label in (('all', 'all test frames (pre-registered)'), ('calm', 'without MMA and grappling frames')):
            if 'probe' not in m:
                continue
            p(f'\n### Probes, {label}\n')
            for t, unit, d in (('depth', 'abs-rel', 3), ('xy', '% of diagonal', 2), ('size', 'abs-rel', 3), ('centre', '3D error, m', 3)):
                q = m['probe'][ds][t]
                p(f"\n**{t}** ({unit}, lower is better; {q['n_objects']} test objects in {q['n_images']} images; layers chosen on training data: "
                  + ', '.join(f'{k} {v}' for k, v in q['layers'].items()) + ')\n')
                p('| readout | error [95% CI] | R (share of the tower\'s gain over the prior) [95% CI] |')
                p('|---|---|---|')
                for k in ('P', 'V', 'L_img', 'L_img_any', 'W', 'W_noimg', 'W_mis', 'V_shuffled', 'V_random'):
                    if k in q['error']:
                        r = q['recovery'].get(k)
                        rs = f"{r['R']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if r else ''
                        p(f"| {FAM[k]} | {ci(q['error'][k], d)} | {rs} |")
                if t == 'depth':
                    p('\nδ<1.25 (higher is better): ' + '; '.join(f"{FAM[k]} {ci(v, 3)}" for k, v in q['delta125'].items()))
                g0 = '; '.join(f"{FAM[k]}: P minus control {ci(v['P_minus_control'], 3)} -> {'pass' if v['pass'] else 'FAIL'}" for k, v in q['G0'].items())
                p(f"\nG0 {g0}. G1: tower gain over the prior {q['G1']['relative_gain']:.1%} -> {'pass' if q['G1']['pass'] else 'FAIL'}."
                  + (f" **Verdict: {q['verdict']}.**" if q.get('verdict') else ''))
            rel = m['probe'][ds]['relations']
            p('\n**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)\n')
            p('| readout | closer | left of | above | bigger | pooled |')
            p('|---|---|---|---|---|---|')
            for f, dd in rel.items():
                p(f"| {FAM[f]} | " + ' | '.join(ci(dd.get(r), 2) + (f" (n={dd[r]['n']})" if r in dd else '') for r in ('closer', 'left_of', 'above', 'bigger', 'pooled')) + ' |')
            dist = m['probe'][ds]['distance']
            p('\n**Distance between two objects from the probes\' 3D centres** (abs-rel; test pairs): '
              + '; '.join(f"{FAM[k]} {ci(v)} (n={v['n']})" for k, v in dist.items()))
        A = m.get('ask', {})
        if 'all' in A:
            for ds, label in (('all', 'all frames'), ('calm', 'without MMA and grappling')):
                p(f'\n### Asked answers, {label} (same objects for every column)\n')
                p(f"Answers: {A['n_answers']} images, {A['valid_json']} parsed as whole JSON.\n")
                p('| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |')
                p('|---|---|---|---|---|---|---|---|')
                for k, v in A[ds].items():
                    if 'ask' not in v:
                        p(f"| {k} | {v.get('coverage', 0):.0%} of {v['n_asked']} | n/a | | | | | |")
                        continue
                    d = 2 if k.startswith('xy') else 3
                    spec = v.get('MoGe-2') or v.get('RF-DETR')
                    sname = 'MoGe-2' if 'MoGe-2' in v else 'RF-DETR' if 'RF-DETR' in v else ''
                    p(f"| {k} | {v['coverage']:.0%} of {v['n_asked']} | {ci(v['ask'], d)} | {ci(v.get('P'), d)} | {ci(v.get('V'), d)} | "
                      f"{ci(v.get('L_img'), d)} | {ci(v.get('W'), d)} | {sname + ' ' + ci(spec, d) if spec else ''} |")
                for k, v in A[ds].items():
                    extra = []
                    if 'ask_rescaled_exploratory' in v:
                        extra.append(f"answers above 100: {v['share_above_100']:.0%}; median error {v['median_ask']:.2f}; read as 0-1000 (exploratory) {ci(v['ask_rescaled_exploratory'], 2)}")
                    if 'ground' in v:
                        extra.append(f"native boxes {ci(v['ground'], 2)} (n={v['ground']['n']}, {v['ground_coverage']:.0%} returned)")
                    if 'P_minus_ask' in v:
                        extra.append(f"prior minus asked {ci(v['P_minus_ask'], d if False else 3)}; asked beats the prior by {v['ask_gain_over_P']:.0%} -> "
                                     + ('a measurement (>= 20%, CI excludes 0)' if v['ask_is_measurement'] else 'qualitative (pre-registered rule)'))
                    if 'delta125_ask' in v:
                        extra.append(f"asked δ<1.25 {ci(v['delta125_ask'])}; median asked/true {v['median_ratio_ask']:.2f}")
                    if extra:
                        p(f"- {k}: " + '; '.join(extra))
        Y = m.get('yesno', {})
        if 'all' in Y:
            for ds, label in (('all', 'all frames'), ('calm', 'without MMA and grappling')):
                p(f'\n### Yes/no questions, {label} (balanced accuracy for relations, accuracy for naming; chance 0.5)\n')
                p('| set | score [95% CI] | n | share answered yes |')
                p('|---|---|---|---|')
                for k, v in Y[ds].items():
                    flag = ' (n<30: no gate)' if v['n'] < 30 else ''
                    p(f"| {k} | {ci(v, 3)} | {v['n']}{flag} | {v['yes_rate']:.2f} |")
    P = R.get('paired', {})
    if P:
        p('\n## Paired model comparisons (asked; same items; difference = first minus second)\n')
        for name, res in P.items():
            for ds in ('all', 'calm'):
                if ds not in res:
                    continue
                p(f'\n**{name}**, {"all frames" if ds == "all" else "without MMA and grappling"}\n')
                p('| measure | first | second | difference [95% CI] |')
                p('|---|---|---|---|')
                A_, B_ = name.split(' vs ')
                for k, v in res[ds].items():
                    d = 2 if k.startswith('xy') else 3
                    p(f"| {k} | {ci(v[A_], d)} | {ci(v[B_], d)} | {ci(v['diff_A_minus_B'], d)} (n={v['diff_A_minus_B']['n']}) |")
    print('\n'.join(out))


if __name__ == '__main__':
    main()
