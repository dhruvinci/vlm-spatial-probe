"""E04 figures from experiments/E04-more-measures/data/results_e04.json (no model is run here).

  fig10_measures_<model>.png  per target: the name prior, the three probes (vision tower, image tokens in the late
                              language layers, object word), the spoken answer and the specialist, on the test split,
                              with 95% intervals; each panel's title gives the pre-registered outcome
  fig11_measures_models.png   the primary targets across the five models: prior, object word, spoken

  python src/make_figures_e04.py --results RESULTS --out docs/assets
"""
import argparse
import json
from pathlib import Path

import numpy as np

from make_figures import AQUA, BLUE, GREY, INK, INK2, ORANGE, plt_setup

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
TITLE = {'facing': 'Facing (degrees)', 'joints': 'Elbow and knee angles (degrees)', 'normal': 'Surface normal (degrees)',
         'surface_class': 'Surface class (1 - balanced acc.)', 'dims': 'Dimensions (abs-rel)', 'speed': 'Speed (m/s)',
         'heading': 'Heading, moving objects (degrees)', 'count': 'Count (absolute error)'}
PRIMARY = ['facing', 'normal', 'speed', 'count']
SHORT = {'falsified': 'falsified', 'supported': 'supported', 'partial': 'partial',
         'premise fails (G1): the tower does not carry the measure': 'premise fails (G1)',
         'controls failed (G0): not interpreted': 'controls failed (G0)'}


def bars(ax, rows_):
    y = np.arange(len(rows_))[::-1]
    for yy, (lab, s, c) in zip(y, rows_):
        ax.barh(yy, s['value'], height=0.6, color=c)
        ax.plot(s['ci'], [yy, yy], color=INK2, lw=1)
        ax.text(s['ci'][1], yy, f"  {s['value']:.2f}" if s['value'] < 10 else f"  {s['value']:.1f}", va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows_], fontsize=8)
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, max(r[1]['ci'][1] for r in rows_) * 1.3)


def per_model(plt, R, model, out):
    T = R['models'][model]['targets']
    keys = [t for t in TITLE if t in T]
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 6.8), gridspec_kw={'wspace': 0.9, 'hspace': 0.6})
    for ax, t in zip(axes.ravel(), keys):
        e, q, sp = T[t]['error'], T[t]['Q1'], T[t]['specialist']
        rows_ = [(lab, s, c) for lab, s, c in [
            ('name prior (no image)', e.get('P'), GREY), ('probe: vision tower', e.get('V'), BLUE),
            ('probe: image tokens, late layers', e.get('L_img'), BLUE), ('probe: object word', e.get('W'), ORANGE),
            ('asked: spoken answer', q.get('spoken'), INK),
            ('specialist: ' + ('MoGe-2' if t == 'normal' else 'RF-DETR'), sp.get('specialist'), AQUA)]
            if s and s.get('value') == s.get('value') and s.get('value') is not None]
        bars(ax, rows_)
        tag = SHORT.get(T[t]['outcome'], T[t]['outcome'])
        ax.set_title(f"{TITLE[t]}{' *' if t in PRIMARY else ''}\n{tag}; n = {T[t]['n_test']}", loc='left', fontsize=9)
        cov = q.get('coverage')
        ax.set_xlabel('lower is better' + (f'; spoken coverage {cov:.0%}' if q.get('spoken') else ''), fontsize=7.5)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    fig.suptitle(f'E04, {model}: reading the hidden states against asking, on measures E01 did not cover '
                 '(test split; 95% intervals; * primary)', x=0.06, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/f'fig10_measures_{model}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('saved', out/f'fig10_measures_{model}.png')


def across_models(plt, R, out):
    run = [m for m in MODELS if R['models'].get(m, {}).get('status') == 'run']
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 0.9 + 1.25 * len(run)), gridspec_kw={'wspace': 0.95})
    for ax, t in zip(axes, PRIMARY):
        rows_ = []
        for m in run:
            T = R['models'][m]['targets'].get(t)
            if not T:
                continue
            for lab, s, c in [(f'{m}: prior', T['error'].get('P'), GREY), (f'{m}: object word', T['error'].get('W'), ORANGE),
                              (f'{m}: spoken', T['Q1'].get('spoken'), INK)]:
                if s and s.get('value') == s.get('value') and s.get('value') is not None:
                    rows_.append((lab, s, c))
        if rows_:
            bars(ax, rows_)
        ax.set_title(TITLE[t], loc='left', fontsize=9.5)
        ax.set_xlabel('lower is better', fontsize=7.5)
    fig.suptitle('E04: the primary measures across the five models (test split; 95% intervals)', x=0.06, ha='left',
                 fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/'fig11_measures_models.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('saved', out/'fig11_measures_models.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    plt = plt_setup()
    out = Path(a.out)
    for m in MODELS:
        if R['models'].get(m, {}).get('status') == 'run':
            per_model(plt, R, m, out)
    across_models(plt, R, out)


if __name__ == '__main__':
    main()
