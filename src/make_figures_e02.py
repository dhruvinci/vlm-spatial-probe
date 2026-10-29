"""E02 figures from experiments/E02-decoder-transfer/data/results_e02.json (no model is run here).

  fig6_transfer_<model>.png  per measure and held-out domain: decoder (object word) vs spoken answer vs specialist vs
                              name prior (calm: without MMA and grappling frames)
  fig7_pass_grid.png         the pre-registered pass bar, every model x measure x held-out domain

  python src/make_figures_e02.py --results RESULTS --out docs/assets
"""
import argparse
import json
from pathlib import Path

import numpy as np

from make_figures import AQUA, BLUE, GREY, INK, INK2, ORANGE, SURF, plt_setup

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
MEASURES = [('depth', 'Distance from the camera (abs-rel)'), ('centre', '3D position (relative error)'),
            ('distance', 'Distance between two objects (abs-rel)'), ('size', 'Size (abs-rel)'), ('xy', 'Position in the image (% diagonal)')]
DOMAIN = {'people': 'people (COCO)', 'indoor': 'indoor (NYU)', 'indoor_synth': 'synthetic indoor (Hypersim)',
          'two_people': 'two people (Harmony4D)', 'street': 'street (KITTI)'}
TAG = ''


def transfer_figure(plt, R, model, out, ds='calm'):
    F = R['models'][model]['folds']
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(19.5, 4.8), gridspec_kw={'wspace': 0.45})
    series = [('name prior', GREY, lambda r: r['error'].get('name prior')),
              ('spoken answer', INK, lambda r: r['bar'].get('spoken')),
              ('decoder on the object word', ORANGE, lambda r: r['error'].get('decoder (word)')),
              ('specialist (MoGe-2 / RF-DETR)', AQUA, lambda r: r['bar'].get('specialist'))]
    for ax, (m, title) in zip(axes, MEASURES):
        keys = [k for k in F if k.startswith(m + '|') and ds in F[k]]
        y0 = np.arange(len(keys))[::-1] * (len(series) + 1)
        cap = 0
        for yk, k in zip(y0, keys):
            r = F[k][ds]
            for j, (lab, col, get) in enumerate(series):
                s = get(r)
                if not s or s.get('value') is None or s['value'] != s['value']:
                    continue
                yy = yk + (len(series) - 1 - j)
                ax.barh(yy, s['value'], height=0.8, color=col, label=lab if k == keys[0] else None)
                ax.plot(s['ci'], [yy, yy], color=INK2, lw=0.8)
                cap = max(cap, s['ci'][1])
            b = r['bar']
            mark = 'pass' if b.get('pass') else ('fail' if 'pass' in b else ('beats spoken' if b.get('pass_spoken_half_only') else ''))
            ax.text(0, yk + len(series) - 0.2, f"held out: {DOMAIN[r['held_out']]}  {mark}", fontsize=7.5, color=INK, va='bottom')
        ax.set_yticks([]); ax.grid(axis='y', visible=False)
        ax.set_xlim(0, cap * 1.08 if cap else 1)
        ax.set_title(title.replace(' (', '\n('), loc='left', fontsize=9.5)
        ax.set_xlabel('lower is better', fontsize=7.5)
    axes[0].legend(loc='lower right', fontsize=7.5)
    fig.suptitle(f'{model}: a decoder trained on the other domains, tested on a domain it never saw (without MMA and grappling frames; 95% intervals)',
                 x=0.05, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/f'fig6_transfer{TAG}_{model}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def pass_grid(plt, R, out, ds='calm'):
    run = [m for m in MODELS if R['models'].get(m, {}).get('status') == 'run']
    cols = [k for k in R['models'][run[0]]['folds'] if not k.startswith('orient')]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 0.9 + 0.5 * len(run)))
    colours = {'pass': BLUE, 'fail': '#e4e3df', 'spoken only': '#9ec5f4', 'n/a': SURF}
    for i, m in enumerate(run):
        for j, k in enumerate(cols):
            b = R['models'][m]['folds'].get(k, {}).get(ds, {}).get('bar', {})
            st = ('pass' if b.get('pass') else 'fail') if 'pass' in b else (
                ('spoken only' if b['pass_spoken_half_only'] else 'fail') if 'pass_spoken_half_only' in b else 'n/a')
            ax.add_patch(plt.Rectangle((j, len(run) - 1 - i), 0.94, 0.9, color=colours[st]))
            ax.text(j + 0.47, len(run) - 1 - i + 0.45, {'pass': 'pass', 'fail': 'fail', 'spoken only': 'beats\nspoken', 'n/a': 'n/a'}[st],
                    ha='center', va='center', fontsize=6.5, color='white' if st == 'pass' else INK)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(run))
    ax.set_xticks(np.arange(len(cols)) + 0.47)
    short = {'people': 'people', 'indoor': 'indoor', 'indoor_synth': 'synth indoor', 'two_people': '2 people', 'street': 'street'}
    ax.set_xticklabels([f"{k.split('|')[0]}\n{short[k.split('|')[1]]}" for k in cols], fontsize=6.5)
    ax.set_yticks(np.arange(len(run)) + 0.45); ax.set_yticklabels(run[::-1], fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Pre-registered bar: the decoder beats the spoken answer and is within 1.5x the specialist (held-out domain; calm frames)',
                 loc='left', fontsize=9.5)
    fig.savefig(out/f'fig7_pass_grid{TAG}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='', help="'_e03' for the E03 figures")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    R = json.loads(Path(a.results).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    for m in MODELS:
        if R['models'].get(m, {}).get('status') == 'run':
            transfer_figure(plt, R, m, out)
    pass_grid(plt, R, out)
    print('figures in', out)


if __name__ == '__main__':
    main()
