"""E07 figure from experiments/E07-transfer-diagnosis/data/results_e07.json (no model is run here).

  fig9_diagnosis.png  per model and held-out scene type: absolute depth (R, A1 scene type erased, A3 camera given, the
                      box + camera control, spoken, MoGe-2) and depth within the image (R, A2, the 2D-cues control,
                      spoken, MoGe-2; scale-invariant). Primary layers (E03's).

  python src/make_figures_e07.py --results RESULTS --out docs/assets
"""
import argparse
import json
from pathlib import Path

import numpy as np

from make_figures import AQUA, BLUE, GREY, INK, INK2, MAGENTA, ORANGE, YELLOW, plt_setup

MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B']
DOMAIN = {'indoor': 'indoor (NYU)', 'indoor_synth': 'synthetic indoor (Hypersim)', 'two_people': 'two people (Harmony4D)',
          'street': 'street (KITTI)'}


def panel(ax, F, series, title):
    keys = [k for k in DOMAIN if k in F]
    y0 = np.arange(len(keys))[::-1] * (len(series) + 2)
    cap = 0
    for yk, k in zip(y0, keys):
        K = F[k]['kinds']['primary']
        for j, (lab, col, get) in enumerate(series):
            s = get(K)
            if not s or s.get('value') is None or s['value'] != s['value']:
                continue
            yy = yk + (len(series) - 1 - j)
            ax.barh(yy, s['value'], height=0.8, color=col, label=lab if k == keys[0] else None)
            ax.plot(s['ci'], [yy, yy], color=INK2, lw=0.8)
            cap = max(cap, s['ci'][1])
        ax.text(0, yk + len(series) - 0.2, f'held out: {DOMAIN[k]}', fontsize=7.5, color=INK, va='bottom')
    ax.set_yticks([]); ax.grid(axis='y', visible=False)
    ax.set_ylim(-1, y0[0] + len(series) + 1.2)
    ax.set_xlim(0, cap * 1.08 if cap else 1)
    ax.set_title(title, loc='left', fontsize=9.5, pad=8)
    ax.set_xlabel('lower is better', fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    run = [m for m in MODELS if R['models'].get(m, {}).get('status') == 'run']
    plt = plt_setup()
    arm = lambda x: (lambda K: K['arms'][x]['error'])
    absolute = [('spoken answer', INK, lambda K: K['arms']['R']['bar'].get('spoken')),
                ('R: E03 decoder', ORANGE, arm('R')),
                ('A1: scene type erased', MAGENTA, arm('A1')),
                ('A3: camera given', YELLOW, arm('A3')),
                ('control: box + camera, no model', GREY, lambda K: K['controls']['name_box_camera']),
                ('MoGe-2', AQUA, lambda K: K['arms']['R']['bar'].get('specialist'))]
    relative = [('spoken answer', INK, lambda K: K['arms']['A2']['bar'].get('spoken')),
                ('R: E03 decoder', ORANGE, lambda K: K['R_scale_invariant']),
                ('A2: within the image', BLUE, arm('A2')),
                ('control: 2D cues of the box', GREY, lambda K: K['controls']['cues2d']),
                ('MoGe-2', AQUA, lambda K: K['arms']['A2']['bar'].get('specialist'))]
    fig, axes = plt.subplots(len(run), 2, figsize=(13.0, 6.0 * len(run)), gridspec_kw={'wspace': 0.3, 'hspace': 0.35}, squeeze=False)
    for i, m in enumerate(run):
        F = R['models'][m]['folds']
        panel(axes[i, 0], F, absolute, f'{m}: distance from the camera (abs-rel)')
        panel(axes[i, 1], F, relative, f'{m}: depth within the image (scale-invariant abs-rel)')
    for j in range(2):
        axes[-1, j].legend(*axes[0, j].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.12),
                           ncol=3, fontsize=8, frameon=False)
    fig.suptitle('E07: why the decoder does not transfer (held-out scene type; E03\'s layers; 95% intervals)',
                 x=0.05, y=0.995, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.savefig(Path(a.out)/'fig9_diagnosis.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('saved', Path(a.out)/'fig9_diagnosis.png')


if __name__ == '__main__':
    main()
