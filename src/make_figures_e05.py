"""E05 figure from experiments/E05-dense-vision-head/data/results_e05.json (no model is run here).

  python src/make_figures_e05.py --results RESULTS --out docs/assets
"""
import argparse
import json
from pathlib import Path

from make_figures import AQUA, INK, INK2, ORANGE, plt_setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3), gridspec_kw={'wspace': 0.6})
    panels = [('Depth per pixel, NYU test scenes\nabs-rel (lower is better)', [('head on Qwen3.8 tower', R['depth_head']['abs_rel'], R['depth_head']['abs_rel_ci'], ORANGE),
                                                                                 ('MoGe-2', R['depth_moge2']['abs_rel'], R['depth_moge2']['abs_rel_ci'], AQUA)]),
              ('Joints, COCO val people\nPCK at 5% of the box diagonal (higher is better)', [('head on Qwen3.8 tower', R['joints_head_pck5']['value'], R['joints_head_pck5']['ci'], ORANGE),
                                                                                             ('ViTPose+ base', R['joints_vitpose_pck5']['value'], R['joints_vitpose_pck5']['ci'], AQUA)]),
              ('Joints, COCO val people\nPCK at 10% of the box diagonal (higher is better)', [('head on Qwen3.8 tower', R['joints_head_pck10']['value'], R['joints_head_pck10']['ci'], ORANGE),
                                                                                              ('ViTPose+ base', R['joints_vitpose_pck10']['value'], R['joints_vitpose_pck10']['ci'], AQUA)])]
    for ax, (title, rows) in zip(axes, panels):
        for y, (lab, v, c, col) in enumerate(rows[::-1]):
            ax.barh(y, v, height=0.6, color=col)
            ax.plot(c, [y, y], color=INK2, lw=1)
            ax.text(c[1], y, f'  {v:.3f}', va='center', fontsize=8, color=INK2)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
        ax.set_title(title, loc='left', fontsize=9.5); ax.grid(axis='y', visible=False)
        ax.set_xlim(0, max(r[2][1] for r in rows) * 1.3)
    fig.suptitle('E05: trained heads on Qwen3.8-27B\'s frozen vision tower against the specialists (95% intervals)', x=0.05, ha='left',
                 fontsize=11, fontweight='bold', color=INK)
    fig.savefig(Path(a.out)/'fig8_dense_head.png', dpi=130, bbox_inches='tight')
    print('saved fig8_dense_head.png')


if __name__ == '__main__':
    main()
