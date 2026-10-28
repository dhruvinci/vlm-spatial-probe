"""Figures for E01 from experiments/E01-spatial-ask-and-probe/data/results.json (and the cache for Figure 1 and the
dense maps). No model is run here.

  python src/make_figures.py --results RESULTS.json --cache CACHE --out docs/assets
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
GREY, INK, INK2, SURF, GRID = '#9a9993', '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'
MODELS = ['Qwen3.8-27B', 'SpatialAxiom-9B', 'Qwen3.5-9B', 'SpaceQwen2.5-VL-3B', 'Qwen2.5-VL-3B']
MODEL_COLOUR = {'Qwen3.8-27B': BLUE, 'SpatialAxiom-9B': ORANGE, 'SpaceQwen2.5-VL-3B': ORANGE, 'Qwen3.5-9B': GREY, 'Qwen2.5-VL-3B': GREY}
SHORT = {'Qwen3.8-27B': 'Qwen3.8-27B', 'SpatialAxiom-9B': 'SpatialAxiom-9B (spatial)', 'Qwen3.5-9B': 'Qwen3.5-9B (its base)',
         'SpaceQwen2.5-VL-3B': 'SpaceQwen2.5-VL-3B (spatial)', 'Qwen2.5-VL-3B': 'Qwen2.5-VL-3B (its base)'}


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                         'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': 0.6, 'legend.frameon': False})
    return plt


# ------------------------------------------------------------------ Figure 1: the baseball photo
def figure1(plt, R, cache, out):
    rows = json.loads((cache/'data/dataset.json').read_text())
    fig_row = next(r for r in rows if r['domain'] == 'figure')
    img = Image.open(cache/'data'/fig_row['image']).convert('RGB')
    W, H = img.size
    run = [m for m in MODELS if R['models'].get(m, {}).get('status') == 'run']
    n = len(run)
    cols = 3
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(12.6, 4.6 * rws), gridspec_kw={'wspace': 0.04, 'hspace': 0.42})
    axes = np.atleast_1d(axes).ravel()
    short = {'player standing on the right': 'standing', 'player diving to the base': 'diving',
             'player in the background': 'background', "standing player's glove": 'glove'}
    from matplotlib.patches import Rectangle
    for ax, model in zip(axes, run):
        ax.imshow(np.asarray(img) / 255 * 0.75 + 0.25, extent=(0, W, H, 0))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ask = R['models'][model]['ask']
        parsed = ask.get('parsed_figure') or [{}] * len(fig_row['objects'])
        ground = ask.get('ground_figure') or [None] * len(fig_row['objects'])
        lines = []
        for k, o in enumerate(fig_row['objects']):
            x1, y1, x2, y2 = (np.array(o['box']) * [W, H, W, H])
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec='white', lw=1.6))
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=INK, lw=0.8, ls='--'))
            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
            p = parsed[k] if k < len(parsed) else {}
            if p.get('xy'):
                px, py = p['xy'][0] * W, p['xy'][1] * H
                inside = 0 <= px <= W and 0 <= py <= H
                px, py = min(max(px, 0), W), min(max(py, 0), H)
                ax.plot([tx, px], [ty, py], color=BLUE, lw=1.2)
                ax.plot(px, py, 'o', ms=8, mfc=BLUE if inside else 'white', mec=BLUE if not inside else 'white', mew=1.5)
            if k < len(ground) and ground[k]:
                gx, gy = ground[k][0] * W, ground[k][1] * H
                ax.plot(gx, gy, 's', ms=7, mfc='none', mec=ORANGE, mew=2)
            ax.text(x1 + 3, y1 - 4, short[o['name']], fontsize=7.5, color=INK, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
            d = p.get('depth')
            lines.append(f"{short[o['name']]} {d:.3g} m" if d else f"{short[o['name']]}: none")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        off = any(q.get('xy') and max(q['xy']) > 1 for q in parsed)
        ax.set_title(model + ('\ncentres came back above 100: drawn at the edge' if off else '\n'), loc='left', fontsize=9.5)
        ax.text(0, -0.03, 'asked distance from the camera:\n' + ', '.join(lines[:2]) + ',\n' + ', '.join(lines[2:]),
                transform=ax.transAxes, fontsize=8, color=INK2, va='top')
    for ax in axes[n:]:
        ax.axis('off')
    if n < len(axes):
        ax = axes[n]
        ax.plot([], [], 'o', ms=8, mfc=BLUE, mec='white', label='centre it gave when asked (percent of width, height)')
        ax.plot([], [], 's', ms=7, mfc='none', mec=ORANGE, mew=2, label="centre of its own box (native 'bbox_2d' format)")
        ax.plot([], [], '--', color=INK, lw=0.8, label='true box (COCO annotation)')
        ax.legend(loc='center left', fontsize=8.5)
        ax.text(0.0, 0.25, 'No metric depth truth exists for this photo;\nthe background player is clearly the farthest.',
                transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')
    fig.suptitle('Figure 1. Asked where four things are and how far away they are', x=0.125, ha='left', fontsize=12,
                 fontweight='bold', color=INK)
    fig.text(0.125, 0.01, 'Photo: "Zach MacPhee Dives Back into First" by Barry Stahl, CC BY 2.0 (Flickr 5505334912; '
             'COCO train2017 246285). Greedy decoding, no thinking.', fontsize=8, color=INK2)
    fig.savefig(out/'fig1_baseball.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: the layer curves
def layer_curves(plt, R, model, out, dense=None):
    res = R['models'][model]
    cur = res['curves']
    meta = res['meta']
    nv, nl = meta['n_vis_layers'], meta['n_lm_layers']
    pr = res['probe']['all']
    targets = [('depth', 'Distance from the camera: abs-rel error', 'abs-rel (lower is better)'),
               ('xy', 'Position in the image: centre error', '% of image diagonal (lower is better)'),
               ('size', 'Physical size (indoor): abs-rel error', 'abs-rel (lower is better)')]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), gridspec_kw={'wspace': 0.22})
    xv = np.arange(nv); xl = nv + 4 + np.arange(nl)
    for ax, (t, title, yl) in zip(axes, targets):
        def band(x, arr, colour, label, ls='-', lw=2.0):
            a = np.array(arr, float)
            ax.fill_between(x, a[:, 1], a[:, 2], color=colour, alpha=0.08, lw=0)
            ax.plot(x, a[:, 0], ls, color=colour, lw=lw, label=label)
        band(xv, cur['vis'][t], BLUE, 'image patches / tokens')
        band(xl, cur['lmimg'][t], BLUE, None)
        band(xl[1:], cur['word'][t][1:], ORANGE, 'object word, with the image')
        if 'word_mis' in cur:
            band(xl[1:], cur['word_mis'][t][1:], AQUA, 'object word, mismatched image', lw=1.4)
        if 'word_noimg' in cur:
            band(xl[1:], cur['word_noimg'][t][1:], GREY, 'object word, no image', ls='--', lw=1.4)
        P = pr[t]['error']['P']['value']
        A_ = res.get('ask', {}).get('all', {}).get(f'{t}_test', {})
        ask = A_.get('ask')
        top = max(np.nanmax(np.array(cur['word_noimg'][t])[1:, 0]) if 'word_noimg' in cur else 0, P) * 1.25
        ax.axhline(P, color=INK2, lw=1, ls=':')
        ask_above = ask and abs(ask['value'] - P) < 0.04 * top and ask['value'] <= P
        ax.text(xl[-1], P * (1.03 if ask_above else 1.0), ' name prior', color=INK2, fontsize=7.5, va='bottom' if ask_above else 'center', ha='left')
        if ask and ask['value'] < top:
            ax.axhline(ask['value'], color=INK, lw=1, ls='--')
            ax.text(xl[-1], ask['value'] * (0.97 if ask_above else 1.0), ' asked (text)', color=INK, fontsize=7.5, va='top' if ask_above else 'center', ha='left')
        elif ask:
            rs = A_.get('ask_rescaled_exploratory')
            ax.text(xl[2], top * 0.97, f"asked (text): {ask['value']:.0f}%, off scale\n({A_.get('share_above_100', 0):.0%} of answers above 100)"
                    + (f";\nread as 0-1000 (exploratory): {rs['value']:.1f}%" if rs else ''), fontsize=7.5, color=INK, va='top')
            if rs:
                ax.axhline(rs['value'], color=INK, lw=1, ls='--')
                ax.text(xl[-1], rs['value'], ' asked*', color=INK, fontsize=7.5, va='center', ha='left')
        mg = A_.get('MoGe-2')
        if mg:
            ax.axhline(mg['value'], color=INK, lw=1, ls='-.')
            ax.text(xl[-1], mg['value'], ' MoGe-2', color=INK, fontsize=7.5, va='center', ha='left')
        ax.axvline(nv, color=INK2, lw=0.8)
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.set_ylabel(yl)
        ticks = [0, nv // 2, nv - 1] + list(xl[::16]) + [xl[-1]]
        ax.set_xticks(ticks); ax.set_xticklabels([f'v{i}' for i in (0, nv // 2, nv - 1)] + [str(i) for i in range(0, nl, 16)] + [str(nl - 1)], fontsize=7.5)
        ax.set_xlabel('vision blocks (v)  |  language layers', fontsize=8)
        ax.set_ylim(0, top)
        ax.set_xlim(-1, xl[-1] + 1)
    axes[0].legend(loc='lower left', fontsize=7.5)
    fig.suptitle(f'{model}: where the measurement can be read, layer by layer (linear probes; test split; 95% intervals)',
                 x=0.09, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/f'fig2_layers_{model}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3: ask vs probe
def ask_vs_probe(plt, R, model, out):
    res = R['models'][model]
    A = res['ask']['all']
    panels = [('depth_test', 'Distance from the camera (abs-rel)'), ('xy_test', 'Position in the image (% diagonal)'),
              ('size_test', 'Physical size, indoor (abs-rel)')]
    names = [('P', 'name prior (no image)', GREY), ('ask', 'asked: text answer', INK), ('ground', "asked: native box", INK),
             ('V', 'probe: vision tower', BLUE), ('L_img', 'probe: image tokens, late language layers', BLUE),
             ('W', 'probe: object word', ORANGE), ('MoGe-2', 'specialist: MoGe-2', AQUA), ('RF-DETR', 'specialist: RF-DETR', AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.6), gridspec_kw={'wspace': 0.65})
    for ax, (key, title) in zip(axes, panels):
        d = dict(A.get(key, {}))
        names_ = list(names)
        if key == 'xy_test' and 'ask_rescaled_exploratory' in d:
            names_.insert(2, ('ask_rescaled_exploratory', 'asked: text, rescaled*', INK2))
        rows_ = [(lab, d[k], c) for k, lab, c in names_ if k in d]
        cap = max(r[1]['ci'][1] for r in rows_ if r[1]['value'] < 30) * 1.3 if key == 'xy_test' else None
        y = np.arange(len(rows_))[::-1]
        for yy, (lab, s, c) in zip(y, rows_):
            if cap and s['value'] > cap:
                ax.barh(yy, cap, height=0.6, color=c)
                ax.text(cap, yy, f"  {s['value']:.0f} (off scale)", va='center', fontsize=7.5, color=INK2)
                continue
            ax.barh(yy, s['value'], height=0.6, color=c)
            ax.plot(s['ci'], [yy, yy], color=INK2, lw=1)
            ax.text(s['ci'][1], yy, f"  {s['value']:.2f}" if s['value'] < 10 else f"  {s['value']:.1f}", va='center', fontsize=7.5, color=INK2)
        if cap:
            ax.set_xlim(0, cap * 1.45)
            ax.text(0, -0.28, f"{d.get('share_above_100', 0):.0%} of the asked centres came back above 100" + ("; * exploratory: read as 0-1000" if 'ask_rescaled_exploratory' in d else ''),
                    transform=ax.transAxes, fontsize=7, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows_], fontsize=8)
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.grid(axis='y', visible=False)
        cov = d.get('coverage')
        if cov is not None:
            ax.set_xlabel(f"lower is better; {d['ask']['n'] if 'ask' in d else 0} objects answered ({cov:.0%} of asked)", fontsize=7.5)
    fig.suptitle(f'{model}: asking against reading the hidden states (same test objects)', x=0.06, ha='left',
                 fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/f'fig3_ask_vs_probe_{model}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4: spatial models vs bases vs Qwen3.8
def models_figure(plt, R, out, dataset='all'):
    run = [m for m in MODELS if R['models'].get(m, {}).get('status') == 'run']
    metrics = [('ask', 'depth_test', 'Distance from camera (abs-rel)', False), ('ask', 'size_test', 'Size, indoor (abs-rel)', False),
               ('ask', 'distance_test', 'Distance between two objects (abs-rel)', False),
               ('ground', 'xy_test', 'Position, native box centre (% diagonal)', False),
               ('yesno', 'pooled_test/all', 'Relations, yes/no (balanced acc.)', True), ('yesno', 'naming/pooled', 'Naming check (accuracy)', True)]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.4), gridspec_kw={'wspace': 0.75, 'hspace': 0.55})
    for ax, (src, key, title, higher) in zip(axes.ravel(), metrics):
        vals = []
        for m in run:
            d = R['models'][m].get('ask' if src == 'ground' else src, {}).get(dataset, {}).get(key, {})
            s = d.get('ask') if src == 'ask' else d.get('ground') if src == 'ground' else d
            if s and 'value' in s:
                vals.append((m, s))
        y = np.arange(len(vals))[::-1]
        for yy, (m, s) in zip(y, vals):
            ax.barh(yy, s['value'], height=0.6, color=MODEL_COLOUR[m])
            ax.plot(s['ci'], [yy, yy], color=INK2, lw=1)
            ax.text(s['ci'][1], yy, f"  {s['value']:.2f}" if s['value'] < 10 else f"  {s['value']:.1f}", va='center', fontsize=7.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m, _ in vals], fontsize=8)
        if higher:
            ax.axvline(0.5, color=INK2, lw=0.8, ls=':')
            ax.set_xlim(0.4, 1.0)
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.set_xlabel('higher is better (chance 0.5)' if higher else 'lower is better', fontsize=7.5)
        ax.grid(axis='y', visible=False)
    fig.suptitle('Spatially trained models against their bases and Qwen3.8-27B (asked; test split; 95% intervals)',
                 x=0.06, ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/'fig4_models.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ Figure 5 (exploratory): dense depth by layer
def dense_figure(plt, cache, model, out):
    p = cache/f'dense_{model}.json'
    if not p.exists():
        return
    D = json.loads(p.read_text())
    rows = {r['id']: r for r in json.loads((cache/'data/dataset.json').read_text())}
    ex = D['example']
    hm, wm = ex['grid']
    fig = plt.figure(figsize=(14.0, 3.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.7, 1, 1, 1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    nv = max(x['layer'] for x in D['vis']) + 1
    xv = [x['layer'] for x in D['vis']]; xl = [nv + 1 + x['layer'] for x in D['lm']]
    for xs, arr, c in ((xv, D['vis'], BLUE), (xl, D['lm'], BLUE)):
        a = np.array([[x['abs_rel'], *x['ci']] for x in arr])
        ax.fill_between(xs, a[:, 1], a[:, 2], color=c, alpha=0.12, lw=0)
        ax.plot(xs, a[:, 0], '-o', color=c, ms=3, lw=2)
    for k, lab, ls in (('train_mean', 'training mean', ':'), ('position_only', 'cell position alone', '--')):
        v = D['baselines'][k]['abs_rel']
        ax.axhline(v, color=INK2, lw=1, ls=ls); ax.text(xl[-1], v, ' ' + lab, fontsize=7.5, color=INK2, va='center')
    ax.axvline(nv, color=INK2, lw=0.8)
    ax.set_title('Per-patch depth, NYU test scenes (abs-rel)', loc='left', fontsize=9.5)
    ax.set_xlabel('vision blocks  |  language layers', fontsize=8)
    ax.set_ylim(0, None)
    img = Image.open(cache/'data'/rows[ex['id']]['image']).convert('RGB')
    truth = np.array(ex['truth']).reshape(hm, wm)
    vmax = np.nanmax(np.where(truth > 0, truth, np.nan))
    best_v = min(D['vis'], key=lambda x: x['abs_rel'])['layer']; last_l = D['lm'][-1]['layer']
    panels = [('image', None), ('Kinect depth (per cell)', truth), (f'decoded: vision block {best_v}', np.array(ex['pred'][f'vis{best_v}']).reshape(hm, wm)),
              (f'decoded: language layer {last_l}', np.array(ex['pred'][f'lm{last_l}']).reshape(hm, wm))]
    for j, (title, arr) in enumerate(panels):
        a2 = fig.add_subplot(gs[j + 1])
        a2.set_xticks([]); a2.set_yticks([]); a2.grid(False)
        if arr is None:
            a2.imshow(img)
        else:
            a2.imshow(np.where(arr > 0, arr, np.nan), cmap='Blues_r', vmin=0, vmax=vmax)
        a2.set_title(title, fontsize=8.5, loc='left')
    fig.suptitle(f'{model} (exploratory, not pre-registered): depth read from each image token, layer by layer', x=0.06,
                 ha='left', fontsize=11, fontweight='bold', color=INK)
    fig.savefig(out/f'fig5_dense_{model}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    figure1(plt, R, cache, out)
    for m in MODELS:
        if R['models'].get(m, {}).get('status') == 'run' and 'curves' in R['models'][m]:
            layer_curves(plt, R, m, out)
            ask_vs_probe(plt, R, m, out)
            dense_figure(plt, cache, m, out)
    models_figure(plt, R, out)
    print('figures in', out)


if __name__ == '__main__':
    main()
