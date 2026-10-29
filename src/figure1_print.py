"""Figure 1, one model's panel at print resolution (no title strip), and its per-object values as JSON for the caption.

  python src/figure1_print.py --cache CACHE --model Qwen3.8-27B --out docs/assets \
      --values experiments/E01-spatial-ask-and-probe/data/figure1_Qwen3.8-27B.json
  (writes fig1_baseball_<model>_print.png at 300 dpi and .pdf, about 6 inches wide)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from analyse import json_blocks, match, parse_ask
from make_figures import BLUE, INK, INK2, ORANGE, plt_setup

SHORT = {'player standing on the right': 'standing', 'player diving to the base': 'diving',
         'player in the background': 'background', "standing player's glove": 'glove'}
CREDIT = '"Zach MacPhee Dives Back into First" by Barry Stahl, CC BY 2.0 (Flickr 5505334912; COCO train2017 246285)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--model', default='Qwen3.8-27B')
    ap.add_argument('--out', required=True)
    ap.add_argument('--values', required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    row = next(r for r in json.loads((cache/'data/dataset.json').read_text()) if r['domain'] == 'figure')
    mdir = cache/'out'/a.model
    ask = next(x for x in (json.loads(l) for l in (mdir/'ask.jsonl').read_text().splitlines()) if x['id'] == row['id'])
    grd = next(x for x in (json.loads(l) for l in (mdir/'ground.jsonl').read_text().splitlines()) if x['id'] == row['id'])
    assert json.loads((mdir/'meta.json').read_text())['kind'] == 'qwen3_5'            # boxes are relative, 0-1000
    parsed, pairs, _ = parse_ask(ask)
    if not pairs:                                   # the answer hit the generation limit inside "pairs": keep the complete ones
        for blk in re.findall(r'\{[^{}]*"distance_m"[^{}]*\}', ask['answer']):
            p_ = json.loads(blk)
            a_, b_ = match([o['name'] for o in row['objects']], p_['a']), match([o['name'] for o in row['objects']], p_['b'])
            if a_ is not None and b_ is not None and a_ != b_:
                pairs[(min(a_, b_), max(a_, b_))] = p_['distance_m']
    d, _ = json_blocks(grd['answer'])
    boxes = [None] * len(row['objects'])
    for j, o in enumerate(d if isinstance(d, list) else []):
        if isinstance(o, dict) and 'bbox_2d' in o and j < len(boxes):
            boxes[j] = [v / 1000 for v in o['bbox_2d']]
    img = Image.open(cache/'data'/row['image']).convert('RGB')
    W, H = img.size
    objs = []
    for k, o in enumerate(row['objects']):
        p = parsed[k]
        objs.append({'name': o['name'], 'true_box': o['box'],
                     'asked_centre_percent': [round(100 * v, 4) for v in p['xy']] if 'xy' in p else None,
                     'asked_distance_from_camera_m': p.get('depth'), 'asked_size_m': p.get('size'),
                     'native_box': boxes[k], 'native_box_centre': [(boxes[k][0] + boxes[k][2]) / 2, (boxes[k][1] + boxes[k][3]) / 2] if boxes[k] else None})
    names = [o['name'] for o in row['objects']]
    vals = {'model': a.model, 'image': row['id'], 'image_size_px': [W, H], 'credit': CREDIT,
            'units': 'boxes and native-box centres: fractions of width and height (x1, y1, x2, y2); asked centres: percent of '
                     'width and height, as the model wrote them; distances in metres. No metric truth exists for this photo.',
            'objects': objs,
            'asked_pair_distances_m': [{'a': names[i], 'b': names[j], 'distance_m': v} for (i, j), v in sorted(pairs.items())],
            'answer_complete': not ask['answer'].rstrip().rstrip('`').rstrip().endswith(('distance', ',', ':')) and ask['answer'].count('{') == ask['answer'].count('}'),
            'raw_answer': ask['answer'], 'raw_native_boxes': grd['answer']}
    Path(a.values).write_text(json.dumps(vals, indent=1))
    plt = plt_setup()
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(6.0, 6.0 * H / W + 0.6))
    ax = fig.add_axes([0, 0.6 / (6.0 * H / W + 0.6), 1, 6.0 * H / W / (6.0 * H / W + 0.6)])
    ax.imshow(np.asarray(img) / 255 * 0.75 + 0.25, extent=(0, W, H, 0), interpolation='lanczos')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for o, v in zip(row['objects'], objs):
        x1, y1, x2, y2 = np.array(o['box']) * [W, H, W, H]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec='white', lw=1.4))
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=INK, lw=0.7, ls='--'))
        tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
        if v['asked_centre_percent']:
            px, py = v['asked_centre_percent'][0] / 100 * W, v['asked_centre_percent'][1] / 100 * H
            ax.plot([tx, px], [ty, py], color=BLUE, lw=1.0)
            ax.plot(px, py, 'o', ms=6.5, mfc=BLUE, mec='white', mew=1.2)
        if v['native_box_centre']:
            ax.plot(v['native_box_centre'][0] * W, v['native_box_centre'][1] * H, 's', ms=6, mfc='none', mec=ORANGE, mew=1.6)
        dist = v['asked_distance_from_camera_m']
        ax.text(x1 + 3, y1 - 4, f"{SHORT[o['name']]}: {dist:.3g} m" if dist else SHORT[o['name']], fontsize=7, color=INK,
                va='bottom', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    lg = fig.add_axes([0, 0, 1, 0.55 / (6.0 * H / W + 0.6)]); lg.axis('off')
    lg.plot([], [], '--', color=INK, lw=0.7, label='true box')
    lg.plot([], [], 'o', ms=6.5, mfc=BLUE, mec='white', label='centre given when asked')
    lg.plot([], [], 's', ms=6, mfc='none', mec=ORANGE, mew=1.6, label="centre of its own box")
    lg.legend(loc='upper left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5, frameon=False, handlelength=1.6, columnspacing=1.2)
    lg.text(0.012, 0.12, 'Labels: the distance from the camera it gave when asked, in metres.', fontsize=7.5, color=INK2, va='bottom', transform=lg.transAxes)
    out = Path(a.out)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'fig1_baseball_{a.model}_print.{ext}', dpi=300)
    plt.close(fig)
    print('saved', out/f'fig1_baseball_{a.model}_print.png', '.pdf', a.values)


if __name__ == '__main__':
    main()
