"""Token-layout fixture audit (no model weights, no inference): for each model's installed image processor, feed an
image whose pixels encode their own patch row and column, and check where each patch lands in the token sequence.

Checks, per processor and grid:
  - raw vision patches come in 2x2-grouped order (a direct reshape is NOT image order: the earlier probe's bug, which
    misplaced 14,080 of 14,400 locations on a 160x90 grid);
  - restore_grid() puts every raw patch back at its image row and column;
  - every run of merge*merge consecutive raw patches is one merged token's block, and merged tokens follow raster
    order over the merged grid (so merged token k sits at (k // (W/merge), k % (W/merge))).

  python src/layout_audit.py --model-dir DIR [--model-dir DIR ...] --out experiments/E00-token-layout-audit/data/audit.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def restore_grid(tokens, height, width, merge=2):
    """Raw patches in grouped order -> [height, width, ...] in image order."""
    a = np.asarray(tokens)
    if a.ndim != 2 or height % merge or width % merge or len(a) != height * width:
        raise ValueError('invalid token/grid/grouping contract')
    return a.reshape(height // merge, width // merge, merge, merge, a.shape[-1]).transpose(0, 2, 1, 3, 4).reshape(height, width, -1)


def merged_cell(k, width, merge=2):
    """Merged token k -> (row, column) on the merged grid."""
    return divmod(k, width // merge)


def audit(model_dir):
    import transformers
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    patch, merge, temporal = int(proc.patch_size), int(proc.merge_size), int(proc.temporal_patch_size)
    records = []
    for gh, gw in ((4, 6), (8, 4), (20, 30), (160, 90)):
        yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
        im = np.zeros((gh * patch, gw * patch, 3), np.uint8)
        im[..., 0] = np.repeat(np.repeat(yy % 256, patch, 0), patch, 1)
        im[..., 1] = np.repeat(np.repeat(xx % 256, patch, 0), patch, 1)
        im[..., 2] = np.repeat(np.repeat(yy // 256 * 16 + xx // 256, patch, 0), patch, 1)
        out = proc(images=Image.fromarray(im), do_resize=False, do_rescale=False, do_normalize=False, return_tensors='pt')
        thw = out['image_grid_thw'][0].tolist()
        pv = out['pixel_values'].numpy()
        tok = pv[:, [0, temporal * patch * patch, 2 * temporal * patch * patch]]          # R, G, B at each patch's (0, 0)
        expected = np.stack([yy % 256, xx % 256, yy // 256 * 16 + xx // 256], -1)
        tok = tok.astype(np.int64)
        rows = tok[:, 0] + 256 * (tok[:, 2] // 16); cols = tok[:, 1] + 256 * (tok[:, 2] % 16)    # each raw patch's true cell
        mm = merge * merge
        groups_ok = all(len({(int(r) // merge, int(c) // merge) for r, c in zip(rows[k * mm:(k + 1) * mm], cols[k * mm:(k + 1) * mm])}) == 1
                        and merged_cell(k, gw, merge) == (int(rows[k * mm]) // merge, int(cols[k * mm]) // merge)
                        for k in range(gh * gw // mm))
        legacy = tok.reshape(gh, gw, 3)
        fixed = restore_grid(tok, gh, gw, merge)
        records.append({'grid_hw': [gh, gw], 'processor_grid_thw': thw, 'total_locations': gh * gw,
                        'direct_reshape_misplaced': int(np.any(legacy != expected, -1).sum()),
                        'restore_grid_misplaced': int(np.any(fixed != expected, -1).sum()),
                        'merged_tokens_raster_order': bool(groups_ok)})
    ok = all(r['restore_grid_misplaced'] == 0 and r['merged_tokens_raster_order'] and r['processor_grid_thw'][1:] == r['grid_hw'] for r in records)
    return {'model_dir': Path(model_dir).name, 'transformers': transformers.__version__, 'processor': type(proc).__name__,
            'patch_size': patch, 'merge_size': merge, 'temporal_patch_size': temporal, 'fixtures': records, 'pass': ok}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', action='append', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = [audit(d) for d in a.model_dir]
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r in res:
        print(r['model_dir'], r['processor'], 'patch', r['patch_size'], 'merge', r['merge_size'], 'PASS' if r['pass'] else 'FAIL',
              [(f['grid_hw'], f['direct_reshape_misplaced'], f['restore_grid_misplaced'], f['merged_tokens_raster_order']) for f in r['fixtures']])
    if not all(r['pass'] for r in res):
        raise SystemExit(1)
