"""E04 part hypersim (surfaces), step 1 of 2: frame and object selection, images and boxes (no HDF5 needed).

Hypersim (Roberts et al., ICCV 2021; CC BY-SA 3.0), read by HTTP range from Apple's per-scene zips. Pre-registration
(vlm-spatial-probe experiments/E04-more-measures/PREREG.md): 600 frames from at least 60 public scenes, at most 10 frames
per scene, camera cam_00, one object per frame of class table, desk, counter, bed, cabinet, shelves, night stand,
dresser or bookshelf (NYU40), whose instance covers at least 2% of the image and whose class has exactly one instance in
the frame.

The selection reads the release's own preview PNGs (images/scene_cam_00_geometry_preview/frame.NNNN.semantic.png and
.semantic_instance.png). Hypersim writes these from the same integer arrays as the .hdf5 files, through a colour
lookup (code/python/tools/scene_generate_images_semantic_segmentation.py: colour per id, black for -1), so a class is
its NYU40 colour and an instance is one colour. Step 2 (hypersim_truth.py, needs h5py) re-derives every mask from the
.hdf5 files, checks it pixel for pixel against the count recorded here, and computes the truth.

Order: public scenes shuffled (seed 2604); in each, the public cam_00 frames shuffled (seed 2604 + scene), taken in
that order while they qualify, up to 10 per scene; one object per frame, the largest qualifying one. 600 rows plus 60
reserve rows (later scenes) in a fixed order: step 2 takes rows in order, drops any that fail its checks, and stops at
600, so a dropped row is replaced by the next one.

Writes built_e04/hypersim_selection.json and built_e04/images/hypersim/<id>.jpg (the tonemapped preview, 640 px).
Run: ~/Code/kakashi-research/dataset/.venv-depth/bin/python -I built_e04/code/hypersim.py
"""
import csv
import io
import json
import random
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, SEED, RemoteZip, save_image  # noqa: E402

PART = 'hypersim'
URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip'
SPLIT_CSV = ('https://raw.githubusercontent.com/apple/ml-hypersim/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/'
             'evermotion_dataset/analysis/metadata_images_split_scene_v1.csv')
CAM_CSV = OUT.parent/'hypersim_meta'/'metadata_camera_parameters.csv'
N_MAIN, N_RESERVE, PER_SCENE, MIN_SHARE = 600, 60, 10, 0.02
# NYU40 id -> (prompt name, colour from code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv)
NYU40_RGB = {1: (174, 199, 232), 2: (152, 223, 138), 3: (31, 119, 180), 4: (255, 187, 120), 5: (188, 189, 34),
             6: (140, 86, 75), 7: (255, 152, 150), 8: (214, 39, 40), 9: (197, 176, 213), 10: (148, 103, 189),
             11: (196, 156, 148), 12: (23, 190, 207), 13: (178, 76, 76), 14: (247, 182, 210), 15: (66, 188, 102),
             16: (219, 219, 141), 17: (140, 57, 197), 18: (202, 185, 52), 19: (51, 176, 203), 20: (200, 54, 131),
             21: (92, 193, 61), 22: (78, 71, 183), 23: (172, 114, 82), 24: (255, 127, 14), 25: (91, 163, 138),
             26: (153, 98, 156), 27: (140, 153, 101), 28: (158, 218, 229), 29: (100, 125, 154), 30: (178, 127, 135),
             31: (120, 185, 128), 32: (146, 111, 194), 33: (44, 160, 44), 34: (112, 128, 144), 35: (96, 207, 209),
             36: (227, 119, 194), 37: (213, 92, 176), 38: (94, 106, 211), 39: (82, 84, 163), 40: (100, 85, 144)}
TARGET = {7: 'table', 14: 'desk', 12: 'counter', 4: 'bed', 3: 'cabinet', 15: 'shelves', 32: 'night stand',
          17: 'dresser', 10: 'bookshelf'}
GEOM = '{s}/images/scene_cam_00_geometry_hdf5/frame.{f:04d}.{k}.hdf5'


def code(rgb):
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


SEM_CODE = {(r << 16) | (g << 8) | b: k for k, (r, g, b) in NYU40_RGB.items()}


def frame_objects(sem_png, ins_png):
    """Qualifying objects in one frame from the two preview PNGs, or None if the semantic PNG has a colour outside NYU40."""
    sc = code(np.asarray(Image.open(io.BytesIO(sem_png)).convert('RGB')))
    ic = code(np.asarray(Image.open(io.BytesIO(ins_png)).convert('RGB')))
    if sc.shape != ic.shape:
        return None
    present = set(np.unique(sc).tolist()) - {0}
    if not present <= set(SEM_CODE):
        return None
    H, W = sc.shape
    out = []
    for colour in present:
        c = SEM_CODE[colour]
        if c not in TARGET:
            continue
        region = sc == colour
        inst = [v for v in np.unique(ic[region]).tolist() if v != 0]
        if len(inst) != 1:
            continue
        m = region & (ic == inst[0])
        if m.sum() < MIN_SHARE * H * W:
            continue
        ys, xs = np.nonzero(m)
        out.append({'name': TARGET[c], 'nyu40_id': c, 'mask_px': int(m.sum()), 'area_frac': float(m.mean()),
                    'box': [xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H],
                    'png_instance_rgb': [inst[0] >> 16, (inst[0] >> 8) & 255, inst[0] & 255], 'png_size': [W, H]})
    return out


def main():
    rows_csv = list(csv.DictReader(io.StringIO(urllib.request.urlopen(SPLIT_CSV, timeout=120).read().decode())))
    frames = {}
    for r in rows_csv:
        if r['included_in_public_release'] == 'True' and r['camera_name'] == 'cam_00':
            frames.setdefault(r['scene_name'], []).append(int(r['frame_id']))
    scale = {r['scene_name']: float(r['settings_units_info_meters_scale']) for r in csv.DictReader(open(CAM_CSV))}
    scenes = sorted(s for s in frames if s in scale)
    random.Random(SEED).shuffle(scenes)
    print('public cam_00 scenes', len(scenes), 'frames', sum(len(frames[s]) for s in scenes), flush=True)

    rows, n_scenes, skipped = [], 0, Counter()
    for scene in scenes:
        if len(rows) >= N_MAIN + N_RESERVE:
            break
        try:
            z = RemoteZip(URL.format(scene), 1 << 18)
        except Exception as e:
            skipped['zip unreadable'] += 1
            print(scene, 'zip unreadable', repr(e)[:120], flush=True)
            continue
        meta = {r[0].strip(): r[1].strip() for r in csv.reader(io.StringIO(z.read(f'{scene}/_detail/metadata_scene.csv').decode()))
                if len(r) >= 2}
        mpu = float(meta.get('meters_per_asset_unit', 'nan'))
        if not abs(mpu - scale[scene]) <= 1e-6 * max(1.0, abs(mpu)):
            skipped['scale mismatch'] += 1
            print(scene, 'scale mismatch', mpu, scale[scene], flush=True)
            continue
        fr = sorted(frames[scene])
        random.Random(f'{SEED}:{scene}').shuffle(fr)
        prev = f'{scene}/images/scene_cam_00_geometry_preview/frame.{{:04d}}.{{}}.png'

        def scan(f):
            try:
                return f, frame_objects(z.read(prev.format(f, 'semantic')), z.read(prev.format(f, 'semantic_instance')))
            except KeyError:
                return f, None

        with ThreadPoolExecutor(8) as ex:
            found = dict(ex.map(scan, fr))
        taken = 0
        for f in fr:
            if taken >= PER_SCENE or len(rows) >= N_MAIN + N_RESERVE:
                break
            objs = found[f]
            if not objs:
                continue
            need = [f'{scene}/images/scene_cam_00_final_preview/frame.{f:04d}.tonemap.jpg'] + \
                   [GEOM.format(s=scene, f=f, k=k) for k in ('semantic', 'semantic_instance', 'normal_cam', 'normal_world', 'position')]
            if not all(n in z.info for n in need):
                skipped['missing members'] += 1
                continue
            o = max(objs, key=lambda o: o['mask_px'])
            rid = f'{PART}_{scene}_{f:04d}'
            img = Image.open(io.BytesIO(z.read(need[0])))
            size = save_image(img, OUT/'images'/PART/f'{rid}.jpg')
            rows.append({'id': rid, 'measure': 'surfaces', 'domain': PART, 'image': f'images/{PART}/{rid}.jpg', 'size': size,
                         'group': scene, 'order': len(rows), 'reserve': len(rows) >= N_MAIN,
                         'source': {'dataset': 'Hypersim', 'scene': scene, 'camera': 'cam_00', 'frame': f,
                                    'licence': 'CC BY-SA 3.0', 'licence_url': 'https://creativecommons.org/licenses/by-sa/3.0/'},
                         'meters_per_asset_unit': mpu,
                         'members': {'normal_cam': need[3], 'normal_world': need[4], 'position': need[5],
                                     'semantic': need[1], 'semantic_instance': need[2],
                                     'camera_positions': f'{scene}/_detail/cam_00/camera_keyframe_positions.hdf5',
                                     'bbox_extents': f'{scene}/_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5'},
                         'objects': [o]})
            taken += 1
        n_scenes += taken > 0
        print(scene, 'frames scanned', len(fr), 'qualifying', sum(bool(v) for v in found.values()), 'taken', taken,
              'rows', len(rows), flush=True)

    (OUT/f'{PART}_selection.json').write_text(json.dumps(rows, indent=0))
    main_rows = [r for r in rows if not r['reserve']]
    print('rows', len(rows), 'main', len(main_rows), 'scenes (main)', len({r['group'] for r in main_rows}),
          'scenes (all)', len({r['group'] for r in rows}), 'skipped', dict(skipped))
    print('classes (main)', Counter(r['objects'][0]['name'] for r in main_rows))


if __name__ == '__main__':
    main()
