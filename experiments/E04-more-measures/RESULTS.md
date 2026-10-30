# E04 results: the measures E01 did not cover

**Status.**
- Pre-registered ([PREREG.md](PREREG.md)): committed and pushed as 40feb44 before any data was built or GPU rented. The
  code was pushed before the rental (6e12662).
- One rental, 2026-10-30: $5.39 against a $9.34 cap ([docs/SPEND.md](../../docs/SPEND.md)). Probes, specialists and
  scoring ran locally.
- Every number below is printed by `src/analyse_e04.py` into [data/tables_e04.md](data/tables_e04.md) and cited as
  `tables_e04.md:line`. Test split only. Errors: lower is better. Intervals: 95% bootstrap over images or clips (2,000
  resamples), paired.
- Harmony4D is calm sequences only (hugging, sword, ballroom, karate). No MMA or grappling appears anywhere in E04.

## Data as built
| measure | source | items (test) | objects (test) |
|---|---|---|---|
| orientation | KITTI object, images not in E01-E03 | 1,000 (400) | 1,000 (400) |
| orientation | Harmony4D, calm sequences, both people | 600 (349) | 1,200 (698) |
| orientation | Objectron: chairs and pedal bicycles | 393 (157) | 393 (157) |
| surfaces | Hypersim, 73 scenes | 600 (240) | 600 (240) |
| surfaces | NYU Depth v2, images not in E01 | 386 (156) | 386 (156) |
| motion | KITTI tracking, 4 frames over 0.9 s | 300 (123) | 300 (123) |
| motion | Harmony4D, 4 frames | 100 (48) | 100 (48) |
| counts | COCO val2017, images not in E01 | 1,000 (400) | 1,000 (400) |
| total | | 4,379 | 4,979 |

Specialists, on the same test items: MoGe-2's normal (median inside the object's mask) is off by 9.3° [7.7, 10.9]
(`tables_e04.md:11`). RF-DETR's count is off by 2.07 [1.82, 2.34] (`tables_e04.md:16`).

## Deviations from the pre-registration
Decided before any model output was seen:
1. **Objectron: 393 frames, not 400** (354 chairs, 39 pedal bicycles).
   - Laptops failed the front-axis audit: 15 of 20 frames had a consistent front. They were dropped as pre-registered
     ([OBJECTRON_AUDIT.md](OBJECTRON_AUDIT.md)).
   - Most of Objectron's "bike" class are motorcycles and scooters, which were excluded by eye. That leaves 39 videos.
2. **NYU: 386 images, not 400.** After E01's 400 images are excluded, these are all the images that pass the gates:
   - 158 have too little floor to fit gravity;
   - 505 have no object that passes the plane test.
3. **Harmony4D motion: 58 of 100 clips span 1.2 s instead of 0.6 s.** Those sequences' released frames are every 8th
   raw frame. The truth speed uses each clip's real span, and the prompt states each clip's real frame spacing.
4. **KITTI tracking** has 117 still, 117 slow and 66 fast clips. There are no fast pedestrians.
5. **KITTI object:** one of the eight observation-angle bins was 11 short. It was filled from the largest bins, as
   pre-registered.
6. **Harmony4D orientation split.** The split is by sequence, and with few sequences it came out at 251 training and
   349 test frames rather than 60/40 by frames.
7. **COCO selection bug, fixed before the run (c1dfbe2).** The first build let one image be picked twice for
   different classes within a count band (45 of 1,000 rows). The pre-registration says one class per image, so COCO
   was rebuilt before any model saw it.
8. **MoGe-2 reference, fixed before scoring (ecbbe8d).** The code first took the median normal over the central half
   of the box. It now takes it inside the object's mask, as pre-registered. The masks are rebuilt exactly as each
   builder chose the object, and checked against the stored box and pixel count. Mask against box:
   - NYU, test: mean error 6.8° against 15.6°;
   - Hypersim, test: 10.9° against 19.6°.
9. **Storage.** Every layer of every model was extracted and kept. Qwen3.8-27B's and SpatialAxiom-9B's features are on
   the private Hugging Face dataset, verified by sha256. The other three models' features hit the account's private
   storage limit, so they are kept locally until there is room.

Changed after seeing Qwen3.8-27B's tables (disclosed; no pre-registered verdict changes):
- **The shuffled-label control for the surface class.**
  - The bug: it averaged each item's 0/1 errors over the five permutations, then counted an item correct only if every
    permutation got it right. That overstated the control's error (0.885).
  - The fix: each permutation is scored with balanced accuracy, then averaged (0.668, `tables_e04.md:12`). G0 passes
    either way.
- **The spoken surface class** was first reported as a plain error rate. It is now 1 − balanced accuracy, the
  pre-registered metric, against the prior on the same items.
- **Added as exploratory, labelled so in the tables:**
  - the spoken motion direction as a 5-way class;
  - counts that ignore the asked JSON form, parsed leniently;
  - spoken joint angles read as flexion;
  - the spoken answer against the specialist, paired. The pre-registration asks for this comparison but the first
    scorer omitted it.

## Primary endpoints (Qwen3.8-27B)
| measure | prior P | tower V | late image tokens | object word W | spoken | outcome | spoken answer |
|---|---|---|---|---|---|---|---|
| facing (°) | 57.4 | 43.2 | 37.5 | **35.6** | 52.4 | **falsified** (R_W 1.53 [1.31, 1.88]) | qualitative |
| surface normal (°) | 40.1 | 31.6 | 33.2 | 32.0 | not asked | **falsified** (R_W 0.95 [0.76, 1.23]) | n/a |
| speed (m/s) | 2.06 | 2.33 | 1.74 | 2.06 | 1.78 | premise fails (G1) | qualitative |
| count | 2.91 | 2.76 | 2.92 | **1.43** | 1.88 | premise fails (G1) | **a measurement** |

(`tables_e04.md:9, 11, 14, 16`; R = (P − X) / (P − V), the share of the tower's gain over the prior that readout X
keeps.)

- **Facing.**
  - The tower beats the prior by 24.9%.
  - The object word beats it by 38.1% and the late image tokens by 34.7%, so both keep more than the tower's whole
    gain. That falsifies the hypothesis.
  - The gain comes from the image. With no image the word scores 57.5°, and with a mismatched image 57.5°, the same as
    the prior's 57.4°.
  - Spoken, the model is 5.0° [1.4, 8.8] better than the prior. That is 8.7%, below the 20% bar, so qualitative.
- **Surface normal.**
  - The tower beats the prior by 21.2%.
  - The word keeps 0.95 of that gain and beats the prior by 20.1% (interval of the gain 6.0-10.2°). That is just over
    the 20% bar, so the result is **falsified, narrowly**.
  - The late image tokens keep 0.81 but beat the prior by only 17.1%.
  - No readout comes near MoGe-2: 9.3° against 32.0° for the word.
  - The model was asked for the surface's class, not its normal (below).
- **Speed.** The tower's read-out is the last frame's patches, so it cannot see motion, and it is worse than the prior
  (2.33 against 2.06).
  - The premise fails by construction.
  - The late image tokens, which can attend to earlier frames, beat the prior by 15.5% (P − X: 0.12-0.52).
  - Spoken speed, 1.78, is 13.6% better than the prior, with an interval that includes no difference.
- **Count.** The tower's read-out averages the patches over the whole image, and it beats the prior by only 5%, so the
  premise fails.
  - The class word beats the prior by 50.8% (1.43 against 2.91). Its no-image and mismatched-image versions gain 2.9%
    and 1.8%.
  - The pre-registered attribution rule (keep less than 0.3 of the tower's gain) reads "no" only because the tower's
    gain is so small.
  - **The spoken count is a measurement:** 1.88, 35% better than the prior, P − spoken 1.03 [0.67, 1.36].
  - The spoken count is level with RF-DETR: difference −0.19 [−0.44, 0.09] (`tables_e04.md:19`).
  - The class word's state, 1.43 [1.29, 1.58], is better than both.

## Secondary measures (Qwen3.8-27B)
- **Joint angles:** controls failed (G0); not interpreted.
  - The shuffled-label probe is better than the prior by 0.07° [0.03, 0.10], a consistent but tiny margin
    (`tables_e04.md:10`).
  - By E01's rule, no joint-angle probe result is read for this model.
- **Surface class (1 − balanced accuracy):** **partial**.
  - The tower is at 0.294 and the word at 0.363, keeping 0.48 of the tower's gain.
  - Spoken, 0.436 is no better than the prior's 0.427 (`tables_e04.md:12`).
- **Dimensions and heading:** premise fails.
  - For dimensions the tower is worse than the prior (0.247 against 0.212); the word is slightly better than the prior
    (0.192).
  - For heading, 76 moving objects, the tower is 15.5% better than the prior, with an interval that reaches no
    difference (`tables_e04.md:13, 15`).

## By source (Qwen3.8-27B; `tables_e04.md:27-40`)
- **Facing splits by kind of object.**
  - On people (Harmony4D) the word reads facing far better than the tower: 28.4° against 44.0° (prior 48.0°).
  - On KITTI objects the tower is better: 38.4° against 43.2° (prior 75.7°).
  - On Objectron chairs and bicycles little is read: 47.7°-51.5° against a prior of 53.2°.
  - The same split holds in all five models (`tables_e04.md:64-66, 101-103, 138-140, 175-177`).
- **E02's 10° is a different measurement.** E02 found the word reads KITTI orientation to about 10°. That was an MLP
  decoder trained within KITTI on E01's objects, mostly cars at their natural angles. Here the probe is E01's linear
  ridge, the angles are balanced over eight 45° bins, KITTI has five classes, and the probe is fitted on three sources
  together. The two numbers are not comparable.
- **Surface normals:** NYU 27.2° for the word against 28.5° for the tower; Hypersim 35.2° against 33.6°.

## All five models
Pre-registered outcome per measure (F = falsified, S = supported, P = partial, G1 = premise fails, G0 = controls
failed):

| measure | Qwen3.8-27B | SpatialAxiom-9B | Qwen3.5-9B | SpaceQwen2.5-VL-3B | Qwen2.5-VL-3B |
|---|---|---|---|---|---|
| facing | F | F | F | F | F |
| joint angles | G0 | P | P | P | P |
| surface normal | F | F | G1 | G1 | G1 |
| surface class | P | S | P | S | S |
| dimensions | G1 | G1 | G1 | G1 | G1 |
| speed | G1 | G1 | G1 | G1 | G1 |
| heading | G1 | G1 | G1 | G1 | G1 |
| count | G1 | G1 | G1 | G1 | G1 |

(`tables_e04.md:9-16, 46-53, 83-90, 120-127, 157-164`.)
- **Facing is falsified in every model.**
- **The surface normal is falsified in the two larger Qwen3-family models.** Elsewhere the tower misses the 20% bar
  (Qwen3.5-9B by 0.1 point: 19.9%).
- **The surface class stays in the tower more than anything else:**
  - supported in three models;
  - partial in two, with the word keeping 0.48-0.55 of the tower's gain.
- **Every model's class word counts.** It beats the prior by 47-51% (1.43-1.53), while the whole-image tower gains
  about 5%.
- **The spoken answers:**

| measure | Qwen3.8-27B | SpatialAxiom-9B | Qwen3.5-9B | SpaceQwen2.5-VL-3B | Qwen2.5-VL-3B |
|---|---|---|---|---|---|
| facing (°; prior 57.4) | 52.4 | 72.6 | 81.4 | 88.2 | 94.6 |
| joint angles (°; prior 23.9) | 22.0 | 104.0 | 30.2 | 56.2 | 60.2 |
| surface class (prior 0.43) | 0.44 | 0.48 | 0.47 | 0.72 | 0.64 |
| speed (m/s; prior 2.06) | 1.78 | 1.41 | 1.65 | 1.80 | 5.62 |
| count (prior 2.91) | **1.88** | **2.06** | 2.10 (65% parsed) | **2.20** | **2.31** |

(Bold: a measurement by the pre-registered rule; every other entry is qualitative. Coverage and intervals:
`tables_e04.md`, each model's "spoken" column.)

- **Only counts are spoken as measurements.** They are in four of five models. Qwen3.5-9B answered a third of the count
  questions in other JSON forms, so by the pre-registered rule they are missing. Parsed leniently, 93% are usable, at
  1.79 (exploratory; `tables_e04.md:94`).
- **Facing is spoken worse than the name prior by every model except Qwen3.8-27B.**
- **SpatialAxiom-9B gives joint angles near 10-15° for straight limbs.** The prompt says 180 means straight. Read as
  flexion, its error is 35.0° (exploratory; `tables_e04.md:58`), still worse than the prior.

## Q3: spatial training (SpatialAxiom-9B against its base Qwen3.5-9B, spoken, paired)
- Facing: better by 9.9° [6.3, 13.9] (72.1 against 82.0). Both are worse than the prior.
- Joint angles: worse by 73.8° [68.7, 79.1] (the convention above).
- Surface class, dimensions, speed and count: no difference (`tables_e04.md:194-199`).
- E01 found the spatial fine-tune made spoken depth a measurement. Here it improves spoken facing a little, without
  making it a measurement.

## Exploratory (not pre-registered)
- **Spoken motion direction** (towards, away, left, right, still; 171 clips).
  - The baseline: always saying "still", the training split's most common class, is right 52-56% of the time.
  - Qwen3.5-9B reaches 0.573, level with that baseline.
  - Qwen3.8-27B reaches 0.409. It often says "towards" for still objects.
  - SpatialAxiom-9B reaches 0.342, and the 3B models 0.13-0.20 (`tables_e04.md:18, 55, 92, 129, 166`).
- **Lenient count parsing** changes only Qwen3.5-9B: from 2.10 at 65% coverage to 1.79 at 93% (`tables_e04.md:94`).

## What this means
- **E01's finding extends from depth to orientation and surface shape.** Where the vision tower carries a measure, the
  object word carries it as well, and it comes from the image. For people's facing, the word carries it better than
  the tower.
- **The spoken answer still does not read it out.** Facing and the surface class are spoken at or below the name prior,
  while the hidden states hold them.
- **Counting is the one measure spoken as a measurement.** Even there, a linear read-out of the class word beats both
  the spoken count and RF-DETR on these images.
- **Speed, heading and dimensions are not in the tower at the pre-registered bar.** For speed, the tower's read-out
  sees one frame. For dimensions, the tower is worse than knowing the object's class. The word adds a little over the
  class (exploratory reading of `tables_e04.md:13`).

## Known limits
- **Linear probes, one wording, images at most 640 px.**
- **Read-out choices that cap G1:**
  - the tower read-out for motion is the last frame's patches;
  - for counts it is the mean over the whole image.
  These choices make G1 fail for speed and counts by design. That is a limit of the read-out, not evidence that the
  tower lacks the measure.
- **KITTI object images from the same drive can fall on both sides of the split.** The public release has no drive ids.
- **Small sets:**
  - heading has 76 moving test objects, 18 of them people;
  - Harmony4D motion has 48 test clips.
- **The spoken surface answer is a class.** The model was not asked for a normal, so for the normal there is no
  spoken comparison.

## Reproduce
```
# data (local), then the GPU run (pod), then everything else locally
ENGAGE3D_SRC=.../engage3d/src python src/build_e04.py kitti_obj --src SRC --out BUILT_E04 --cache CACHE --cache-e03 CACHE_E03
#   ... and h4d, h4d_motion, coco; built_e04/code/{kitti_trk,objectron,hypersim}.py; on the pod: nyu,
#   code/hypersim_truth.py, finish (src/pod/run_e04.sh)
python src/pod/launch.py create|upload|start --script run_e04.sh|watch --stop-at 9.34
python src/e04_parts/masks.py nyu --mat nyu_depth_v2_labeled.mat --data DATA_E04 --out DATA_E04/masks_nyu.npz   # needs h5py
python src/e04_parts/masks.py hypersim --data DATA_E04 --out DATA_E04/masks_hypersim.npz
python src/cv_e04.py normals --data DATA_E04 --out CACHE_E04/cv_e04_normals.json
python src/cv_e04.py counts --data DATA_E04 --out CACHE_E04/cv_e04_counts.json --weights rf-detr-medium.pth
python src/probe_e04.py --data DATA_E04 --feats CACHE_E04/e04_<model> --out CACHE_E04/probes     # each model
python src/analyse_e04.py --data DATA_E04 --probes CACHE_E04/probes --answers CACHE_E04 --cv CACHE_E04 \
    --out experiments/E04-more-measures/data/results_e04.json
python src/make_figures_e04.py --results experiments/E04-more-measures/data/results_e04.json --out docs/assets
```
