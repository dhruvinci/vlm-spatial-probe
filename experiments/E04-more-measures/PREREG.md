# E04 pre-registration: the measures E01 did not cover (orientation, surfaces and shape, motion, counts)

Committed and pushed before any data is built or any GPU is rented. Nothing below is changed after the run; anything
decided later is labelled exploratory. E01's rules hold:
- public data only;
- every result is reported, including failures;
- no gate is rewritten.

Approved by Dhruva on 2026-10-30, with $7 added: the paper 04 cap is now $17 in total, and **this rental's cap is
$9.34**.

## Question
E01 found that the object word holds metric depth but the spoken answer does not use it. It tested only depth, size,
3D position, distance and 2D position. Does the same hold for the other quantities a camera-based system measures?
- **orientation:** which way an object or a person faces, and joint angles;
- **surfaces and shape:** which way an object's main surface faces, and its dimensions;
- **motion from video:** speed and heading over four frames;
- **counts:** how many objects of a class.

## Data (built by `src/build_e04.py`, seed 2604; 5,600 image-equivalents; every licence re-checked and recorded in `docs/DATA.md`)
| measure | source (licence) | items | truth |
|---|---|---|---|
| orientation | KITTI object, training images not used in E01-E03 (CC BY-NC-SA 3.0) | 1,000 images, one object each (car, van, truck, pedestrian, cyclist; occlusion <= 1, truncation <= 0.3, box >= 25 px tall), 125 per 45-degree bin of the observation angle | facing angle relative to the camera, from `alpha`; dimensions h, w, l |
| orientation | Harmony4D calm sequences (hugging, sword, ballroom, karate), cameras 01/06/11/16, frames not in E01/E03 (MIT) | 600 frames, both people | body facing from the SMPL fit's global orientation, on the floor plane, relative to the camera; elbow and knee flexion from the fitted joints |
| orientation | Objectron test set (C-UDA 1.0) | 400 frames, one per video: chair, laptop, bicycle | facing from `R_cam` (the axis that marks the front, per class, after an audit of 20 frames per class; a class that fails the audit is dropped); dimensions |
| surfaces | Hypersim (CC BY-SA 3.0), at least 60 scenes, at most 10 frames each | 600 frames, one object each: table, desk, counter, bed, cabinet, shelves or similar, covering at least 2% of the image | main-surface normal in the camera frame (mean of `normal_cam` over the largest group of the object's pixels within 10 degrees of its most common normal); class up / towards the camera / sideways from `normal_world` and the view ray; dimensions from the object-aligned box extents (sorted) |
| surfaces | NYU Depth v2 labelled set, images not in E01 (no licence stated; research use) | 400 images, one object each | plane fitted to the raw depth inside the object's mask, kept only if at least 60% inliers and within 10 degrees of horizontal or vertical (gravity from a fitted floor plane); same class definition |
| motion | KITTI tracking, training sequences (CC BY-NC-SA 3.0) | 300 clips of 4 frames (t, t+3, t+6, t+9: 0.9 s), one object each: 150 cars, 100 pedestrians, 50 cyclists, spread over still / slow / fast | ground speed (m/s) with the car's own motion removed (`oxts` and calibration); heading in the first frame's camera frame; still if under 0.5 m/s |
| motion | Harmony4D calm sequences | 100 clips of 4 frames (index step 4; 20 frames per second, from the paper, as the release has no rate field), one person each | pelvis ground speed and heading |
| counts | COCO val2017, images not in E01 (annotations CC BY 4.0; images under their Flickr licences) | 1,000 images, one class each with no crowd region for that class: 250 each with 1-2, 3-5, 6-9 and 10-13 instances (a short band is filled from the next one; reported) | number of instances; secondary: instances at least 32 x 32 px |

**Changes from `docs/PLAN_E03.md`** (from a licence and access check run before writing this):
- Omni3D has its own licence (CC BY-NC 4.0); only Objectron is used from it, directly.
- TAPVid-3D's annotations are CC BY 4.0, but none of its three video sources allows redistribution of derived data, so
  it is dropped.
- Hypersim's object boxes do not mark a front, so they give dimensions only.
- KITTI tracking labels are in the moving car's frame, so ground speed removes the car's own motion.
- Identity across frames is not tested.
- Images are kept at most 640 px on the long side, as in E01. Splits are by group: KITTI object by image, Harmony4D by
  sequence, Objectron by video, Hypersim and NYU by scene, KITTI tracking by sequence, COCO by image. 60% train, 40%
  test.

## Models and what is run
- All five E01 models: Qwen3.8-27B, SpatialAxiom-9B, Qwen3.5-9B, SpaceQwen2.5-VL-3B, Qwen2.5-VL-3B. If the cap nears, the
  3B pair is cut first.
- Same prompt layout as E01: "Objects: the <name>." and the question.
  - For counts, the named object is the class ("the cars").
  - For motion, the four frames are given as four images in one message.
- **Features, every layer, as E01** (`src/extract.py`, extended):
  - the vision tower (patches inside the object's box, mean-pooled); for motion, the last frame's box; for counts, the
    whole image;
  - the image tokens inside the box at every language layer;
  - the object (or class) word's state;
  - the word with no image, and with a mismatched image from the same source.
  All saved in full on the private dataset.
- **Asked answers** (JSON, greedy, no thinking; 512 new tokens):
  - orientation: facing angle in degrees (0 = towards the camera, 90 = to the image's right, 180 = away, 270 = to its
    left), and for people the elbow and knee angles;
  - surfaces: up / towards the camera / sideways, and width, height and depth in metres;
  - motion: speed in m/s and direction (towards, away, left, right, still);
  - counts: a number.

## Probes and metrics (local, on the downloaded features)
- As E01: ridge per readout, layer and target on standardised features; alpha 10^0-10^5 by 5-fold cross-validation
  grouped as the split, on the training split.
- Layer families: V (best vision block), L_img (best image-token layer in the last quarter of the language model) and W
  (best word layer), each chosen on training only.
- Targets: facing as (sin, cos); joint angles in degrees; the normal as a unit 3-vector; log dimensions (sorted); speed
  in m/s; heading as (sin, cos) on moving objects; log(1 + count).

| measure | metric (lower is better unless stated) |
|---|---|
| facing | mean absolute angular error (degrees) |
| joint angles | mean absolute error (degrees) |
| surface normal | angle between the predicted and true normals (degrees); class: balanced accuracy (higher) |
| dimensions | abs-rel, mean over the three sorted dimensions |
| speed | mean absolute error (m/s) |
| heading | mean absolute angular error on moving objects (degrees) |
| count | mean absolute error; exact match (higher) |

- **Name-only prior P:** the training mean per object name or class (a circular mean for angles).
- **Specialists:**
  - MoGe-2's normals (surfaces), the median normal inside the object's mask;
  - RF-DETR (COCO) detections at score >= 0.5 for counts.
  - None for orientation or motion.
- **Controls:** shuffled labels (5 permutations); norm-matched random features; the word with no image; the word with a
  mismatched image.
- **Uncertainty:** 95% intervals by bootstrap over images or clips (2,000 resamples, paired). A rate on fewer than 30
  items enters no gate.

## Criteria (Qwen3.8-27B; the test split, all sources of a measure pooled; each source also reported)
Per measure, with E01's rules unchanged:

| # | gate | pass if |
|---|---|---|
| G0 | controls behave | on V, the shuffled-label and random-feature probes are no better than P |
| G1 | the tower carries the measure | V beats P by at least 20% relative, and the interval of P - V excludes 0 |

- **Recovery:** R_X = (P - X) / (P - V) for X = L_img and W.
- **Outcome per measure** (E01's hypothesis: measurement stays in the tower, the language side only names):
  - **falsified** if R >= 0.8 for L_img or W, with that readout beating P by at least 20% (interval excluding 0);
  - **supported** if R < 0.5 for both, with both upper bounds below 0.8;
  - **partial** otherwise.
  - If G1 fails, the premise fails for that measure, and that is the finding.
- **Attribution:** the no-image and mismatched-image word states must each keep less than 0.3 of the tower's gain;
  otherwise W's gain is not attributed to the image.
- **Ask (Q1):** the spoken answers are "qualitative" for a measure unless they beat P by at least 20% relative with the
  paired interval excluding 0. They are also compared with W and, where one exists, with the specialist. Unparseable
  answers count as missing; coverage is reported.
- **Spatial training (Q3):** SpatialAxiom-9B against Qwen3.5-9B on the asked measures, paired. A difference counts when
  its interval excludes 0.

**Primary endpoints:** for Qwen3.8-27B, Q1 and the outcome for **facing, surface normal, speed and count**. Joint
angles, dimensions, heading and the surface class are secondary. Harmony4D is calm sequences only: no MMA or
grappling anywhere in E04.

## Run, budget and disk
- One rental (`src/pod/launch.py`, as E01), with the watchdog set to stop at the cap.
- Expected about 3.9 GPU hours plus setup and transfer, about $8; **cap $9.34**.
- Every layer is kept, about 32 GB of features. Each model's features are downloaded, probed, uploaded to the private HF
  dataset, verified by sha256, and only then deleted locally.

## Known risks
- Linear probes; one wording; images at most 640 px.
- KITTI object images from the same drive can fall on both sides of the split (the public release has no drive ids).
- Facing truth for Objectron depends on the per-class audit.
- NYU planes depend on the quality gate.
- Harmony4D's frame rate is taken from the paper.
- COCO counts above about a dozen are capped by crowd regions; such pairs are excluded.
