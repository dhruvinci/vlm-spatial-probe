# Data and model licences

**Images** are not redistributed. `src/build_dataset.py` rebuilds the dataset from the original sources. The private
dataset on Hugging Face holds only the extracted features, the models' answers and the labels.

## Data

| source | used for | licence |
|---|---|---|
| COCO val2017 (Lin et al., 2014) | people: 384 images | annotations CC BY 4.0; each image keeps its Flickr licence (recorded per image in `dataset.json`) |
| NYU Depth v2, labelled set (Silberman et al., ECCV 2012) | indoor: 400 images, Kinect depth | no licence stated by the authors; research use. Only figures with an NYU image are shown |
| Harmony4D (Khirodkar et al., NeurIPS 2024) | two_people: 200 frames, camera 1, multi-view body fits | MIT (Hugging Face dataset card `Jyun-Ting/Harmony4D`) |
| KITTI object detection, training set (Geiger et al., CVPR 2012) | street: 400 images, held out | CC BY-NC-SA 3.0. Never shown in a figure |
| COCO train2017 246285, "Zach MacPhee Dives Back into First" by Barry Stahl | Figure 1 only | CC BY 2.0 ([Flickr](https://www.flickr.com/photo.gne?id=5505334912)) |

Harmony4D frames by sequence type: sword 61, karate 51, hugging 42, MMA 21, grappling 19, ballroom 6 (200). MMA and
grappling frames never appear in a figure. Every result is also reported without them ("calm": 160 frames).

E03 uses calm sequences only: E01's 160 calm frames plus 407 new frames from the same sequences, 567 in total
(sword 61 + 172 = 233, hugging 42 + 115 = 157, karate 51 + 110 = 161, ballroom 6 + 10 = 16). No MMA or grappling.

**Correction (2026-10-29).** This file and E01's RESULTS gave 216 Harmony4D frames (sword 67, karate 51, hugging 50,
MMA 23, grappling 19, ballroom 6), and E03's calm set as 581. Those are the frames extracted before the build dropped
the 16 that fall between each sequence's training and test parts (`src/build_dataset.py`, `finish`). The dataset, and
every result, has 200 (E01) and 567 (E03). No result changes.

## Counts (from `dataset.json`)
E01 (`src/build_dataset.py`, `finish`): 1,384 scored images and 3,252 named objects, plus the Figure 1 photo (4 objects,
not scored). The models were asked about all 1,385 images.

| domain | images (train / test) | objects (train / test) |
|---|---|---|
| people (COCO val2017) | 384 (269 / 115) | 910 (635 / 275) |
| indoor (NYU Depth v2, split by scene) | 400 (276 / 124) | 1,086 (748 / 338) |
| two_people (Harmony4D) | 200 (129 / 71); calm 160 | 400 (258 / 142); calm 320 |
| street (KITTI) | 400, held out | 856 |
| total scored | 1,384; calm 1,344 | 3,252; calm 3,172 |

E03 adds (`finish_e03`), calm only:

| domain | new images | new objects | E03 total with E01's calm images |
|---|---|---|---|
| indoor_synth (Hypersim) | 465 | 1,183 | 465 images, 1,183 objects |
| two_people (Harmony4D) | 407 | 814 | 567 frames, 1,134 people |
| street (KITTI) | 201 | 421 | 601 images, 1,277 objects |
| total | 1,073 | 2,418 | |

E03's indoor (NYU) set is E01's 400 images (1,086 objects); people (COCO) is E01's 384, used for 2D position only.
E07 uses E03's depth sets.

## E04 sources (licences checked on 2026-10-30 on each official page)
| source | used for | licence / terms |
|---|---|---|
| KITTI object, training images not used in E01-E03 | orientation, dimensions | CC BY-NC-SA 3.0 ("All datasets and benchmarks on this page are copyright by us and published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License") |
| KITTI tracking, training sequences (`oxts` for the car's own motion) | motion | CC BY-NC-SA 3.0 (same statement) |
| Harmony4D, calm sequences (hugging, sword, ballroom, karate) | orientation, joint angles, motion | MIT (Hugging Face dataset card `Jyun-Ting/Harmony4D`); frame rate 20 per second, from the paper |
| Objectron test set: chairs, and pedal bicycles only (most of its "bike" class are motorcycles and scooters, excluded by eye) | orientation, dimensions | C-UDA 1.0 ("The C-UDA does not impose any restriction with respect to the use, modification, or distribution of Results") |
| Hypersim | surfaces, dimensions | CC BY-SA 3.0 |
| NYU Depth v2, labelled set, images not in E01 | surfaces | no licence stated by the authors; research use |
| COCO val2017, images not in E01 | counts | annotations CC BY 4.0; images under their Flickr licences |

E04 as built (`src/build_e04.py finish`): 4,379 images or clips and 4,979 objects, split by group (60/40 by group;
test in brackets).

| source | measure | items (test) | objects (test) |
|---|---|---|---|
| KITTI object | orientation | 1,000 (400) | 1,000 (400) |
| Harmony4D calm sequences | orientation | 600 (349) | 1,200 (698) |
| Objectron chairs and pedal bicycles | orientation | 393 (157) | 393 (157) |
| Hypersim, 73 scenes | surfaces | 600 (240) | 600 (240) |
| NYU Depth v2 | surfaces | 386 (156) | 386 (156) |
| KITTI tracking | motion | 300 (123) | 300 (123) |
| Harmony4D calm sequences | motion | 100 (48) | 100 (48) |
| COCO val2017 | counts | 1,000 (400) | 1,000 (400) |

Fewer NYU and Objectron items than pre-registered, and why: [E04 RESULTS](../experiments/E04-more-measures/RESULTS.md).

Not used: Omni3D's own annotations (CC BY-NC 4.0; Objectron is read directly), SUN RGB-D (no licence stated),
nuScenes (terms not readable without an account), ARKitScenes (Apple non-commercial licence; not needed), and
TAPVid-3D (annotations CC BY 4.0, but none of its video sources allows redistribution of derived data). Images are not
redistributed; `src/build_e04.py` and `src/e04_parts/` rebuild them.

## Models

| model | revision | licence |
|---|---|---|
| `Qwen/Qwen3.8-27B` | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 | Apache-2.0 |
| `SpatialAxiom/SpatialAxiom-9B` | bfb2cc3508976c8a77f753a3c7ef0ca4e44e27d4 | CC BY-NC 4.0 |
| `Qwen/Qwen3.5-9B` | c202236235762e1c871ad0ccb60c8ee5ba337b9a | Apache-2.0 |
| `remyxai/SpaceQwen2.5-VL-3B-Instruct` | 332ff3fbc40d0e62af64b1bd2e0a606b3cca028e | card header `qwen-research`; README text says Apache-2.0 (both recorded) |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 66285546d2b821cf421d4f5eb2576359d3770cd3 | `qwen-research` |
| MoGe-2 (`Ruicheng/moge-2-vitl-normal`) | CV reference | MIT |
| RF-DETR Medium | CV reference | Apache-2.0 |

Official SpatialVLM weights are not public. SpaceQwen2.5-VL-3B is an open reproduction of its data recipe.
