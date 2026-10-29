# Data and model licences

**Images** are not redistributed. `src/build_dataset.py` rebuilds the dataset from the original sources. The private
dataset on Hugging Face holds only the extracted features, the models' answers and the labels.

## Data

| source | used for | licence |
|---|---|---|
| COCO val2017 (Lin et al., 2014) | people: 384 images | annotations CC BY 4.0; each image keeps its Flickr licence (recorded per image in `dataset.json`) |
| NYU Depth v2, labelled set (Silberman et al., ECCV 2012) | indoor: 400 images, Kinect depth | no licence stated by the authors; research use. Only figures with an NYU image are shown |
| Harmony4D (Khirodkar et al., NeurIPS 2024) | two_people: 216 frames, camera 1, multi-view body fits | MIT (Hugging Face dataset card `Jyun-Ting/Harmony4D`) |
| KITTI object detection, training set (Geiger et al., CVPR 2012) | street: 400 images, held out | CC BY-NC-SA 3.0. Never shown in a figure |
| COCO train2017 246285, "Zach MacPhee Dives Back into First" by Barry Stahl | Figure 1 only | CC BY 2.0 ([Flickr](https://www.flickr.com/photo.gne?id=5505334912)) |

Harmony4D frames by sequence type: sword 67, karate 51, hugging 50, MMA 23, grappling 19, ballroom 6. MMA and grappling
frames never appear in a figure. Every result is also reported without them.

E03 uses calm sequences only: E01's 174 calm frames plus 407 new frames from the same sequences, 581 in total
(sword 67 + 172 = 239, hugging 50 + 115 = 165, karate 51 + 110 = 161, ballroom 6 + 10 = 16). No MMA or grappling.

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
