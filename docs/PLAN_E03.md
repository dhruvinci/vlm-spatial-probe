# Plan (not run): E03, a cross-dataset decoder, and E04, the measures E01 did not cover

**Status: a plan only.** Nothing here has been run or bought. Each part runs only on Dhruva's direct go-ahead, with
its own pre-registration committed (and pushed) before any GPU time is rented. Every part has its own hard cap,
checked by the same watchdog as E01 (`src/pod/launch.py watch`).

## Timings this estimate rests on
E01, on one RTX PRO 6000 at $1.69/h; each model's `meta.json` on the dataset. Seconds per image:

| model | hidden-state passes (with image, no image, mismatched) | asked JSON + native box |
|---|---|---|
| Qwen3.8-27B | 0.35 | 1.74 |
| SpatialAxiom-9B | 0.14 | 0.44 |
| Qwen3.5-9B | 0.15 | 0.53 |
| SpaceQwen2.5-VL-3B | 0.13 | 0.31 |
| Qwen2.5-VL-3B | 0.13 | 0.30 |

Setup and model downloads took about 0.4 h. Feature downloads ran at about 10 MB/s.

## E03: a decoder trained across many datasets
**E02 did not pass** (2 of 50 folds, both weak; `experiments/E02-decoder-transfer/RESULTS.md`), so E03 is not
triggered by its own rule. Its failure mode, an absolute scale learned from one or two domains, is the thing a
many-domain decoder would test. Whether to run E03 anyway is Dhruva's call.
**Question.** A decoder on the object word's state, trained on many scene types, is tested on whole datasets it never
saw. Does it beat the spoken answer, and come within 1.5x the specialist (E02's bar), on most of them?

**Data:** depth, 3D centres, sizes and orientation, all with sensor or annotation truth.

| dataset | images | licence |
|---|---|---|
| NYU Depth v2, all 1,449 labelled | 1,449 | no licence stated; research use |
| Hypersim | 6,000 sampled frames | CC BY-SA 3.0, as stated in the ml-hypersim README; depth in metres and surface normals included |
| KITTI object | 3,000 | CC BY-NC-SA 3.0 |
| nuScenes | 2,000 front-camera frames | CC BY-NC-SA 4.0, as stated by its authors; check the current terms of use before download |
| Harmony4D, calm sequences only | 1,500 | MIT |
| Omni3D's indoor sources (SUN RGB-D, ARKitScenes) | 1,000 | Omni3D states no licence of its own, so each source's terms apply. **Check each before download; leave out any that forbid research redistribution of derived features** |

About 15,000 images in all. Whole datasets are held out in turn.

**Models:** Qwen3.8-27B (the probe subject) and SpatialAxiom-9B (the spatially fine-tuned model).

**Saved:** the with-image pass on every image. The no-image and mismatched-image passes, and the asked answers, on the
1,500 held-out test images only.

**Layers kept:** every 4th vision block and every 8th language layer (from the E01 and E02 layer choices), to keep the
cache under 15 GB. The local disk has 41 GB free and must keep 30 GB.

**Cost:**
- features, 15,000 images x (0.16 + 0.07) s: 0.94 h;
- held-out controls and answers, 1,500 x (0.35 + 1.74 + 0.14 + 0.44) s: 1.1 h;
- setup: 0.4 h.

Total about 2.5 h, so **about $4.2** at $1.69/h. **Cap $6.**

**Pass bar (pre-registered as in E02).** On each held-out dataset (n ≥ 30):
- the spoken-minus-decoder interval lies above 0;
- the decoder's error is at most 1.5x the specialist's (MoGe-2 for depth and 3D; RF-DETR for 2D).

"Transfers" means passing on at least 4 of the 6 held-out datasets. Each dataset is reported on its own.

## E04: the measures E01 did not cover (with ground truth)
| measure | data (licence) | truth | asked as |
|---|---|---|---|
| orientation and angles | KITTI object (CC BY-NC-SA 3.0): yaw and observation angle; Omni3D indoor sources (per-source terms, checked first): 3D rotation `R_cam`; Harmony4D (MIT): body facing and joint angles from the multi-view fits | angles in degrees | "Which way is the car facing, in degrees from the camera's axis?" |
| shape and surfaces | Hypersim (CC BY-SA 3.0): surface normals; NYU Depth v2: normals computed from the Kinect depth; KITTI and Omni3D: box dimensions and aspect | the normal of the object's main surface (angle error); dimensions | "Is the table top facing up, towards the camera, or sideways?", plus each dimension in metres |
| motion from video | KITTI tracking (CC BY-NC-SA 3.0): 3D tracks of cars and people; Harmony4D (MIT): people's 3D motion; TAPVid-3D (annotations Apache-2.0, but its videos come from Aria Digital Twin, Waymo Open and Panoptic Studio, each with its own terms; use only sources whose terms allow research use) | speed (m/s), heading, and identity across frames | 4 frames per clip: "How fast is the car moving, and in which direction?" |
| counts | COCO (annotations CC BY 4.0; images under their Flickr licences): instances per class | count | "How many cars are in this image?" |

**Probes.** As in E01, every layer, with the same controls. Motion and identity are probed on the object word's state
when the model sees four frames. Counts are probed on the class word's state ("the cars").

**Not used:**
- FSC-147: its repository states no licence.
- Kubric MOVi: the MOVi page states no licence for the rendered data. It is used only if a licence is found.

**Size:**
- orientation: 2,000 images;
- surfaces: 1,000 images;
- counts: 1,000 images;
- motion: 400 clips x 4 frames, about 1,600 image-equivalents.

About 5,600 image-equivalents in all, with 40% held out for testing.

**Cost, all five models:**
- with-image passes on everything: 5,600 x 0.43 s = 0.67 h;
- controls on the test set: 2,240 x 0.46 s = 0.28 h;
- asked answers on the test set: 2,240 x 3.3 s = 2.1 h;
- yes/no relation and count questions: about 0.4 h;
- setup: 0.4 h.

Total about 3.9 h, so **about $6.6**. **Cap $9.** If the cap nears, the 3B pair is cut first (it saves about 0.8 h); no
rate drops below n = 30.

## Order and approvals
1. E03 runs only on Dhruva's go-ahead (E02 did not pass, so it is not automatic). About $4.2; cap $6.
2. E04 runs on a separate go-ahead. About $6.6; cap $9.

The two are never in one rental.

Before any download, every dataset's licence is re-checked on its official page and recorded in `docs/DATA.md`. Any
dataset whose terms are unclear is dropped, not assumed.
