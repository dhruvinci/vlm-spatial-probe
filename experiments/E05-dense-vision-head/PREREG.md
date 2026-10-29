# E05 pre-registration: can a head on Qwen3.8's own vision tower measure depth and joints like the specialists?

Committed and pushed before any head is trained. Run locally. The frozen vision tower is loaded from the one checkpoint
shard that holds it (Qwen/Qwen3.8-27B @ 1d4bf0f2, shard 1 of 18).

**Question.** E01 and E02 show coarse, object-level geometry on the language side. Is fine geometry (per-pixel depth,
people's joints) available from the vision tower to a trained head, at the precision engagement measures need?

## Depth (NYU Depth v2, E01's split by scene)
- **Input.** Each image at 960x720 (patch 16, so 60x45 patches).
- **Head.** Features from blocks 6, 13, 20 and 26 (concatenated), then an MLP per patch (hidden 512): log depth on a
  2x2 sub-patch grid, upsampled bilinearly to the full image.
  - Trained on the training scenes (all labelled NYU images in them) with AdamW, learning rate 1e-3, 20 epochs, seed
    2604.
- **Scored** on the test scenes, per valid Kinect pixel: abs-rel (lower is better) and δ < 1.25 (higher is better).
- **Specialist:** MoGe-2 on the same images and pixels, metric, with no intrinsics.
- **Pass:** the head's abs-rel is within 1.5x MoGe-2's.

## Joints (COCO 2017 person keypoints)
- **Data.**
  - Train: 3,000 people from train2017 (seeded), each with at least 10 visible keypoints and a box of at least
    96 px.
  - Test: 600 people from val2017 by the same rule.
  - Crop: the true box padded by 25%, resized to 384x288 (24x18 patches).
- **Head.** Features from blocks 13, 20 and 26, then per patch: 17 joint logits, and for each joint an (x, y) offset
  inside the patch.
  - The prediction is the highest-scoring patch plus its offset.
  - AdamW, learning rate 1e-3, 30 epochs, seed 2604.
- **Scored** on visible keypoints: PCK, the share within 5% and within 10% of the box diagonal (higher is better).
- **Specialist:** ViTPose+ base (`usyd-community/vitpose-plus-base`) given the same box.
- **Pass:** the head's PCK@5% is no more than 10 points below ViTPose's.

## Reporting
- Both measures are reported with 95% bootstrap intervals over images (people).
- Failures are reported as they are. If the tower cannot be loaded from the one shard, that is reported, and E05 is not
  run.
