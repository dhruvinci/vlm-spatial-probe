# Objectron front-axis audit (E04, orientation)

The audit asks which column of Omni3D's `R_cam`, and which sign, points to each class's front. The pre-registration
(vlm-spatial-probe `experiments/E04-more-measures/PREREG.md`) sets the procedure:
- 20 test frames per class, each from a different video;
- a class is dropped unless one column and sign is consistent in at least 18 of the 20.

Drawn by `code/objectron_audit.py` (seed 2604); audited 2026-10-30.
- **Frames:** `objectron_audit/frames.json` lists the 60 frames with their video, file and annotation ids.
- **Grids:** `objectron_audit/<class>_<k>.jpg`, four frames each, numbered 1-20.
- **What each frame shows:**
  - the 3D box;
  - `R_cam`'s columns as arrows: +X red, +Y green, +Z blue (camera frame: x right, y down, z forward);
  - the +X face shaded red and the +Z face shaded blue.
- **Axis lengths:** Omni3D puts the box extents l, h, w along X, Y, Z, and `dimensions` is [w, h, l]. This was checked
  on 400 annotations: the corners rebuilt with that order match `bbox3D_cam` to under 1 mm (the 4 exceptions have w = l).
- **Close-ups:** unclear frames were redrawn at full size next to the unannotated image before deciding.

## Decisions
| class | decision | front | consistent | inconsistent or undeterminable |
|---|---|---|---|---|
| chair | **kept** | +X (column 0, sign +) | 18 / 20 | 1: front is -X (the box is turned 180 degrees); 19: round ottoman, no front |
| laptop | **dropped** | +X where visible | 15 / 20 (none contradicts) | 3: open, oblique view, box looks turned about 45 degrees to the base; 7, 9, 13, 18: closed lid, the front edge cannot be seen |
| bicycle | **kept**, pedal bicycles only (see below) | +X (column 0, sign +) | 20 / 20 | none |

**Front.** "Front" means:
- chair: the seat's front edge, away from the backrest;
- laptop: the touchpad edge, away from the hinge;
- bicycle: the front wheel and handlebars.

**Up.** In every chair and bicycle frame, +Y (green) points down along the vertical, so the box is gravity-aligned.
- The pre-registered `facing_deg` uses up = the camera's -y, because the annotation has no gravity field.
- `facing_deg_box_up` (exploratory, not pre-registered) uses up = -Y of the box.
- Across all chair, laptop and bicycle annotations, the box's Y axis is a median of 40 degrees from the camera's +y
  (5th to 95th percentile: 24 to 91 degrees). The two definitions differ most for steep downward views.

## Finding 2: Objectron's "bike" class is mostly motorised
Objectron's `bike` class (Omni3D calls it `bicycle`) also holds motorcycles and scooters. In the audit, 10 of the 20
frames were motorcycles or scooters; the 10 pedal bicycles among them were all consistent.

Calling those "the bicycle" in the prompt would be wrong. So every bicycle test video (91) was drawn once, unannotated,
in `objectron_audit/bicycle_videos_<k>.jpg` (list: `objectron_audit/bicycle_videos.json`), and labelled by eye:
- **39 pedal bicycles** (kept): sheet numbers 9-17, 19-29, 31, 38, 41, 43, 59, 61, 68, 70, 72, 79, 81-84, 86-88, 90,
  91. This includes children's bicycles with training wheels, one bike-share bicycle in its dock (22) and one
  fat-tyre e-bike with pedals (26).
- **47 motorcycles or scooters** (excluded).
- **5 unclear** (excluded): 5, 7, 36, 64, 85, all privacy-blurred beyond recognition.

The labels agree with the audit grids on all 19 audited videos that could be decided; audit frame 18 is "unclear" on
its sheet and motorised in the grid, so it is excluded either way. The video ids are in `objectron_audit.json`
(`videos_bicycle`, `videos_motorised`, `videos_unclear_blurred`).

## Consequences for the build (`code/objectron.py`)
- **Classes kept:** chair (354 test videos with a usable frame) and bicycle (39 videos). The laptop is dropped.
- **Size:** the 400 frames are split evenly but capped by supply: 39 bicycles and 354 chairs, so **393 frames**, not
  400.
- **Known risk:** the chair class includes stools and ottomans with no front. One of the 20 audited frames was an
  ottoman. They are not filtered out.
- **Option not taken:** keep laptops with open lids only. The box height would separate them (closed lids are a few
  centimetres tall), but that is a new rule and would need a fresh 20-frame audit of open laptops. It is left to the
  pre-registration owner.

## Frames checked (grid number: video, file)

**chair**

- 1: chair_batch_32_28, `objectron/test/chair_batch_32_28_0000000.jpg`
- 2: chair_batch_36_41, `objectron/test/chair_batch_36_41_0000000.jpg`
- 3: chair_batch_29_31, `objectron/test/chair_batch_29_31_0000200.jpg`
- 4: chair_batch_4_45, `objectron/test/chair_batch_4_45_0000000.jpg`
- 5: chair_batch_5_47, `objectron/test/chair_batch_5_47_0000200.jpg`
- 6: chair_batch_13_32, `objectron/test/chair_batch_13_32_0000200.jpg`
- 7: chair_batch_30_12, `objectron/test/chair_batch_30_12_0000100.jpg`
- 8: chair_batch_18_41, `objectron/test/chair_batch_18_41_0000100.jpg`
- 9: chair_batch_37_45, `objectron/test/chair_batch_37_45_0000100.jpg`
- 10: chair_batch_31_24, `objectron/test/chair_batch_31_24_0000100.jpg`
- 11: chair_batch_36_44, `objectron/test/chair_batch_36_44_0000000.jpg`
- 12: chair_batch_3_24, `objectron/test/chair_batch_3_24_0000100.jpg`
- 13: chair_batch_14_10, `objectron/test/chair_batch_14_10_0000200.jpg`
- 14: chair_batch_14_11, `objectron/test/chair_batch_14_11_0000000.jpg`
- 15: chair_batch_35_42, `objectron/test/chair_batch_35_42_0000000.jpg`
- 16: chair_batch_17_4, `objectron/test/chair_batch_17_4_0000000.jpg`
- 17: chair_batch_25_16, `objectron/test/chair_batch_25_16_0000100.jpg`
- 18: chair_batch_8_11, `objectron/test/chair_batch_8_11_0000000.jpg`
- 19: chair_batch_11_13, `objectron/test/chair_batch_11_13_0000000.jpg`
- 20: chair_batch_26_18, `objectron/test/chair_batch_26_18_0000100.jpg`

**laptop**

- 1: laptop_batch_24_42, `objectron/test/laptop_batch_24_42_0000100.jpg`
- 2: laptop_batch_21_36, `objectron/test/laptop_batch_21_36_0000000.jpg`
- 3: laptop_batch_21_29, `objectron/test/laptop_batch_21_29_0000000.jpg`
- 4: laptop_batch_30_0, `objectron/test/laptop_batch_30_0_0000100.jpg`
- 5: laptop_batch_38_41, `objectron/test/laptop_batch_38_41_0000000.jpg`
- 6: laptop_batch_17_37, `objectron/test/laptop_batch_17_37_0000100.jpg`
- 7: laptop_batch_26_42, `objectron/test/laptop_batch_26_42_0000300.jpg`
- 8: laptop_batch_4_47, `objectron/test/laptop_batch_4_47_0000000.jpg`
- 9: laptop_batch_13_35, `objectron/test/laptop_batch_13_35_0000300.jpg`
- 10: laptop_batch_17_25, `objectron/test/laptop_batch_17_25_0000300.jpg`
- 11: laptop_batch_25_0, `objectron/test/laptop_batch_25_0_0000000.jpg`
- 12: laptop_batch_7_8, `objectron/test/laptop_batch_7_8_0000000.jpg`
- 13: laptop_batch_35_30, `objectron/test/laptop_batch_35_30_0000200.jpg`
- 14: laptop_batch_14_46, `objectron/test/laptop_batch_14_46_0000300.jpg`
- 15: laptop_batch_24_13, `objectron/test/laptop_batch_24_13_0000200.jpg`
- 16: laptop_batch_27_25, `objectron/test/laptop_batch_27_25_0000000.jpg`
- 17: laptop_batch_8_20, `objectron/test/laptop_batch_8_20_0000100.jpg`
- 18: laptop_batch_25_9, `objectron/test/laptop_batch_25_9_0000000.jpg`
- 19: laptop_batch_17_31, `objectron/test/laptop_batch_17_31_0000300.jpg`
- 20: laptop_batch_12_8, `objectron/test/laptop_batch_12_8_0000100.jpg`

**bicycle**

- 1: bike_batch_12_33, `objectron/test/bike_batch_12_33_0000100.jpg`
- 2: bike_batch_6_35, `objectron/test/bike_batch_6_35_0000200.jpg`
- 3: bike_batch_9_44, `objectron/test/bike_batch_9_44_0000000.jpg`
- 4: bike_batch_1_29, `objectron/test/bike_batch_1_29_0000100.jpg`
- 5: bike_batch_0_36, `objectron/test/bike_batch_0_36_0000300.jpg`
- 6: bike_batch_1_25, `objectron/test/bike_batch_1_25_0000400.jpg`
- 7: bike_batch_9_9, `objectron/test/bike_batch_9_9_0000200.jpg`
- 8: bike_batch_2_27, `objectron/test/bike_batch_2_27_0000100.jpg`
- 9: bike_batch_6_48, `objectron/test/bike_batch_6_48_0000000.jpg`
- 10: bike_batch_10_9, `objectron/test/bike_batch_10_9_0000300.jpg`
- 11: bike_batch_6_16, `objectron/test/bike_batch_6_16_0000300.jpg`
- 12: bike_batch_11_18, `objectron/test/bike_batch_11_18_0000300.jpg`
- 13: bike_batch_10_21, `objectron/test/bike_batch_10_21_0000400.jpg`
- 14: bike_batch_11_7, `objectron/test/bike_batch_11_7_0000200.jpg`
- 15: bike_batch_1_6, `objectron/test/bike_batch_1_6_0000000.jpg`
- 16: bike_batch_7_23, `objectron/test/bike_batch_7_23_0000200.jpg`
- 17: bike_batch_11_2, `objectron/test/bike_batch_11_2_0000000.jpg`
- 18: bike_batch_0_5, `objectron/test/bike_batch_0_5_0000200.jpg`
- 19: bike_batch_2_18, `objectron/test/bike_batch_2_18_0000100.jpg`
- 20: bike_batch_3_30, `objectron/test/bike_batch_3_30_0000200.jpg`
