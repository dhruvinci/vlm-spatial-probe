# E07 pre-registration: why doesn't the decoder transfer? A diagnosis on the cached features

Committed and pushed before any fit is made. Nothing below is changed after the run; anything decided later is labelled
exploratory. E01's rules hold:
- public data only;
- every result is reported, including failures;
- no gate is rewritten.

Cost: $0. It runs on CPU on the E03 cache; no new GPU run and no new extraction.

## Question
E02 and E03 trained a decoder on the object word's hidden state and tested it on a scene type it never saw. On depth it
lost to the model's own spoken answer on every held-out scene type, for both models (E03: 1 of 26 folds passed, weakly,
and that one was 2D). The write-up lists likely causes, all untested. This experiment tests three of them on the same
data, folds and objects, changing one thing at a time:

| hypothesis | test (arm) |
|---|---|
| H1: the decoder learns scene type, not geometry | **A1, erase scene type.** Remove the training scene types' mean differences from the features, then fit as before |
| H2: what fails is absolute scale; the geometry inside an image transfers | **A2, within the image.** Fit and score depth relative to the other objects in the same image |
| H3: without the camera, metric depth is ambiguous | **A3, give the camera.** Predict the object's physical size; recover depth from its box and the camera's focal length |

**Not tested here:** a read-out with attention over all image tokens (hypothesis "one summary vector per object is too
narrow"). The cache holds only per-object summaries (the vision tower and image tokens mean-pooled inside the object's
box, and the object word). Testing it needs a new extraction that saves every image token, which is stage B, not
stage A. Also untested: "too little variety" (E03 tested it) and "the tower was trained for meaning" (needs training).

## Data, models, folds (as E03)
- E03's data: E01's calm frames plus E03's new images. Depth is measured in four scene types: indoor (NYU, 1,086
  objects), synthetic indoor (Hypersim, 1,183), two people (Harmony4D calm sequences, 1,134), street (KITTI, 1,277).
  Every image has at least two objects with depth.
- Models: Qwen3.8-27B (primary) and SpatialAxiom-9B.
- **Measure: depth only** (the headline measure). 3D centre, distance between objects and size are not re-tested.
- Folds: E03's four. Hold out one scene type, train on the other three.
- Read-out: the object word.
- Heads: ridge, and the same small MLP. The head is chosen by inner leave-one-scene-type-out error on the training
  scene types, as in E03.
- **Layer, primary analysis:** E03's chosen layer for each model and fold. Every arm then differs from the E03 decoder
  in one thing only.
- **Layer, exploratory:** each arm re-chooses its layer by E03's inner rule, over every 4th language layer plus E03's
  layer. Labelled exploratory.

## Arms
**R, reference.** E03's decoder, refitted at E03's layer by this experiment's code (`src/diagnose_e07.py`).
- Target: log depth. Score: abs-rel.
- Check: R must reproduce E03's published held-out depth error (`tables_e03.md`) to within 5% on each fold. A larger
  gap is reported; the comparisons below still use R, since it shares this experiment's code.

**A1, erase scene type.**
- On the rows the decoder trains on, take each training scene type's mean feature minus the overall mean (two
  directions for three scene types). Project them out of every row, training and held-out.
- After this, the training scene types have identical mean features, so no linear read-out can tell them apart (the
  guarantee of LEACE, Belrose et al. 2023, without its whitening).
- Inside the inner folds the projection is fitted on the inner training rows only.
- Target and score: as R.
- Diagnostic: how well a linear classifier names the scene type, before and after erasure. Three-way, 5 folds by
  image, on the training scene types, at the fold's layer.

**A2, within the image.**
- Subtract each image's mean from its objects' features and from their log depths, in training and test alike. The
  decoder then learns only how depth differs between objects in the same image. Every between-image and
  between-scene-type difference is removed.
- Score: **scale-invariant abs-rel.** Per image, the predicted depths are rescaled by the median of true over
  predicted, as in E03's relative-depth table.
- R, the spoken answer and MoGe-2 are scored the same way on the same images.
- Control, **2D cues:** ridge on the box alone (log box size in pixels, box centre x and y, log aspect ratio), with the
  same per-image subtraction.

**A3, give the camera.**
- Pinhole: depth = physical size / angular size. Angular size `s` is the square root of the box's width times its
  height in normalised camera coordinates: pixels divided by the focal length, after undistortion.
  - NYU and KITTI: the intrinsics in `dataset.json`.
  - Hypersim: each scene's projection matrix (`metadata_camera_parameters.csv`).
  - Harmony4D: each sequence's camera 1 fisheye intrinsics and distortion, undistorting the box's edge midpoints.
- Target: log(depth x s), the physical size the pinhole implies. Prediction: depth = exp(prediction) / s. Score:
  abs-rel on depth, as R.
- Control, **box + camera, no model:** the training mean of the same target per object name (the overall mean for a name
  not in training), turned into depth the same way.
- The box is the true box, the same one the vision tower and image-token features were pooled over. The spoken answer
  did not get a box, so A3 against the spoken answer is not like for like; the control measures how much comes from
  box and camera alone.

## Comparisons (per model and held-out scene type; paired 95% bootstrap over images, 2,000 resamples)
1. **Arm against R** on the same objects (A2 against R's scale-invariant error). An arm "helps" on a fold if the
   interval of R minus arm is above 0, and "hurts" if below.
2. **E03's pass bar** for each arm: beats the spoken answer (the interval of spoken minus arm above 0, n >= 30) and
   is within 1.5x MoGe-2 on the same objects. A2 uses the scale-invariant errors of the spoken answer and MoGe-2, on
   images where every object has a spoken answer (or a MoGe-2 depth).
3. A2 against its 2D-cues control, and A3 against its box + camera control: does the model's state add to the cue?

## Reading (fixed now)
Over the 8 folds (4 held-out scene types x 2 models):
- A hypothesis is **supported** if its arm helps in at least 4 folds and hurts in none.
- It is **not supported** if the arm helps in at most 1 fold.
- Anything else is **mixed**, reported fold by fold.
- An arm that passes E03's bar on a fold where R fails identifies a cause for that fold.
- An arm that helps but does not beat its control is credited to the cue (box, camera, 2D layout), not to the model's
  state.

## Known risks
- Erasing mean differences leaves non-linear scene-type information, which the MLP head can still use.
- Within-image depth has a much narrower range than absolute depth, and two-person images have only two objects.
- Harmony4D's fisheye is undistorted with its calibration; KITTI and NYU are treated as pinhole (no distortion
  correction).
- The fixed layers come from E03's selection on absolute depth. For Qwen3.8-27B on two people that layer is the first
  language layer, which transferred badly in E03. The exploratory layer re-choice addresses this.
