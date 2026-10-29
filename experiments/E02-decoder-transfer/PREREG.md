# E02 pre-registration: does a decoder on the hidden states transfer to a domain it never saw?

Written and committed (and pushed) before anything is fitted. Nothing below is changed after the run; anything decided
later is labelled exploratory. E01's rules hold:
- public data only;
- every result is reported, including failures;
- no gate is rewritten.

## Question
E01 found measurement (depth, size, position) in the language layers, but not in the spoken answer. A probe trained
and tested within the same domains says nothing about whether a read-out generalises.

Here a decoder is trained on some domains and tested on a domain it never saw. It is compared on the same objects with
the model's own spoken answer, with the name prior, and with a specialist vision model. If it transfers, attaching a
decoder (or fine-tuning towards one) is a sound route to measured answers. If it does not, the language side's
measurement is domain-bound.

## Data and models
- The E01 cache, with no new GPU run: 3,252 objects in 1,384 images.
- The same five models: Qwen3.8-27B (primary), SpatialAxiom-9B, Qwen3.5-9B, SpaceQwen2.5-VL-3B and Qwen2.5-VL-3B.
- E01's train/test split is not used. Each held-out domain is tested whole, and the decoder trains on every image of
  the other domains.

## Folds (leave one domain out; every fold reported on its own, never only pooled)
| measure | domains with truth | folds |
|---|---|---|
| depth, 3D centre, distance between two objects | indoor (NYU), two_people (Harmony4D), street (KITTI) | 3: hold out each one, train on the other two |
| size | indoor, street | 2: hold out each one, train on the other |
| 2D position | people (COCO), indoor, two_people, street | 4 |
| orientation (KITTI observation angle `alpha`) | street only | within street, 5 folds by image; **not a transfer test**, since no other domain has orientation truth |

**Orientation labels.** E01 did not save them. They are read from the original KITTI label files for exactly the same
objects (frame plus class, which is unique in each frame).

**Not measurable from the saved features:**
- Counts (COCO). The features are per named object, and every named object is unique in its image by construction.
- Body or facing angles (Harmony4D). No angle labels were saved.

Both are deferred to the step-2 plan, `docs/PLAN_E03.md`.

## Targets and metrics
| measure | target fitted | metric (better is) |
|---|---|---|
| depth | log depth (m) | abs-rel (lower). Also δ < 1.25 (higher), and a scale-invariant abs-rel: each image's predictions rescaled by the median of true over predicted, on images with at least 2 objects (lower) |
| 3D centre | camera-frame X, Y, Z (m) | relative 3D error, ‖Ĉ − C‖ / ‖C‖ (lower) |
| distance between two objects | from the two predicted 3D centres | abs-rel (lower) |
| size | log of the longest dimension (m) | abs-rel (lower) |
| 2D position | the centre as fractions of width and height | % of the image diagonal (lower) |
| orientation | sin α, cos α | mean absolute angular error in degrees (lower; a uniform guess gives 90) |

## Readouts, heads, layer choice
**Readouts** (as in E01):

| readout | what it is |
|---|---|
| V | vision blocks |
| L_img | image tokens in the last quarter of the language layers |
| L_img_any | image tokens in any language layer |
| W | the object word's state, with the image |

- **The primary decoder is W.** Like the spoken answer, it needs no box. V and L_img pool over the true box (as the
  specialist's depth crop does), so they are secondary. For 2D position only W is a fair read-out, because the box
  gives the position away.
- **Controls, at W's selected layer:**
  - the word with no image;
  - the word with a mismatched image;
  - shuffled training labels (5 permutations);
  - norm-matched random features;
  - the name prior (training-domain mean per object name; global training mean for unseen names).

**Layer choice and hyperparameters** use the training domains only, never the held-out domain:
- With two or more training domains: an inner leave-one-domain-out over them.
- With one: 5 folds by image.

The layer, the ridge alpha (10^0 to 10^5) and the head are all chosen on these inner folds.

**Heads:**
- **Ridge** on standardised features (as in E01).
- **One small MLP**, trained on the layer the ridge selected:
  - one hidden layer of 256, ReLU, dropout 0.1;
  - AdamW, learning rate 1e-3, weight decay 1e-2, 400 full-batch steps;
  - standardised inputs and targets; seed 2604.
- The head used for the pass test is whichever of the two has the lower inner-fold error.

## Comparisons, on the same held-out objects
- **The model's own spoken answer.** E01's JSON answers for depth, size, distance between two objects and 2D centre;
  for 2D, also the native box.
- **The name prior.**
- **The specialist:**
  - depth: MoGe-2, median over the central half of the box;
  - 3D centre: MoGe-2's median point;
  - distance between two objects: from MoGe-2's centres;
  - 2D position: RF-DETR Medium's most confident box of the object's COCO class. The class map is fixed here:
    - indoor: chair, bed, sofa→couch, table→dining table, television→tv, monitor→tv, microwave, sink, toilet, oven,
      laptop, plant→potted plant, vase, bottle, clock, keyboard, bowl;
    - street: car, truck, pedestrian→person, tram→train;
    - people and two_people as in E01;
    - objects of other classes have no specialist and are left out of the specialist comparison only.
  - Size has no specialist in the cache, so its bar is the spoken-answer half only.

## Pass bar (per model, measure and held-out fold; n ≥ 30 objects)
The decoder (W, chosen head) passes a fold if both hold:
1. it beats the spoken answer: the paired 95% bootstrap interval of (spoken − decoder) error lies above 0 (2,000
   resamples over images);
2. its error is at most 1.5 times the specialist's on the same objects (point estimates).

**Reporting:**
- The verdict per measure is the number of held-out folds passed, out of the folds that have both comparisons. It is
  never pooled.
- The paper-facing numbers exclude the MMA and grappling frames ("calm"). Both versions are reported.
- The same bar is reported for V and L_img as secondary.

## Known risks
- **Street extrapolation.** Street depths (5-120 m) lie far outside the indoor and two_people range (0.7-10 m). A
  linear read-out in log depth trained on the latter has to extrapolate.
- **Domain-unique names.** Most names occur in only one domain, so the name prior falls back to the global mean on the
  held-out domain.
- **Size has two domains only.**
- **Orientation is a within-domain probe, not a transfer test.**
