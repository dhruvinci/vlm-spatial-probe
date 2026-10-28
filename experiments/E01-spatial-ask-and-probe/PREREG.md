# E01 pre-registration: ask the model, then probe it, layer by layer

Written and committed before any GPU run. Nothing below is changed after the run; anything decided later is labelled
exploratory.

## Hypothesis and falsifier
**Hypothesis (Dhruva's).** Measurable spatial information (where an object is in the image, how far away it is, how
big it is) lives in the vision tower. The language layers tie it to words, not to measurable positions, so the
model's text answers are qualitative and no decoder on the language side recovers measurements.

**Falsifier.** The hypothesis fails if the image tokens in late language layers, or the object word's state, recover
depth or position close to the vision tower's level and clearly above the name-only prior (criteria below). The
finding would then be "present but never read out", which supports adding a decoder or fine-tuning. Either outcome is
reported.

## Data (built by `src/build_dataset.py`, seed 2604; all public)
| domain | source | truth | split |
|---|---|---|---|
| people | COCO val2017: one person plus 1-2 objects of a class that occurs once | 2D boxes | 70/30 by image |
| indoor | NYU Depth v2, labelled set | Kinect depth, instance masks: 2D, depth, 3D centre, size | 70/30 by scene |
| two_people | Harmony4D multi-person capture, camera 1: mostly hugging, ballroom, sword and karate, with a minority of MMA and grappling; the two people are named "the person on the left / right" | multi-view body fits: 2D, depth (pelvis), 3D centre | by time in each sequence (first 65% train, last 25% test, 10% gap); grappling 70/30 by sequence |
| street | KITTI object training set | LiDAR-aided 3D boxes: 2D, depth, 3D centre, size | entirely held out |

A named object is unambiguous (its class occurs once in the image). Relations are defined only with clear margins:
left/right and above/below need centres at least 10% of the width or height apart; closer/farther needs depths at least
15% apart; bigger/smaller needs sizes at least 1.5x apart. Figure 1's baseball photo (COCO train2017 246285) is run
through every model but not scored.

## Models (in run order; the cap is $10 in total on one rented GPU)
1. Qwen3.8-27B (`Qwen/Qwen3.8-27B` @ 1d4bf0f2), the probe subject.
2. SpatialAxiom-9B (`SpatialAxiom/SpatialAxiom-9B` @ bfb2cc35), a full fine-tune of Qwen3.5-9B on spatial data, and
   its base Qwen3.5-9B (`Qwen/Qwen3.5-9B` @ c2022362).
3. SpaceQwen2.5-VL-3B (`remyxai/SpaceQwen2.5-VL-3B-Instruct` @ 332ff3fb), an open reproduction of the SpatialVLM data
   recipe, and its base Qwen2.5-VL-3B-Instruct (@ 66285546).

The official SpatialVLM weights are not public. If spend forces a cut, it is made from the end of this list and
recorded as not run. No rate is shrunk below n = 30 to fit.

## What is run (`src/extract.py`)
Greedy decoding with no thinking, bf16. The image comes first, then `Objects: the <a>; the <b>; ...` and the question.
- **Ask.** JSON answers: each object's centre (x, y in percent), distance from the camera (m) and size (m), and each
  pair's distance (m). Also the model's native boxes (`bbox_2d`).
- **Relations and naming.** Yes/no questions scored by the next token, P(yes) against P(no):
  - closer, left of, above, bigger. The a/b order is flipped at random, so the truth is balanced. left/right is not
    asked for two_people, because the names give it away.
  - "Is there a <name> in this image?" with one present and one absent name per image.
  - Test and held-out images only; two_people uses all images, to reach n. Street is capped at 400 per relation and
    200 images for naming.
- **Hidden states.** One readout family per row below:

| readout | what is saved |
|---|---|
| vision tower, every block | patches inside the object's box, mean-pooled, image order restored (E00) |
| image tokens, every language layer | merged tokens inside the box, mean-pooled (layer 0 = the projected tower output) |
| object word, every language layer | the state at the last token of "the <name>", with the image |
| object word, no image | the same text with no image |
| object word, mismatched image | the same text with another image of the same domain |

## Probes (`src/probe.py`, local)
- **Probe.** One ridge regression per readout, layer and target, on features standardised with training statistics.
  Alpha is chosen from 10^0 to 10^5 by 5-fold cross-validation on the training split, grouped by image.
- **Targets and training domains:**

  | target | trained on |
  |---|---|
  | x, y (fractions) | people, indoor, two_people |
  | log depth; 3D centre | indoor, two_people |
  | log size | indoor |

- **Derived quantities.** Pair distance is the distance between the two predicted 3D centres. Relations come from the
  per-object predictions.
- **Layer choice.** Each readout family's layer is chosen by cross-validated score on the training split, never on test:
  - V = the best vision block.
  - L_img = the best image-token layer in the last quarter of the language model (layers 48-64 of 64 for Qwen3.8-27B).
  - W = the best word-state layer (1 to the last).

  Every layer's test score is also plotted.
- **Metrics:**

  | quantity | metric | better is |
  |---|---|---|
  | 2D centre | error as % of the image diagonal | lower |
  | depth | abs-rel | lower |
  | depth | δ < 1.25 | higher |
  | size, pair distance | abs-rel | lower |
  | relations | balanced accuracy (chance 50%) | higher |
  | naming | accuracy (chance 50%) | higher |

- **Baselines:**
  - chance;
  - **name-only prior P**: the training mean for that object name (log space for depth and size), or the global mean
    for a name unseen in training;
  - box geometry alone (box corners and log area, i.e. which patches were pooled);
  - CV references: MoGe-2 metric depth (median over the box's central half) and RF-DETR boxes for 2D, on the same
    objects.
- **Controls, on each selected layer:**
  - training labels shuffled (5 permutations);
  - norm-matched random features;
  - the word state with no image;
  - the word state with a mismatched image.
- **Uncertainty.** 95% intervals by bootstrap over images (2,000 resamples, paired for differences). Any rate with fewer
  than 30 items is reported but enters no gate.

## Criteria (Qwen3.8-27B; test split of the training domains; depth is primary)
Controls come first.

| # | gate | pass if |
|---|---|---|
| G0 | controls behave | on V, the shuffled-label and random-feature probes are no better than P (the 95% interval of P − probe abs-rel includes 0 or lies below it) |
| G1 | the tower carries depth | V beats P by at least 20% relative, and the interval of P − V excludes 0 |

If G0 fails, no probe result is interpreted. If G1 fails, the hypothesis's premise fails, and that is the finding.

**Recovery.** R_X = (P − X) / (P − V), in abs-rel, for X = L_img and X = W. It is the share of the tower's gain over the
name prior that the language side keeps. Its interval comes from the image bootstrap.

**Outcome:**
- **Falsified:** R ≥ 0.8 for L_img or W, and that readout beats P by at least 20% relative with the interval of P − X
  excluding 0.
- **Supported:** R < 0.5 for both, and both upper 95% bounds are below 0.8.
- **Partial:** anything else.

**Attribution check for W.** The word state's gain must come from the image. The no-image word state and the
mismatched-image word state should each keep less than 0.3 of the tower's gain. If either keeps 0.3 or more, W's gain is
not attributed to the image.

**Same rule, secondary:**
- 2D position (W only; L_img's position is fixed by the pooling, so it is shown but not gated);
- size (indoor only);
- pair distance.

**Ask (Q1).**
- The text depth answers are "qualitative" (not a usable measurement) unless they beat P by at least 20% relative with
  the paired interval excluding 0.
- They are also compared with V and with MoGe-2.
- Relations: balanced accuracy against 50%.
- 2D: the JSON centres and the native boxes.
- Unparseable, zero or negative answers count as missing; coverage is reported, and every comparison uses the same
  objects.

**Spatial training (Q3).**
- SpatialAxiom-9B against Qwen3.5-9B, paired on the same items:
  - ask abs-rel for depth, size and distance, and the 2D error;
  - relation balanced accuracy (pooled, and per relation);
  - naming accuracy.
- A difference counts when its 95% interval excludes 0.
- The same comparisons are made against Qwen3.8-27B, and for SpaceQwen against its base if it runs.
- Each model's R values are reported with intervals.

**Primary endpoints:**
1. Q1: Qwen3.8-27B's asked depth against P.
2. Q2: R for L_img and W on depth (Qwen3.8-27B).
3. Q3: SpatialAxiom-9B against Qwen3.5-9B on asked depth.

Everything else is secondary.

## Known risks
- Linear probes only.
- One prompt wording.
- Images at most 640 px on the long side.
- COCO has 2D truth only.
- two_people comes from a few sequences of one studio, so its test frames share scenes with training (time split), and
  its depth range is narrow, so its prior is strong.
- KITTI's depth range is far wider than the training domains'.
- The models see the object list before the question, so the word states cannot see the question.
