# Findings

E01, pre-registered. Every number is from `experiments/E01-spatial-ask-and-probe/data/tables.md` (line given).
abs-rel and % of the image diagonal: lower is better. Accuracy: higher is better.

1. **Qwen3.8-27B's object word holds metric depth almost as well as its vision tower.**
   - The word's state, with the image present, predicts depth at abs-rel 0.169 [0.152, 0.185]. The vision tower
     scores 0.161 and the name-only prior 0.381, so the word keeps 0.96 [0.91, 1.02] of the tower's gain
     (tables.md:12-16).
   - The image tokens in the last quarter of the language layers keep 0.93 [0.88, 0.98] (tables.md:14).
   - The pre-registered hypothesis, that the language side holds no measurement, is **falsified for depth in all five
     models** (tables.md:24, 309, 594, 879, 1164).
   - It is also falsified for size in the three Qwen3-family models (tables.md:56, 341, 626). For the two 3B models
     the tower itself does not beat the prior on size by 20% (G1; tables.md:911, 1196).
2. **That depth comes from the image.** The same word state with no image keeps 0.09 of the gain, and with a
   mismatched image 0.05 (tables.md:17-18).
3. **Asked in words, Qwen3.8-27B's depth is no better than the name prior.** It scores abs-rel 0.377 against the prior's
   0.381; prior minus asked is 0.003 [-0.031, 0.039] (tables.md:169, 180). By the pre-registered rule it is
   qualitative: present but not read out.
4. **2D position is read out, in the model's own format.**
   - Qwen3.8-27B's native boxes put object centres 2.73% of the diagonal from truth, against 3.91% for the
     vision-tower probe (tables.md:173, 184).
   - Asked for percentages, 34% of its answers came back on its 0-1000 grounding scale, so the pre-registered
     percent-format score is 167%.
5. **Spatial fine-tuning changes the read-out, not the representation.**
   - SpatialAxiom-9B's asked depth beats its base Qwen3.5-9B by 0.117 abs-rel (0.233 vs 0.350; [-0.153, -0.080])
     (tables.md:1434).
   - Its yes/no relations improve by 8.5 points of balanced accuracy (0.923 vs 0.838) (tables.md:1442).
   - Its probes match its base's (word R = 0.97 vs 0.95; tables.md:301, 586).
   - It costs a little naming (0.884 vs 0.912; tables.md:1443) and much of its 2D grounding (native boxes 11.7% vs
     3.5%; tables.md:469, 754).
6. **The SpatialVLM-recipe reproduction does not help on these scenes.** SpaceQwen2.5-VL-3B against its base
   (tables.md:1494-1501):
   - test depth: no difference, +0.021 [-0.017, 0.061] (lower is better);
   - held-out street depth: better by 0.344;
   - yes/no relations: worse by 0.131 (0.652 vs 0.783);
   - naming: worse by 0.041.
7. **Removing the MMA and grappling frames changes no verdict** (tables.md:84-162 and the "without MMA and grappling"
   tables).

Exploratory (not pre-registered): depth read from individual image tokens is best in the vision tower (block 18,
abs-rel 0.223). It fades through the language layers to 0.345 at the last one, close to the cell position alone
(0.368) (`data/dense_Qwen3.8-27B.json` on the dataset; Figure 5).

## E02: does a decoder transfer to an unseen domain? (pre-registered)
`experiments/E02-decoder-transfer/data/tables_e02.md`; calm frames; lower is better.

8. **A decoder on the object word does not transfer across scene types.** Trained on two domains and tested on the
   third, Qwen3.8-27B's depth decoder scores abs-rel 5.32, 0.73 and 0.82 (indoor, two people, street). Its spoken
   answer scores 0.29, 0.57 and 0.20, and MoGe-2 0.10, 0.23 and 0.10 (tables_e02.md:12-14).
9. **Across all five models it passes the pre-registered bar in 2 of 50 folds.** Both are 2D position on streets, and
   both are weak: the spoken answers were on the wrong scale, and the models' own native boxes beat the decoder there
   (tables_e02.md:94,107,162,175).
10. **Relative depth does not transfer either.** Scale-invariant abs-rel is 0.287, 0.123 and 0.386 for the decoder,
    against 0.115, 0.051 and 0.150 for the spoken answer (tables_e02.md:33-35).
11. **Inside one domain, the word holds orientation.** On KITTI, the object word reads a car's facing direction to
    within 9.4°-10.6°. The vision tower manages 13.9°-15.4° and the name prior 68°-80° (tables_e02.md:27-31). This is
    not a transfer test.

## E03: a decoder trained on more scene types (pre-registered)
`experiments/E03-many-domain-decoder/data/tables_e03.md`; calm frames only; lower is better.

16. **With three training domains, the decoder still loses to the spoken answer on every unseen scene type.**
    Qwen3.8-27B's depth decoder scores abs-rel 0.454, 0.284, 0.743 and 0.802 (held-out indoor, synthetic indoor, two
    people, street), against 0.288, 0.231, 0.574 and 0.199 spoken and 0.101, 0.178, 0.222 and 0.102 for MoGe-2
    (tables_e03.md:12-15). Across both models it passes the pre-registered bar in 1 of 26 folds, a weak 2D street fold
    as in E02 (tables_e03.md:101,110).
17. **Variety helps where a similar scene type is in training.** On E02's held-out objects, Qwen3.8-27B's indoor depth
    error falls from 5.32 to 0.45 once synthetic indoor is added; street, with no sibling, goes from 0.82 to 0.80.
    Variety helps in 11 of 12 paired comparisons (experiments/E03-many-domain-decoder/data/variety.json).
18. **Relative depth still does not transfer.** Scale-invariant abs-rel is 0.193-0.935 for the decoder against
    0.051-0.156 spoken (tables_e03.md:33-36).

## E05: a head on Qwen3.8's own vision tower (pre-registered)
`experiments/E05-dense-vision-head/data/results_e05.json`

12. **Per-pixel depth from the tower is coarse.** A trained head on Qwen3.8-27B's frozen vision tower reads NYU depth
    at abs-rel 0.186 [0.173, 0.198], against MoGe-2's 0.096 [0.085, 0.107] on the same pixels (lower is better). That
    is 1.94x, failing the pre-registered 1.5x bar.
13. **Joints from the tower fall well short of a pose model.** Given the same person box, a head on the tower places
    68.6% [67.0, 70.1] of visible COCO joints within 5% of the box diagonal, against 95.7% [94.9, 96.4] for ViTPose+
    base (higher is better). That is 27 points below, failing the pre-registered 10-point bar.

## E06: a second wording (pre-registered)
`experiments/E06-prompt-robustness/data/tables_e06.md`

14. **The spoken-answer findings survive rewording.** Qwen3.8-27B's asked depth stays at the name prior (0.395 vs
    0.394, tables_e06.md:16). SpatialAxiom-9B still beats its base (0.244 vs 0.342; difference -0.097 [-0.136, -0.059],
    tables_e06.md:10).
15. **Relations asked the other way round cost the base and older models balanced accuracy.**
    - Test split: up to 10 points. SpaceQwen2.5-VL-3B drops from 0.698 to 0.598; SpatialAxiom-9B does not move (0.926
      vs 0.926) (tables_e06.md:27,39).
    - All items, streets included: up to 12.8 points, for Qwen2.5-VL-3B (tables_e06.md:44).

## E07: why the decoder does not transfer (pre-registered)
`experiments/E07-transfer-diagnosis/data/tables_e07.md`; depth only; lower is better; E03's layers.

19. **What fails to transfer is absolute scale.** A decoder fitted to depth within each image (features and log
    depths centred per image) beats E03's decoder on relative depth in 7 of 8 held-out folds and never loses (for
    example, Qwen3.8-27B indoor 0.193 -> 0.142, street 0.452 -> 0.318; tables_e07.md:11,13,26,28,53). It still loses to
    the model's spoken answer on every fold: the words order objects better than any read-out trained here.
20. **Erasing scene type does not help.** With the training scene types made linearly indistinguishable (a classifier
    falls to the majority share, tables_e07.md:111-118), the decoder is worse in 5 of 8 folds and better in 2
    (tables_e07.md:52). The directions that separate scene types also carry depth level.
21. **Giving the camera helps only where object size is predictable.** Predicting physical size and dividing by the
    box's angular size passes E03's bar once: Qwen3.8-27B with two people held out, 0.155 against 0.574 spoken and 0.222
    for MoGe-2 (tables_e07.md:22). There, box plus camera alone already reach 0.266. Elsewhere it is mixed (4 folds
    better than E03's decoder, 4 worse; tables_e07.md:54).

## E04: the measures E01 did not cover (pre-registered)
`experiments/E04-more-measures/data/tables_e04.md`; test split; lower is better. Harmony4D calm sequences only.

22. **Qwen3.8-27B's object word holds which way things face, and the surface normal, from the image.**
    - Facing: the word reads it at 35.6° against 43.2° for the vision tower and 57.4° for the name prior. That keeps
      1.53 [1.31, 1.88] of the tower's gain (tables_e04.md:9).
    - With no image or a mismatched image, the word is at the prior (57.5°).
    - Surface normal: 32.0° against 31.6° for the tower and 40.1° for the prior. That keeps 0.95 of the tower's gain,
      and is 20.1% better than the prior, just over the 20% bar (tables_e04.md:11).
    - The pre-registered hypothesis is falsified for both. Facing is falsified in all five models, the normal in
      Qwen3.8-27B and SpatialAxiom-9B (tables_e04.md:46, 48, 83, 120, 157).
    - On people, every model's word reads facing better than its tower (Qwen3.8-27B 28.4° against 44.0°). On KITTI
      objects the tower is better (38.4° against 43.2°) (tables_e04.md:27-28).
23. **Asked, the models do not say it.**
    - Qwen3.8-27B's spoken facing is 52.4°, 8.7% better than the prior, so qualitative. The other four models are worse
      than the prior, at 72.6°-94.6° (tables_e04.md:9, 46, 83, 120, 157).
    - Spoken surface classes are no better than the prior (0.44 against 0.43, 1 - balanced accuracy;
      tables_e04.md:12).
24. **Counting is the one measure spoken as a measurement.**
    - Qwen3.8-27B's spoken count is off by 1.88 against the prior's 2.91 (35% better), and level with RF-DETR's 2.07
      (difference -0.19 [-0.44, 0.09]) (tables_e04.md:16, 19).
    - The class word's state does better than both, at 1.43. It beats the prior by 47-51% in every model, while the
      whole-image vision read-out gains about 5% (tables_e04.md:16, 53, 90, 127, 164).
25. **Speed, heading and dimensions are not in the vision read-out at the pre-registered bar**, in any model (G1).
    - For speed this is by construction: the tower's read-out sees only the last of the four frames.
    - Late image tokens, which can attend to earlier frames, read speed 15.5% better than the prior (tables_e04.md:14).
26. **Spatial fine-tuning improves spoken facing a little, without making it a measurement.** SpatialAxiom-9B is 9.9°
    [6.3, 13.9] better than its base Qwen3.5-9B, at 72.1° against 82.0° (tables_e04.md:194).
    - It gives joint angles in the wrong convention (about 10-15° for straight limbs, where 180 was asked).
