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
