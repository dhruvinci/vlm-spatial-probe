# E06 results: the spoken answers under a second wording

**Status.**
- Pre-registered ([PREREG.md](PREREG.md), pushed as 0072699 before the run).
- Run on the shared E03/E06 rental: RTX PRO 6000, 2.09 h in all, $3.59 as settled; see `docs/SPEND.md`.
- All five models, E01's 1,385 images.
- Every number is printed by `src/report_e06.py` into [data/tables_e06.md](data/tables_e06.md), cited as
  `tables_e06.md:line`.
- Paired 95% bootstrap over images. Calm frames (no MMA or grappling) first; all frames follow.

## Pre-registered criteria: both hold (`tables_e06.md:7-10`)
1. **Qwen3.8-27B's asked depth is still "qualitative" under the second wording.**

   | | abs-rel |
   |---|---|
   | wording 1 | 0.370 |
   | wording 2 | 0.395 |
   | name prior | 0.394 |

   Prior minus wording 2 is -0.001 [-0.036, 0.036] (`tables_e06.md:16`).
2. **SpatialAxiom-9B still beats Qwen3.5-9B on asked depth under the second wording:** 0.244 vs 0.342, difference
   -0.097 [-0.136, -0.059], on the same 451 objects.

## What else changes with the wording (calm frames)
- **Qwen3.8-27B:**
  - test depth is slightly worse: +0.025 [0.012, 0.038];
  - street depth is unchanged: 0.199 vs 0.191;
  - relations (test) 0.844 vs 0.829, no significant change (`tables_e06.md:16-21`).
- **SpatialAxiom-9B:**
  - test depth does not change: +0.011 [-0.007, 0.038];
  - relations stay at 0.926 (38 of 1,313 answers flipped, in both directions) (`tables_e06.md:22-27`).
- **The base and older models are more sensitive:**
  - Qwen3.5-9B's street depth worsens from 0.359 to 0.519 (`tables_e06.md:29`).
  - Relations asked the other way round drop on the test split for Qwen3.5-9B (-0.024), Qwen2.5-VL-3B (-0.075) and
    SpaceQwen2.5-VL-3B (-0.100, from 0.698 to 0.598) (`tables_e06.md:33,39,45`).
  - On all relation items, held-out streets included, Qwen2.5-VL-3B drops 12.8 points (0.784 to 0.656,
    `tables_e06.md:44`).

## What this means
- **The paper's spoken-answer claims are not an artefact of one phrasing.**
  - The general models' depth stays at name-prior level.
  - The spatial fine-tune's advantage over its base holds.
  - The spatial fine-tune is also the most stable under rewording.
- **The relation answers of the base and older models depend on how the question is put.** Asked the other way
  round, the SpatialVLM-recipe model loses 10 points of balanced accuracy on the test split; Qwen2.5-VL-3B loses 12.8
  points on all items.
