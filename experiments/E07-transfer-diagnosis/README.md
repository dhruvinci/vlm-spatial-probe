# E07 — Why doesn't the decoder transfer?

**Objective.** E02 and E03: a decoder on the object word's hidden state loses to the model's own spoken answer on every
scene type it never saw. Three likely causes, each tested by changing one thing about E03's decoder: it learns scene
type rather than geometry (A1, erase scene type); absolute scale fails while the geometry inside an image transfers
(A2, fit and score depth within each image); metric depth is ambiguous without the camera (A3, predict physical size
and divide by the box's angular size).

**Setup.**
- Pre-registered ([PREREG.md](PREREG.md), pushed before any fit). E03's data, folds, layers and heads; Qwen3.8-27B and
  SpatialAxiom-9B; depth only.
- Controls: the box's own 2D cues (A2); box + camera with no model (A3).
- Cost: $0, CPU, on the cached features.

![E07](../../docs/assets/fig9_diagnosis.png)

**Results** (8 held-out folds; `data/tables_e07.md:52-54`):

| hypothesis | arm | helps | hurts | reading |
|---|---|---|---|---|
| scene-type shortcut | A1, erase scene type | 2 | 5 | mixed (mostly hurts) |
| absolute scale | A2, within the image | 7 | 0 | **supported** |
| missing camera | A3, give the camera | 4 | 4 | mixed; one pass (Qwen3.8-27B, two people: 0.155 vs 0.574 spoken, 0.222 MoGe-2) |

**Verdict.** The decoder's failure on new scene types is mostly a failure of absolute scale. The ordering of depths
inside an image transfers; metres do not. The model's spoken answer still orders objects better than any read-out
here. Metres need what specialists have: the camera and objects of known size.

**Checks.** The refitted E03 decoder reproduces E03 within 5% on 7 of 8 folds (the miss: Qwen3.8-27B, two people,
0.819 vs 0.743, an MLP on the first language layer refitted on CPU). After erasure, a classifier cannot tell the
training scene types apart.

Details: [RESULTS.md](RESULTS.md); every number: [data/tables_e07.md](data/tables_e07.md). Code: `src/cameras_e07.py`,
`src/diagnose_e07.py`, `src/analyse_e07.py`, `src/make_figures_e07.py`.
