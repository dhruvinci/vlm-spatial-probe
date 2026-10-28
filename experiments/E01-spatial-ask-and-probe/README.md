# E01 — Ask the model, then probe it, layer by layer

**Objective.** When a vision-language model is asked where an object is, how far away and how big, is its answer a
measurement? Where inside the model does the measurement live: in the vision tower, in the image tokens as they pass
through the language layers, or in the state of the word that names the object? Does spatial fine-tuning change
either?

**Setup.** Pre-registered ([PREREG.md](PREREG.md), committed before the run).

| item | details |
|---|---|
| data | 1,384 public images with truth: COCO people, NYU Depth v2 rooms, Harmony4D two-person capture, and KITTI streets held out entirely |
| models | Qwen3.8-27B; SpatialAxiom-9B and its base Qwen3.5-9B; SpaceQwen2.5-VL-3B (a SpatialVLM-recipe reproduction) and its base Qwen2.5-VL-3B |
| asked | JSON numbers, native boxes, yes/no relations and a naming check |
| probed | ridge probes on every vision block and every language layer: object-pooled image tokens, and the object word with the image, with no image, and with a mismatched image |
| baselines | name-only prior, shuffled labels, random features, MoGe-2 and RF-DETR |
| cost | one rented GPU, 2.37 h, $4.07 |

![Figure 1](../../docs/assets/fig1_baseball.png)

**Results** (Qwen3.8-27B; test split; abs-rel and % of the diagonal are lower-is-better). Details:
[RESULTS.md](RESULTS.md); every number: [data/tables.md](data/tables.md).

| depth (abs-rel) | value |
|---|---|
| name prior | 0.381 |
| asked (text) | 0.377 |
| probe: vision tower | 0.161 |
| probe: late-layer image tokens | 0.177 |
| probe: object word | 0.169 |
| MoGe-2 | 0.146 |

| 2D position (% of diagonal) | value |
|---|---|
| name prior | 16.2 |
| asked (native box) | 2.73 |
| asked (percent format, pre-registered; 34% of answers on a 0-1000 scale) | 167.0 |
| probe: vision tower | 3.91 |
| probe: object word | 6.98 |

- **Hypothesis falsified for depth and size.** The language side keeps the measurement: R = 0.93 for the late-layer
  image tokens and 0.96 for the object word. The word's depth comes from the image: with no image R = 0.09, with a
  mismatched image R = 0.05.
- **But the text answer doesn't use it.** Asked depth is no better than the name prior. The measurement is present but
  not read out.
- **Spatial fine-tuning fixes part of the read-out.** SpatialAxiom-9B asked depth is 0.233, against its base's 0.350.
  Relations go from 0.838 to 0.923. Its probes are unchanged, and naming drops slightly (0.912 to 0.884).
- **The SpatialVLM-recipe model does not help on these scenes.** SpaceQwen's test depth is no better than its base's
  (+0.021 [-0.017, 0.061]); it is better on street depth (-0.344) and worse on relations (-0.131) and naming (-0.041).

**Verdict.** Measurement is present in the language layers of all five models but is not read out in words, except
after spatial fine-tuning, and even then only partly (0.233 against its own probe's 0.161).

**Learned.** The bottleneck is the read-out, not the representation. Fine-tuning a pre-existing VLM can move the
spoken numbers towards what its hidden states already hold.

**Next.** Train a small decoder on the word state, and compare it with spatial fine-tuning on the same budget.

Code: `src/build_dataset.py`, `src/extract.py`, `src/probe.py`, `src/dense_probe.py`, `src/cv_reference.py`,
`src/analyse.py`, `src/report_tables.py`, `src/make_figures.py`; the rented-GPU run: `src/pod/`.
