# E00 — Token-layout audit: where each image patch lands in the token sequence

**Objective.** Before any feature is read, check that every patch and merged image token can be put back at its true
image row and column, for each model's processor. An earlier probe on Qwen3.8 reshaped the raw patch sequence directly
into the image grid. That placed most locations in the wrong cell, because the processor emits patches in 2x2 groups.

**Setup.** Model-free: a synthetic image whose pixels encode their own patch row and column goes through each
installed image processor (transformers 5.21.0), on four grids (4x6, 8x4, 20x30, 160x90 patches). No weights, no
inference. A second check inside `src/extract.py` covers what the processor cannot show: Qwen2.5-VL's vision tower
reorders patches into attention windows. The extractor undoes that order and asserts, on the first batch, that the
restored block input equals the patch embeddings. With randomly initialised copies of each architecture
(`tests/check_extract.py`), batched (left-padded) states also equal single-image states within 1e-6.

**Results** (`data/audit.json`)

| processor | patch | direct reshape: misplaced locations (4x6 / 8x4 / 20x30 / 160x90) | restore_grid: misplaced | merged tokens in raster order |
|---|---|---|---|---|
| Qwen3.8-27B | 16 | 16 / 16 / 560 / 14,080 | 0 | yes |
| SpatialAxiom-9B | 16 | 16 / 16 / 560 / 14,080 | 0 | yes |
| Qwen3.5-9B | 16 | 16 / 16 / 560 / 14,080 | 0 | yes |
| SpaceQwen2.5-VL-3B | 14 | 16 / 16 / 560 / 14,080 | 0 | yes |
| Qwen2.5-VL-3B | 14 | 16 / 16 / 560 / 14,080 | 0 | yes |

**Verdict.** PASS on all five. A direct reshape misplaces 14,080 of 14,400 locations on a 160x90 grid; `restore_grid`
misplaces none, and merged token k sits at row k // (W/2), column k % (W/2) of the merged grid.

Code: `src/layout_audit.py`, `src/extract.py` (`restore_grid`, the window-order check), `tests/check_extract.py`.
