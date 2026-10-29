# E05 — Can a head on Qwen3.8's own vision tower measure like the specialists?

**Objective.** E01 and E02 show coarse geometry on the language side. Is fine geometry available from the vision
tower to a trained head, at the precision engagement measures need?

**Setup.**
- Pre-registered ([PREREG.md](PREREG.md)).
- Qwen3.8-27B's frozen vision tower, loaded alone from one checkpoint shard.
- **Depth head:** blocks 6, 13, 20, 26 → per-patch MLP; NYU, E01's split by scene. Specialist: MoGe-2.
- **Joints head:** blocks 13, 20, 26 → which patch holds each joint, plus an offset; 3,000 COCO people to train, 600 to
  test. Specialist: ViTPose+ base, given the same box.
- Local GPU; $0.

![E05](../../docs/assets/fig8_dense_head.png)

**Results:**

| measure | head | specialist |
|---|---|---|
| per-pixel depth, abs-rel (lower is better) | 0.186 | MoGe-2 0.096 |
| joints, PCK@5% of the box diagonal (higher is better) | 0.686 | ViTPose+ 0.957 |

Both fail the pre-registered bar: depth by 1.94x, joints by 27 points.

**Verdict.** The tower holds geometry a head can read, at about half the specialists' precision. Not enough for joint-
or centimetre-level engagement measures.

**Next.** Keep the specialists as the measurement layer, with the VLM reasoning over their numbers.

Details: [RESULTS.md](RESULTS.md). Code: `src/dense_head.py`, `src/make_figures_e05.py`.
