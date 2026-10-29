# E06 — Do the spoken answers hold under a second wording?

**Objective.** E01 asked each question one way. Do its conclusions about what the models say survive a different
wording of the same questions?

**Setup.**
- Pre-registered ([PREREG.md](PREREG.md)).
- All five models, E01's 1,385 images.
- A second JSON ask with the same schema, and the relation questions asked the other way round (same truth).
- Cost: part of the shared E03/E06 rental ($3.47 in all).

**Results** (calm frames; abs-rel, lower is better):
- Qwen3.8-27B's asked depth stays at the name prior: 0.395 vs 0.394.
- SpatialAxiom-9B still beats its base: 0.244 vs 0.342, difference -0.097 [-0.136, -0.059].
- **Both pre-registered criteria hold.**
- SpatialAxiom's relations do not move (0.926 vs 0.926). The base and older models lose up to 10 points when a relation
  is asked the other way round.

**Verdict.** The spoken-answer findings are robust to the wording. The relation answers of the base and older models
are not.

Details: [RESULTS.md](RESULTS.md); every number: [data/tables_e06.md](data/tables_e06.md). Code: `src/extract.py`
(parts `ask2`, `yesno2`), `src/analyse_e06.py`, `src/report_e06.py`.
