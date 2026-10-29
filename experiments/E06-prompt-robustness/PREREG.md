# E06 pre-registration: do the spoken answers hold under a second wording?

Committed and pushed before the run. E01's rules hold.

**Question.** E01 asked each question one way. Do its conclusions about what the models *say* survive a different
wording of the same questions?

**Run.** All five models, E01's 1,385 images, greedy decoding, no thinking (`src/extract.py`, parts `ask2` and
`yesno2`).
- **The second JSON ask** (same output schema as E01, so the same parser applies): "Look at the listed objects. For each
  one, say how far it is from the camera in metres, how big it is (its longest side, in metres), and where its centre
  is in the image as a percentage of the width and height (0 to 100, from the top-left). Then give the distance in
  metres between every pair."
- **The relations, asked the other way round** (same truth, roles swapped):

  | relation | second wording |
  |---|---|
  | closer | "Is the {b} farther from the camera than the {a}?" |
  | left of | "In the image, is the {b} to the right of the {a}?" |
  | above | "In the image, is the {b} lower down than the {a}?" |
  | bigger | "In reality, is the {b} smaller than the {a}?" |

  Same items as E01.

**Scored as E01:**
- asked depth, size and pair-distance abs-rel against the name prior, on the test split and on held-out streets;
- the 2D centre;
- relation balanced accuracy.

The two wordings are compared by paired bootstrap over images (2,000 resamples).

**The conclusions are robust if both hold under the second wording:**
1. Qwen3.8-27B's asked depth (test split) is still "qualitative" by E01's rule: under 20% better than the name prior,
   or a CI that includes 0.
2. SpatialAxiom-9B's asked depth is still better than Qwen3.5-9B's: the paired CI of the difference lies below 0.

Each part is reported separately. If either fails, the paper says the result depends on the wording.
