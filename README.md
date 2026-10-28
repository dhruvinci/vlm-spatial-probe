# Naming isn't locating: a probe inside the spatial information of vision-language models

**Status: pre-registered, running.** The pre-registration
([experiments/E01-spatial-ask-and-probe/PREREG.md](experiments/E01-spatial-ask-and-probe/PREREG.md)) was committed
before any GPU run. Results, figures and the findings will be added here when the run is complete.

**Question.** When a vision-language model is asked where an object is, how far away it is or how big it is, does its
answer carry a measurement? And where inside the model does that measurement live: in the vision tower, in the image
tokens as they pass through the language layers, or in the state of the word that names the object?

Experiments:
- [E00](experiments/E00-token-layout-audit/): token-layout audit. Where each image patch lands in the token sequence,
  checked on each model's processor before any features are read.
- [E01](experiments/E01-spatial-ask-and-probe/): ask the model, then probe it layer by layer (Qwen3.8-27B; SpatialAxiom-9B
  against its base Qwen3.5-9B; SpaceQwen2.5-VL-3B against its base Qwen2.5-VL-3B).

Code: MIT (`LICENSE`). Text and figures: CC BY 4.0 (`LICENSE-docs`).
