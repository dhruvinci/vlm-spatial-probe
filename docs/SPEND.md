# Spend log (RunPod)

Hard cap: $10 in total, including failed runs and storage. One pod, no network volume.

| item | value |
|---|---|
| pod | `ty10p2jowde0oz`, community cloud, 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB), $1.69/h (no A100 80 GB was available) |
| image | `runpod/pytorch:1.4.0-cu1290-torch291-ubuntu2404` (torch 2.9.1+cu128 inside), container disk 200 GB |
| created / terminated | 2026-10-28 17:47 UTC / 2026-10-28 20:09 UTC (uptime 2.37 h) |
| account balance | $21.6038 before; $17.5386 after billing settled, with no pods and no volumes left on the account |
| **total spend** | **$4.07** |

What the time went on (from `logs/run_all.log` and each model's `meta.json` on the dataset):
- **Setup:** about 20 minutes, including two false starts.
  - The image's Python is marked externally managed, so pip needed `PIP_BREAK_SYSTEM_PACKAGES=1`.
  - The Conv3d patch embedding ran at about 45 s per batch until it was replaced by the equivalent matrix product.
- **Extraction:**

  | model | time |
  |---|---|
  | Qwen3.8-27B | 53.0 min |
  | SpaceQwen2.5-VL-3B | 12.6 min |
  | Qwen2.5-VL-3B | 12.2 min |
  | SpatialAxiom-9B | 15.5 min |
  | Qwen3.5-9B | 17.5 min |

- **Download:** 22 GB of features to the local cache at about 10 MB/s, mostly overlapped with the extractions.

The pod was terminated by `src/pod/launch.py watch` once every model had been fetched, and the empty pod list was
verified through the API.
