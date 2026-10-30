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

## Second rental: E03 and E06 (2026-10-29)
| item | value |
|---|---|
| pod | `i8z2otkeq1lt2z`, community cloud, 1x RTX PRO 6000 Blackwell Workstation (96 GB), $1.69/h |
| uptime | 2.09 h (04:32-06:37 UTC) |
| account balance | $17.5386 before; $13.9462 after billing settled, with no pods and no volumes |
| **spend** | **$3.59** (cap for this rental: $7) |

What the time went on:
- **E06 (second wording), all five models:** 27B 37 min; 9B 10-17 min each; 3B about 8 min each.
- **E03 extraction:**
  - SpatialAxiom-9B: 15 min.
  - Qwen3.8-27B: 16 min, after a re-download. Its first attempt ran out of GPU memory because I had started
    SpatialAxiom's E06 alongside it, and the run script then deleted the 27B weights. About 10 extra minutes.
- **E03 data build:**
  - KITTI by byte range.
  - Hypersim by byte range: 50 scenes gave 465 frames with at least 2 qualifying objects.
  - The NYU superset was dropped. The NYU server gave the pod 0.4 MB/s, and uploads to the pod ran at about
    50 KB/s, so E03 uses E01's 400 NYU images.
- **Idle:** about 15 minutes at the end, during a local power and network cut that stopped the watchdog. It was
  restarted, fetched everything, and terminated the pod.

E07 (2026-10-29) ran locally on CPU on the cached features: $0.

## Third rental: E04 (2026-10-30)
| item | value |
|---|---|
| pod | `ebhchtsyp63y3n`, community cloud, 1x RTX PRO 6000 Blackwell Workstation (96 GB), $1.69/h |
| uptime | 3.22 h (05:42-08:55 UTC) |
| account balance | $13.9462 before; $8.4211 after billing settled, with no pods left on the account |
| **spend** | **$5.53** (cap for this rental: $9.34, approved with $7 added; paper 04 cap $17) |

What the time went on (`logs/run_all.log` and each model's `meta.json`):
- **Setup and data, 20 minutes before the first extraction.**
  - The NYU labelled file (2.9 GB) came from its server. It started at about 80 KB/s, so a local copy was uploaded as a
    fallback, but the server sped up and finished first. The fallback was deleted unused.
  - The NYU and Hypersim surface truth were built on the pod, which has h5py. Hypersim was read by byte range: 600
    frames in about 11 minutes.
  - The five models downloaded meanwhile.
- **Extraction** (features with the image, no image and a mismatched image, then the asked answers):

  | model | time | of which asked |
  |---|---|---|
  | Qwen3.8-27B | 70.3 min | 36.1 min |
  | SpatialAxiom-9B | 25.8 min | 13.4 min |
  | Qwen3.5-9B | 35.6 min | 23.4 min |
  | SpaceQwen2.5-VL-3B | 17.2 min | 8.2 min |
  | Qwen2.5-VL-3B | 16.5 min | 7.4 min |

- **Download:** about 32 GB of features (every layer) at about 11-13 MB/s, overlapped with the extractions. The last
  fetch and the termination took 5 minutes after the run ended.
- **No-cost hiccup:** the start command's SSH session hung, though the run had started.

`src/pod/launch.py watch`, restarted by a wrapper after any SSH error, terminated the pod once every model was fetched,
and the empty pod list was verified through the API.

**Paper 04 total, three rentals: $13.18** ($21.6038 → $8.4211), against the $17 cap.
