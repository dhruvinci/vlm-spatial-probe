# E03 pre-registration: does a decoder trained on more scene types transfer?

Committed and pushed before any new data is extracted or any fit is made. E01's and E02's rules hold.

**Question.** E02's decoders trained on one or two domains and failed on the held-out one. Their absolute scale stayed
tied to the training domains. Here every decoder trains on three domains, and indoor gets a sibling domain
(synthetic indoor). Does the decoder then pass E02's bar?

## Data (public; built by `src/build_dataset.py`)
| domain | images | truth | licence |
|---|---|---|---|
| indoor (NYU Depth v2) | every labelled image with at least 2 qualifying objects (the E01 rules) | Kinect depth, 3D centre, size | no licence stated; research use |
| indoor_synth (Hypersim) | 50 public scenes (seeded), camera 00, 20 evenly spaced public frames each | from the scene's position images and camera pose: planar depth, 3D centre and size of each NYU40 object whose class is unique in the frame | CC BY-SA 3.0 |
| street (KITTI) | E01's 400 plus the next 600 in the same seeded order | LiDAR boxes | CC BY-NC-SA 3.0 |
| two_people (Harmony4D, calm sequences only: hugging, ballroom, sword, karate) | E01's calm frames plus new frames from the same sequences | multi-view body fits | MIT |
| people (COCO val2017) | E01's 384; 2D only | 2D boxes | annotations CC BY 4.0 |

Hypersim's per-pixel 3D comes from `position.hdf5` and the camera keyframe pose, in metres via the scene's
`meters_scale`. As a check, it must reproduce `depth_meters` (Euclidean distance) to within 1% median. If it does not,
Hypersim is dropped and recorded as such. Selection rules for named objects are E01's:
- the class occurs once in the frame;
- the object covers at least 1% of the image;
- at least 80% of its pixels are valid.

MMA and grappling frames are left out of E03 entirely; it is calm-only.

## Models and features
- Qwen3.8-27B and SpatialAxiom-9B, run with E01's code and prompt (`src/extract.py`).
- For the new images: the with-image and no-image passes. The mismatched-image pass is dropped to save GPU time.
- Spoken answers (E01's JSON ask) are collected on the Hypersim images. For the other domains, the E01 images already
  have them, and the spoken comparisons use those objects.

## Folds, decoders, comparisons, bar
Exactly as E02, with these domains:

| measure | folds |
|---|---|
| depth, 3D centre, distance between two objects | 4: hold out indoor, indoor_synth, street or two_people; train on the other three |
| size | 3: indoor, indoor_synth, street |
| 2D position | 5: people added |

Everything else is as in E02:
- layer and head chosen by inner leave-one-domain-out on the training domains;
- ridge and the same small MLP;
- the primary decoder is the object word;
- the same controls;
- the same specialists: MoGe-2 is run on the new images, RF-DETR with E02's class map plus the NYU40 names in that map.

**Pass bar (per model, measure and held-out domain; n ≥ 30):** the decoder beats the spoken answer (paired 95% CI of
spoken minus decoder above 0) and is within 1.5x the specialist on the same objects.

**Does variety help?** On the held-out indoor and street objects that E02 also tested, compare E03's decoder error with
E02's (paired; the CI must exclude 0). E03 trains on one more domain than E02 did.

## Known risks
- Hypersim is synthetic, and its frames come from a few cameras per scene.
- Street has no sibling domain: no licence-clear second street dataset was practical to build here (Argoverse 2 is
  CC BY-NC-SA 4.0 but too heavy for this run; Virtual KITTI 2's licence is unclear).
- Two models only.
