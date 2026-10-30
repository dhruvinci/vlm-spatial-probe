# E04 tables (printed by src/analyse_e04.py from results_e04.json)

Test split. Errors: lower is better (angles in degrees; speed in m/s; count: absolute error; dims: abs-rel; surface class: 1 - balanced accuracy). 95% intervals: bootstrap over images or clips, paired. P = name-only prior.

## Qwen3.8-27B

| target | n | P | V | L_img | W | W no image | W mismatched | shuffled | random | spoken (coverage) | specialist | G0 | G1 | R L_img | R W | outcome | attribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| facing | 1255 | 57.433 [54.786, 59.978] | 43.155 [40.932, 45.289] | 37.502 [35.324, 39.738] | 35.552 [33.221, 37.712] | 57.540 [54.880, 60.086] | 57.526 [54.840, 60.061] | 83.114 [80.534, 85.839] | 84.489 [82.058, 87.090] | 52.427 [49.456, 55.439] (100%) | n/a | pass | pass | 1.40 | 1.53 | falsified | yes |
| joints | 698 | 23.936 [23.119, 24.778] | 16.664 [15.618, 17.705] | 20.842 [19.839, 21.894] | 19.686 [18.497, 20.914] | 23.934 [23.122, 24.775] | 23.913 [23.103, 24.757] | 23.871 [23.066, 24.711] | 23.923 [23.113, 24.764] | 22.012 [20.995, 23.030] (95%) | n/a | fail | pass | 0.43 | 0.58 | controls failed (G0): not interpreted | yes |
| normal | 396 | 40.113 [37.082, 43.192] | 31.604 [29.726, 33.587] | 33.246 [31.422, 35.090] | 32.040 [29.785, 34.352] | 40.155 [37.147, 43.162] | 40.190 [37.182, 43.206] | 52.979 [51.104, 54.945] | 52.884 [51.010, 54.845] | n/a | 9.270 [7.741, 10.924] | pass | pass | 0.81 | 0.95 | falsified | yes |
| surface_class | 396 | 0.427 [0.385, 0.468] | 0.294 [0.252, 0.337] | 0.364 [0.324, 0.406] | 0.363 [0.323, 0.403] | 0.443 [0.412, 0.475] | 0.440 [0.397, 0.483] | 0.668 [0.654, 0.681] | 0.679 [0.637, 0.716] | 0.436 [0.402, 0.470] (100%) | n/a | pass | pass | 0.47 | 0.48 | partial | yes |
| dims | 797 | 0.212 [0.195, 0.232] | 0.247 [0.230, 0.267] | 0.258 [0.229, 0.301] | 0.192 [0.177, 0.209] | 0.211 [0.194, 0.231] | 0.212 [0.195, 0.232] | 0.574 [0.530, 0.630] | 0.571 [0.527, 0.626] | 0.427 [0.330, 0.583] (30%) | n/a | pass | fail | 1.32 | -0.57 | premise fails (G1): the tower does not carry the measure | yes |
| speed | 171 | 2.058 [1.708, 2.508] | 2.332 [1.989, 2.729] | 1.738 [1.427, 2.116] | 2.064 [1.717, 2.512] | 2.033 [1.693, 2.490] | 2.058 [1.710, 2.505] | 2.381 [2.044, 2.808] | 2.376 [2.046, 2.799] | 1.779 [1.375, 2.262] (100%) | n/a | pass | fail | -1.17 | 0.02 | premise fails (G1): the tower does not carry the measure | yes |
| heading | 76 | 87.147 [73.611, 100.563] | 73.680 [61.137, 86.383] | 72.463 [60.393, 85.083] | 76.031 [63.106, 89.030] | 87.018 [73.557, 100.478] | 86.285 [73.076, 99.467] | 92.848 [87.442, 98.112] | 84.980 [72.196, 97.347] | n/a | n/a | pass | fail | 1.09 | 0.83 | premise fails (G1): the tower does not carry the measure | yes |
| count | 400 | 2.910 [2.693, 3.116] | 2.763 [2.563, 2.960] | 2.925 [2.722, 3.124] | 1.431 [1.293, 1.577] | 2.825 [2.625, 3.018] | 2.857 [2.651, 3.057] | 3.281 [3.072, 3.483] | 3.394 [3.172, 3.617] | 1.883 [1.592, 2.215] (100%) | 2.072 [1.817, 2.338] | pass | fail | -0.10 | 10.09 | premise fails (G1): the tower does not carry the measure | no |

Exploratory (not pre-registered): spoken motion direction, 5 classes, n = 171 answered of 171: accuracy 0.409 [0.333, 0.485]; always 'still' (the training split's most common class) 0.556 [0.480, 0.632].
Spoken against the specialist (count, paired, n = 400): spoken 1.883 [1.592, 2.215], specialist 2.072 [1.817, 2.338], difference -0.190 [-0.437, 0.090].
Exploratory (not pre-registered): spoken count with off-form answers also parsed ({class: n}, {class: [{count}]}, a list of boxes): coverage 100%, error 1.883 [1.592, 2.215].
Exploratory (not pre-registered): spoken joint angles read as flexion (180 - answer): 111.151 [108.010, 114.283].

Per source (Qwen3.8-27B; the probes are fitted on all sources of a measure together):

| target | source | n | P | V | L_img | W |
|---|---|---|---|---|---|---|
| facing | h4d | 698 | 47.951 | 43.999 | 33.736 | 28.443 |
| facing | kitti_obj | 400 | 75.655 | 38.412 | 39.014 | 43.185 |
| facing | objectron | 157 | 53.165 | 51.488 | 50.387 | 47.711 |
| joints | h4d | 698 | 23.936 | 16.664 | 20.842 | 19.686 |
| normal | hypersim | 240 | 41.784 | 33.620 | 35.133 | 35.199 |
| normal | nyu | 156 | 37.541 | 28.503 | 30.344 | 27.180 |
| dims | hypersim | 240 | 0.418 | 0.395 | 0.431 | 0.411 |
| dims | kitti_obj | 400 | 0.109 | 0.188 | 0.190 | 0.089 |
| dims | objectron | 157 | 0.160 | 0.172 | 0.170 | 0.120 |
| speed | h4d_motion | 48 | 0.229 | 0.906 | 0.384 | 0.247 |
| speed | kitti_trk | 123 | 2.771 | 2.889 | 2.266 | 2.773 |
| heading | h4d_motion | 18 | 86.593 | 100.186 | 90.689 | 75.493 |
| heading | kitti_trk | 58 | 87.319 | 65.454 | 66.806 | 76.199 |
| count | coco | 400 | 2.910 | 2.763 | 2.925 | 1.431 |

## SpatialAxiom-9B

| target | n | P | V | L_img | W | W no image | W mismatched | shuffled | random | spoken (coverage) | specialist | G0 | G1 | R L_img | R W | outcome | attribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| facing | 1255 | 57.433 [54.786, 59.978] | 40.006 [37.763, 42.182] | 40.444 [38.156, 42.756] | 37.456 [35.332, 39.711] | 57.517 [54.870, 60.075] | 59.565 [56.654, 62.171] | 83.021 [80.449, 85.765] | 84.383 [81.930, 86.982] | 72.635 [69.753, 75.461] (96%) | n/a | pass | pass | 0.97 | 1.15 | falsified | yes |
| joints | 698 | 23.936 [23.119, 24.778] | 17.100 [16.043, 18.154] | 18.745 [17.648, 19.864] | 19.085 [17.917, 20.278] | 23.934 [23.120, 24.775] | 24.020 [23.212, 24.869] | 23.900 [23.092, 24.740] | 23.923 [23.113, 24.764] | 103.960 [99.652, 108.302] (100%) | n/a | pass | pass | 0.76 | 0.71 | partial | yes |
| normal | 396 | 40.113 [37.082, 43.192] | 31.210 [29.397, 33.159] | 32.934 [31.062, 34.782] | 29.875 [27.634, 32.151] | 40.178 [37.169, 43.207] | 40.022 [37.020, 43.041] | 52.922 [51.072, 54.872] | 52.884 [51.010, 54.845] | n/a | 9.270 [7.741, 10.924] | pass | pass | 0.81 | 1.15 | falsified | yes |
| surface_class | 396 | 0.427 [0.385, 0.468] | 0.233 [0.192, 0.277] | 0.359 [0.318, 0.404] | 0.338 [0.299, 0.379] | 0.445 [0.414, 0.478] | 0.436 [0.393, 0.481] | 0.643 [0.627, 0.659] | 0.687 [0.645, 0.726] | 0.475 [0.424, 0.524] (95%) | n/a | pass | pass | 0.35 | 0.46 | supported | yes |
| dims | 797 | 0.212 [0.195, 0.232] | 0.239 [0.223, 0.260] | 0.261 [0.231, 0.307] | 0.171 [0.159, 0.185] | 0.211 [0.194, 0.231] | 0.211 [0.194, 0.231] | 0.574 [0.530, 0.630] | 0.570 [0.527, 0.624] | 0.392 [0.366, 0.419] (29%) | n/a | pass | fail | 1.81 | -1.52 | premise fails (G1): the tower does not carry the measure | yes |
| speed | 171 | 2.058 [1.708, 2.508] | 2.143 [1.822, 2.553] | 1.860 [1.537, 2.227] | 1.879 [1.578, 2.256] | 2.025 [1.686, 2.484] | 2.056 [1.711, 2.502] | 2.385 [2.052, 2.813] | 2.376 [2.046, 2.799] | 1.413 [1.015, 1.951] (87%) | n/a | pass | fail | -2.31 | -2.09 | premise fails (G1): the tower does not carry the measure | yes |
| heading | 76 | 87.147 [73.611, 100.563] | 76.168 [63.295, 89.183] | 74.083 [61.679, 86.909] | 81.272 [69.792, 93.093] | 87.001 [73.691, 100.501] | 85.598 [72.237, 98.753] | 89.181 [81.385, 96.588] | 86.374 [73.780, 99.072] | n/a | n/a | pass | fail | 1.19 | 0.54 | premise fails (G1): the tower does not carry the measure | yes |
| count | 400 | 2.910 [2.693, 3.116] | 2.776 [2.578, 2.966] | 2.815 [2.620, 3.005] | 1.491 [1.354, 1.632] | 2.822 [2.620, 3.016] | 2.845 [2.649, 3.038] | 3.282 [3.074, 3.485] | 3.367 [3.146, 3.589] | 2.055 [1.792, 2.342] (100%) | 2.072 [1.817, 2.338] | pass | fail | 0.71 | 10.58 | premise fails (G1): the tower does not carry the measure | no |

Exploratory (not pre-registered): spoken motion direction, 5 classes, n = 149 answered of 171: accuracy 0.342 [0.268, 0.423]; always 'still' (the training split's most common class) 0.517 [0.436, 0.597].
Spoken against the specialist (count, paired, n = 399): spoken 2.055 [1.792, 2.342], specialist 2.060 [1.805, 2.319], difference -0.005 [-0.185, 0.175].
Exploratory (not pre-registered): spoken count with off-form answers also parsed ({class: n}, {class: [{count}]}, a list of boxes): coverage 100%, error 2.055 [1.792, 2.342].
Exploratory (not pre-registered): spoken joint angles read as flexion (180 - answer): 34.986 [32.834, 37.175].

Per source (SpatialAxiom-9B; the probes are fitted on all sources of a measure together):

| target | source | n | P | V | L_img | W |
|---|---|---|---|---|---|---|
| facing | h4d | 698 | 47.951 | 39.753 | 35.778 | 29.559 |
| facing | kitti_obj | 400 | 75.655 | 36.867 | 45.825 | 46.257 |
| facing | objectron | 157 | 53.165 | 49.128 | 47.477 | 50.140 |
| joints | h4d | 698 | 23.936 | 17.100 | 18.745 | 19.085 |
| normal | hypersim | 240 | 41.784 | 32.843 | 34.409 | 32.709 |
| normal | nyu | 156 | 37.541 | 28.699 | 30.665 | 25.514 |
| dims | hypersim | 240 | 0.418 | 0.389 | 0.427 | 0.340 |
| dims | kitti_obj | 400 | 0.109 | 0.176 | 0.209 | 0.089 |
| dims | objectron | 157 | 0.160 | 0.171 | 0.141 | 0.121 |
| speed | h4d_motion | 48 | 0.229 | 0.941 | 0.327 | 0.423 |
| speed | kitti_trk | 123 | 2.771 | 2.612 | 2.459 | 2.448 |
| heading | h4d_motion | 18 | 86.593 | 94.650 | 96.955 | 85.118 |
| heading | kitti_trk | 58 | 87.319 | 70.432 | 66.984 | 80.079 |
| count | coco | 400 | 2.910 | 2.776 | 2.815 | 1.491 |

## Qwen3.5-9B

| target | n | P | V | L_img | W | W no image | W mismatched | shuffled | random | spoken (coverage) | specialist | G0 | G1 | R L_img | R W | outcome | attribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| facing | 1255 | 57.433 [54.786, 59.978] | 42.267 [39.972, 44.468] | 41.013 [38.801, 43.189] | 37.109 [34.872, 39.424] | 57.493 [54.847, 60.056] | 58.437 [55.591, 61.087] | 83.042 [80.465, 85.789] | 84.423 [81.950, 86.994] | 81.436 [78.312, 84.430] (94%) | n/a | pass | pass | 1.08 | 1.34 | falsified | yes |
| joints | 698 | 23.936 [23.119, 24.778] | 16.504 [15.488, 17.550] | 18.602 [17.493, 19.681] | 19.041 [17.875, 20.195] | 23.929 [23.124, 24.772] | 23.917 [23.107, 24.763] | 23.932 [23.121, 24.779] | 23.923 [23.113, 24.764] | 30.159 [28.705, 31.657] (100%) | n/a | pass | pass | 0.72 | 0.66 | partial | yes |
| normal | 396 | 40.113 [37.082, 43.192] | 32.133 [30.289, 34.112] | 32.064 [30.089, 34.016] | 30.282 [27.985, 32.536] | 40.173 [37.166, 43.191] | 39.961 [36.987, 43.008] | 53.028 [51.150, 54.985] | 52.884 [51.010, 54.845] | n/a | 9.270 [7.741, 10.924] | pass | fail | 1.01 | 1.23 | premise fails (G1): the tower does not carry the measure | yes |
| surface_class | 396 | 0.427 [0.385, 0.468] | 0.242 [0.200, 0.285] | 0.370 [0.329, 0.414] | 0.325 [0.282, 0.367] | 0.445 [0.414, 0.478] | 0.418 [0.376, 0.461] | 0.659 [0.647, 0.671] | 0.684 [0.644, 0.721] | 0.474 [0.439, 0.512] (86%) | n/a | pass | pass | 0.31 | 0.55 | partial | yes |
| dims | 797 | 0.212 [0.195, 0.232] | 0.237 [0.222, 0.254] | 0.260 [0.234, 0.297] | 0.179 [0.166, 0.193] | 0.211 [0.194, 0.231] | 0.212 [0.194, 0.232] | 0.574 [0.530, 0.630] | 0.570 [0.527, 0.625] | 0.461 [0.360, 0.613] (24%) | n/a | pass | fail | 1.91 | -1.35 | premise fails (G1): the tower does not carry the measure | yes |
| speed | 171 | 2.058 [1.708, 2.508] | 1.950 [1.621, 2.357] | 1.950 [1.605, 2.353] | 2.001 [1.674, 2.387] | 2.028 [1.689, 2.484] | 2.060 [1.715, 2.503] | 2.381 [2.045, 2.807] | 2.376 [2.046, 2.799] | 1.653 [1.256, 2.125] (100%) | n/a | pass | fail | 1.00 | 0.53 | premise fails (G1): the tower does not carry the measure | yes |
| heading | 76 | 87.147 [73.611, 100.563] | 73.954 [60.894, 87.169] | 77.415 [65.107, 90.298] | 83.653 [70.307, 97.839] | 87.019 [73.574, 100.470] | 87.339 [73.175, 100.627] | 88.243 [79.845, 96.059] | 86.322 [73.707, 99.338] | n/a | n/a | pass | fail | 0.74 | 0.26 | premise fails (G1): the tower does not carry the measure | yes |
| count | 400 | 2.910 [2.693, 3.116] | 2.748 [2.555, 2.934] | 2.840 [2.639, 3.030] | 1.442 [1.313, 1.582] | 2.828 [2.627, 3.021] | 2.840 [2.637, 3.033] | 3.281 [3.071, 3.485] | 3.368 [3.146, 3.589] | 2.100 [1.521, 2.973] (65%) | 2.072 [1.817, 2.338] | pass | fail | 0.43 | 9.03 | premise fails (G1): the tower does not carry the measure | no |

Exploratory (not pre-registered): spoken motion direction, 5 classes, n = 171 answered of 171: accuracy 0.573 [0.497, 0.649]; always 'still' (the training split's most common class) 0.556 [0.480, 0.632].
Spoken against the specialist (count, paired, n = 261): spoken 2.100 [1.521, 2.973], specialist 2.379 [2.046, 2.725], difference -0.280 [-0.810, 0.556].
Exploratory (not pre-registered): spoken count with off-form answers also parsed ({class: n}, {class: [{count}]}, a list of boxes): coverage 93%, error 1.790 [1.361, 2.410].
Exploratory (not pre-registered): spoken joint angles read as flexion (180 - answer): 113.226 [109.286, 117.267].

Per source (Qwen3.5-9B; the probes are fitted on all sources of a measure together):

| target | source | n | P | V | L_img | W |
|---|---|---|---|---|---|---|
| facing | h4d | 698 | 47.951 | 43.993 | 35.570 | 27.939 |
| facing | kitti_obj | 400 | 75.655 | 36.888 | 46.884 | 48.161 |
| facing | objectron | 157 | 53.165 | 48.301 | 50.255 | 49.716 |
| joints | h4d | 698 | 23.936 | 16.504 | 18.602 | 19.041 |
| normal | hypersim | 240 | 41.784 | 33.909 | 33.339 | 32.760 |
| normal | nyu | 156 | 37.541 | 29.400 | 30.103 | 26.469 |
| dims | hypersim | 240 | 0.418 | 0.391 | 0.416 | 0.353 |
| dims | kitti_obj | 400 | 0.109 | 0.176 | 0.205 | 0.096 |
| dims | objectron | 157 | 0.160 | 0.159 | 0.160 | 0.122 |
| speed | h4d_motion | 48 | 0.229 | 0.567 | 0.575 | 0.324 |
| speed | kitti_trk | 123 | 2.771 | 2.489 | 2.486 | 2.655 |
| heading | h4d_motion | 18 | 86.593 | 92.423 | 106.658 | 81.996 |
| heading | kitti_trk | 58 | 87.319 | 68.222 | 68.339 | 84.167 |
| count | coco | 400 | 2.910 | 2.748 | 2.840 | 1.442 |

## SpaceQwen2.5-VL-3B

| target | n | P | V | L_img | W | W no image | W mismatched | shuffled | random | spoken (coverage) | specialist | G0 | G1 | R L_img | R W | outcome | attribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| facing | 1255 | 57.433 [54.786, 59.978] | 44.115 [41.786, 46.466] | 42.538 [40.327, 44.813] | 43.521 [40.910, 46.039] | 57.468 [54.823, 60.042] | 57.825 [55.147, 60.443] | 83.037 [80.463, 85.762] | 83.071 [80.495, 85.819] | 88.219 [85.380, 91.128] (100%) | n/a | pass | pass | 1.12 | 1.04 | falsified | yes |
| joints | 698 | 23.936 [23.119, 24.778] | 16.922 [15.867, 17.977] | 19.998 [18.938, 21.073] | 19.357 [18.235, 20.500] | 23.930 [23.122, 24.771] | 23.926 [23.121, 24.769] | 23.902 [23.094, 24.737] | 23.923 [23.113, 24.764] | 56.175 [53.416, 58.914] (100%) | n/a | pass | pass | 0.56 | 0.65 | partial | yes |
| normal | 396 | 40.113 [37.082, 43.192] | 34.153 [32.123, 36.167] | 35.129 [33.062, 37.233] | 35.343 [32.817, 37.977] | 40.132 [37.157, 43.120] | 40.355 [37.364, 43.262] | 52.770 [50.888, 54.742] | 52.938 [51.098, 54.871] | n/a | 9.270 [7.741, 10.924] | pass | fail | 0.84 | 0.80 | premise fails (G1): the tower does not carry the measure | yes |
| surface_class | 396 | 0.427 [0.385, 0.468] | 0.317 [0.273, 0.363] | 0.388 [0.346, 0.430] | 0.389 [0.349, 0.429] | 0.456 [0.425, 0.490] | 0.446 [0.404, 0.490] | 0.665 [0.655, 0.675] | 0.683 [0.647, 0.719] | 0.723 [0.679, 0.768] (100%) | n/a | pass | pass | 0.35 | 0.35 | supported | yes |
| dims | 797 | 0.212 [0.195, 0.232] | 0.258 [0.238, 0.281] | 0.274 [0.245, 0.317] | 0.201 [0.187, 0.216] | 0.211 [0.194, 0.231] | 0.211 [0.194, 0.231] | 0.575 [0.530, 0.630] | 0.571 [0.526, 0.627] | 0.641 [0.507, 0.810] (20%) | n/a | pass | fail | 1.36 | -0.25 | premise fails (G1): the tower does not carry the measure | yes |
| speed | 171 | 2.058 [1.708, 2.508] | 2.022 [1.711, 2.418] | 1.967 [1.641, 2.390] | 2.035 [1.697, 2.480] | 2.029 [1.690, 2.487] | 2.039 [1.701, 2.492] | 2.388 [2.048, 2.815] | 2.379 [2.045, 2.802] | 1.801 [1.405, 2.258] (92%) | n/a | pass | fail | 2.54 | 0.65 | premise fails (G1): the tower does not carry the measure | no |
| heading | 76 | 87.147 [73.611, 100.563] | 75.999 [63.817, 88.685] | 83.801 [70.899, 96.809] | 79.044 [66.482, 91.668] | 87.030 [73.563, 100.447] | 84.094 [71.098, 96.695] | 92.433 [86.515, 98.308] | 95.217 [82.541, 106.943] | n/a | n/a | pass | fail | 0.30 | 0.73 | premise fails (G1): the tower does not carry the measure | yes |
| count | 400 | 2.910 [2.693, 3.116] | 2.754 [2.557, 2.949] | 2.756 [2.560, 2.951] | 1.532 [1.402, 1.683] | 2.821 [2.619, 3.015] | 2.852 [2.645, 3.051] | 3.275 [3.066, 3.480] | 3.280 [3.070, 3.486] | 2.195 [1.952, 2.460] (100%) | 2.072 [1.817, 2.338] | pass | fail | 0.99 | 8.84 | premise fails (G1): the tower does not carry the measure | no |

Exploratory (not pre-registered): spoken motion direction, 5 classes, n = 157 answered of 171: accuracy 0.204 [0.140, 0.268]; always 'still' (the training split's most common class) 0.529 [0.452, 0.605].
Spoken against the specialist (count, paired, n = 399): spoken 2.195 [1.952, 2.460], specialist 2.075 [1.822, 2.340], difference 0.120 [-0.103, 0.339].
Exploratory (not pre-registered): spoken count with off-form answers also parsed ({class: n}, {class: [{count}]}, a list of boxes): coverage 100%, error 2.195 [1.952, 2.460].
Exploratory (not pre-registered): spoken joint angles read as flexion (180 - answer): 110.419 [106.672, 113.993].

Per source (SpaceQwen2.5-VL-3B; the probes are fitted on all sources of a measure together):

| target | source | n | P | V | L_img | W |
|---|---|---|---|---|---|---|
| facing | h4d | 698 | 47.951 | 45.212 | 36.401 | 36.587 |
| facing | kitti_obj | 400 | 75.655 | 39.388 | 48.049 | 52.174 |
| facing | objectron | 157 | 53.165 | 51.282 | 55.786 | 52.303 |
| joints | h4d | 698 | 23.936 | 16.922 | 19.998 | 19.357 |
| normal | hypersim | 240 | 41.784 | 35.257 | 36.011 | 38.115 |
| normal | nyu | 156 | 37.541 | 32.454 | 33.771 | 31.079 |
| dims | hypersim | 240 | 0.418 | 0.421 | 0.449 | 0.422 |
| dims | kitti_obj | 400 | 0.109 | 0.198 | 0.211 | 0.099 |
| dims | objectron | 157 | 0.160 | 0.161 | 0.167 | 0.122 |
| speed | h4d_motion | 48 | 0.229 | 0.909 | 0.648 | 0.334 |
| speed | kitti_trk | 123 | 2.771 | 2.456 | 2.482 | 2.698 |
| heading | h4d_motion | 18 | 86.593 | 100.366 | 86.013 | 81.709 |
| heading | kitti_trk | 58 | 87.319 | 68.436 | 83.115 | 78.217 |
| count | coco | 400 | 2.910 | 2.754 | 2.756 | 1.532 |

## Qwen2.5-VL-3B

| target | n | P | V | L_img | W | W no image | W mismatched | shuffled | random | spoken (coverage) | specialist | G0 | G1 | R L_img | R W | outcome | attribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| facing | 1255 | 57.433 [54.786, 59.978] | 44.049 [41.712, 46.420] | 42.244 [40.022, 44.541] | 43.928 [41.393, 46.393] | 57.564 [54.904, 60.127] | 57.747 [55.093, 60.380] | 83.038 [80.463, 85.762] | 83.071 [80.495, 85.819] | 94.561 [91.793, 97.441] (98%) | n/a | pass | pass | 1.13 | 1.01 | falsified | yes |
| joints | 698 | 23.936 [23.119, 24.778] | 16.920 [15.866, 17.976] | 19.982 [18.926, 21.062] | 18.966 [17.932, 20.035] | 23.932 [23.123, 24.773] | 23.927 [23.122, 24.769] | 23.901 [23.094, 24.737] | 23.923 [23.113, 24.764] | 60.157 [58.890, 61.414] (98%) | n/a | pass | pass | 0.56 | 0.71 | partial | yes |
| normal | 396 | 40.113 [37.082, 43.192] | 34.157 [32.129, 36.175] | 34.524 [32.438, 36.615] | 34.067 [31.662, 36.601] | 40.136 [37.164, 43.130] | 40.355 [37.344, 43.273] | 52.770 [50.888, 54.743] | 52.939 [51.099, 54.872] | n/a | 9.270 [7.741, 10.924] | pass | fail | 0.94 | 1.02 | premise fails (G1): the tower does not carry the measure | yes |
| surface_class | 396 | 0.427 [0.385, 0.468] | 0.316 [0.271, 0.363] | 0.406 [0.364, 0.448] | 0.392 [0.350, 0.432] | 0.456 [0.425, 0.490] | 0.438 [0.395, 0.480] | 0.665 [0.655, 0.675] | 0.683 [0.647, 0.719] | 0.641 [0.614, 0.667] (100%) | n/a | pass | pass | 0.20 | 0.32 | supported | yes |
| dims | 797 | 0.212 [0.195, 0.232] | 0.258 [0.238, 0.281] | 0.273 [0.244, 0.316] | 0.200 [0.186, 0.216] | 0.210 [0.194, 0.230] | 0.211 [0.194, 0.231] | 0.575 [0.530, 0.630] | 0.571 [0.526, 0.627] | 13.373 [0.599, 30.032] (9%) | n/a | pass | fail | 1.34 | -0.27 | premise fails (G1): the tower does not carry the measure | yes |
| speed | 171 | 2.058 [1.708, 2.508] | 2.023 [1.711, 2.418] | 2.018 [1.701, 2.431] | 2.033 [1.697, 2.478] | 2.027 [1.688, 2.485] | 2.039 [1.701, 2.492] | 2.388 [2.048, 2.815] | 2.378 [2.045, 2.802] | 5.621 [4.680, 6.660] (87%) | n/a | pass | fail | 1.13 | 0.70 | premise fails (G1): the tower does not carry the measure | no |
| heading | 76 | 87.147 [73.611, 100.563] | 75.970 [63.800, 88.674] | 81.322 [68.393, 94.315] | 78.210 [65.841, 90.416] | 87.039 [73.567, 100.477] | 83.627 [70.179, 97.143] | 92.392 [86.489, 98.237] | 95.214 [82.546, 106.928] | n/a | n/a | pass | fail | 0.52 | 0.80 | premise fails (G1): the tower does not carry the measure | no |
| count | 400 | 2.910 [2.693, 3.116] | 2.754 [2.556, 2.949] | 2.767 [2.571, 2.960] | 1.526 [1.391, 1.673] | 2.824 [2.621, 3.018] | 2.855 [2.652, 3.050] | 3.275 [3.066, 3.480] | 3.280 [3.070, 3.486] | 2.307 [1.911, 2.900] (97%) | 2.072 [1.817, 2.338] | pass | fail | 0.92 | 8.86 | premise fails (G1): the tower does not carry the measure | no |

Exploratory (not pre-registered): spoken motion direction, 5 classes, n = 148 answered of 171: accuracy 0.128 [0.074, 0.182]; always 'still' (the training split's most common class) 0.541 [0.459, 0.622].
Spoken against the specialist (count, paired, n = 387): spoken 2.307 [1.911, 2.900], specialist 2.023 [1.766, 2.287], difference 0.284 [-0.073, 0.786].
Exploratory (not pre-registered): spoken count with off-form answers also parsed ({class: n}, {class: [{count}]}, a list of boxes): coverage 98%, error 2.298 [1.909, 2.876].
Exploratory (not pre-registered): spoken joint angles read as flexion (180 - answer): 69.644 [67.485, 71.837].

Per source (Qwen2.5-VL-3B; the probes are fitted on all sources of a measure together):

| target | source | n | P | V | L_img | W |
|---|---|---|---|---|---|---|
| facing | h4d | 698 | 47.951 | 45.151 | 35.932 | 36.538 |
| facing | kitti_obj | 400 | 75.655 | 39.272 | 47.989 | 52.860 |
| facing | objectron | 157 | 53.165 | 51.319 | 55.666 | 54.023 |
| joints | h4d | 698 | 23.936 | 16.920 | 19.982 | 18.966 |
| normal | hypersim | 240 | 41.784 | 35.256 | 35.204 | 37.414 |
| normal | nyu | 156 | 37.541 | 32.467 | 33.480 | 28.918 |
| dims | hypersim | 240 | 0.418 | 0.421 | 0.447 | 0.419 |
| dims | kitti_obj | 400 | 0.109 | 0.198 | 0.211 | 0.098 |
| dims | objectron | 157 | 0.160 | 0.162 | 0.165 | 0.123 |
| speed | h4d_motion | 48 | 0.229 | 0.908 | 0.771 | 0.331 |
| speed | kitti_trk | 123 | 2.771 | 2.458 | 2.505 | 2.698 |
| heading | h4d_motion | 18 | 86.593 | 100.375 | 83.951 | 80.209 |
| heading | kitti_trk | 58 | 87.319 | 68.396 | 80.506 | 77.590 |
| count | coco | 400 | 2.910 | 2.754 | 2.767 | 1.526 |

## Q3: SpatialAxiom-9B against Qwen3.5-9B, asked (paired)

| target | SpatialAxiom-9B | Qwen3.5-9B | difference | counts |
|---|---|---|---|---|
| facing | 72.121 [68.942, 75.029] | 82.039 [79.080, 85.089] | -9.918 [-13.891, -6.255] | True |
| joints | 103.960 [99.652, 108.302] | 30.159 [28.705, 31.657] | 73.801 [68.677, 79.055] | True |
| surface_class | 0.461 [0.409, 0.520] | 0.484 [0.445, 0.525] | -0.022 [-0.073, 0.029] | False |
| dims | 0.398 [0.366, 0.430] | 0.460 [0.357, 0.616] | -0.062 [-0.206, 0.041] | False |
| speed | 1.413 [1.001, 1.898] | 1.659 [1.206, 2.109] | -0.246 [-0.646, 0.186] | False |
| count | 2.421 [2.088, 2.789] | 2.100 [1.510, 2.973] | 0.322 [-0.487, 0.870] | False |
