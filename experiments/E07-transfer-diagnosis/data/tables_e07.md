# E07 tables (printed by src/analyse_e07.py from results_e07.json)

Depth. Errors: lower is better (abs-rel; scale-invariant abs-rel for A2 and its comparisons). Intervals: 95% bootstrap over the held-out images (2,000 resamples), paired. "helps" = the interval of R minus the arm is above 0.

## Primary: E03 layers

| model | held out | n | arm (layer, head) | error | spoken | MoGe-2 | R minus arm | vs R | spoken minus arm | arm / MoGe-2 | pass | control | control minus arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | indoor | 1086 | R, E03 decoder (32, ridge) | 0.454 [0.417, 0.493] | 0.288 [0.269, 0.306] | 0.101 [0.094, 0.108] | n/a |  | -0.167 [-0.201, -0.135] | 4.51 | fail |  | n/a |
| Qwen3.8-27B | indoor | 1086 | A1, scene type erased (32, mlp) | 0.620 [0.575, 0.669] | 0.288 [0.269, 0.306] | 0.101 [0.094, 0.108] | -0.165 [-0.192, -0.141] | hurts | -0.332 [-0.379, -0.290] | 6.15 | fail |  | n/a |
| Qwen3.8-27B | indoor | 1086 | A2, within the image (32, ridge) | 0.142 [0.133, 0.151] | 0.115 [0.106, 0.125] | 0.061 [0.054, 0.069] | 0.051 [0.038, 0.064] | helps | -0.027 [-0.037, -0.016] | 2.33 | fail | 2D cues 0.198 [0.183, 0.214] | 0.056 [0.041, 0.071] |
| Qwen3.8-27B | indoor | 1086 | A3, camera given (32, ridge) | 1.270 [1.183, 1.359] | 0.288 [0.269, 0.306] | 0.101 [0.094, 0.108] | -0.815 [-0.883, -0.750] | hurts | -0.982 [-1.068, -0.899] | 12.60 | fail | box + camera 0.973 [0.896, 1.051] | -0.297 [-0.356, -0.236] |
| Qwen3.8-27B | indoor | 1086 | R, scale-invariant | 0.193 [0.177, 0.210] | | | | | | | | | |
| Qwen3.8-27B | indoor_synth | 1183 | R, E03 decoder (36, mlp) | 0.291 [0.270, 0.315] | 0.231 [0.213, 0.252] | 0.178 [0.155, 0.209] | n/a |  | -0.060 [-0.075, -0.046] | 1.63 | fail |  | n/a |
| Qwen3.8-27B | indoor_synth | 1183 | A1, scene type erased (36, mlp) | 0.276 [0.255, 0.299] | 0.231 [0.213, 0.252] | 0.178 [0.155, 0.209] | 0.015 [0.005, 0.025] | helps | -0.045 [-0.060, -0.030] | 1.55 | fail |  | n/a |
| Qwen3.8-27B | indoor_synth | 1183 | A2, within the image (36, ridge) | 0.207 [0.186, 0.231] | 0.156 [0.141, 0.172] | 0.096 [0.079, 0.120] | -0.004 [-0.018, 0.010] | no difference | -0.051 [-0.067, -0.036] | 2.14 | fail | 2D cues 0.337 [0.301, 0.379] | 0.131 [0.105, 0.160] |
| Qwen3.8-27B | indoor_synth | 1183 | A3, camera given (36, ridge) | 0.676 [0.620, 0.741] | 0.231 [0.213, 0.252] | 0.178 [0.155, 0.209] | -0.384 [-0.439, -0.336] | hurts | -0.447 [-0.506, -0.397] | 3.79 | fail | box + camera 0.465 [0.437, 0.495] | -0.211 [-0.258, -0.167] |
| Qwen3.8-27B | indoor_synth | 1183 | R, scale-invariant | 0.203 [0.186, 0.222] | | | | | | | | | |
| Qwen3.8-27B | two_people | 1134 | R, E03 decoder (1, mlp) | 0.819 [0.795, 0.847] | 0.574 [0.546, 0.603] | 0.222 [0.208, 0.235] | n/a |  | -0.237 [-0.283, -0.198] | 3.70 | fail |  | n/a |
| Qwen3.8-27B | two_people | 1134 | A1, scene type erased (1, mlp) | 1.134 [1.095, 1.178] | 0.574 [0.546, 0.603] | 0.222 [0.208, 0.235] | -0.315 [-0.336, -0.294] | hurts | -0.540 [-0.608, -0.481] | 5.12 | fail |  | n/a |
| Qwen3.8-27B | two_people | 1134 | A2, within the image (1, ridge) | 0.200 [0.187, 0.213] | 0.051 [0.045, 0.057] | 0.079 [0.072, 0.087] | 0.803 [0.757, 0.847] | helps | -0.144 [-0.168, -0.122] | 2.52 | fail | 2D cues 0.094 [0.088, 0.099] | -0.106 [-0.114, -0.098] |
| Qwen3.8-27B | two_people | 1134 | A3, camera given (1, ridge) | 0.155 [0.149, 0.161] | 0.574 [0.546, 0.603] | 0.222 [0.208, 0.235] | 0.664 [0.638, 0.694] | helps | 0.411 [0.378, 0.445] | 0.70 | PASS | box + camera 0.266 [0.257, 0.276] | 0.111 [0.096, 0.126] |
| Qwen3.8-27B | two_people | 1134 | R, scale-invariant | 1.003 [0.966, 1.039] | | | | | | | | | |
| Qwen3.8-27B | street | 1277 | R, E03 decoder (40, mlp) | 0.802 [0.795, 0.808] | 0.199 [0.187, 0.212] | 0.102 [0.098, 0.107] | n/a |  | -0.601 [-0.616, -0.586] | 7.83 | fail |  | n/a |
| Qwen3.8-27B | street | 1277 | A1, scene type erased (40, mlp) | 0.782 [0.774, 0.790] | 0.199 [0.187, 0.212] | 0.102 [0.098, 0.107] | 0.020 [0.017, 0.023] | helps | -0.579 [-0.595, -0.562] | 7.64 | fail |  | n/a |
| Qwen3.8-27B | street | 1277 | A2, within the image (40, ridge) | 0.318 [0.298, 0.338] | 0.151 [0.133, 0.170] | 0.072 [0.067, 0.077] | 0.135 [0.116, 0.154] | helps | -0.162 [-0.190, -0.136] | 4.40 | fail | 2D cues 0.260 [0.244, 0.277] | -0.057 [-0.069, -0.047] |
| Qwen3.8-27B | street | 1277 | A3, camera given (40, ridge) | 0.669 [0.660, 0.677] | 0.199 [0.187, 0.212] | 0.102 [0.098, 0.107] | 0.133 [0.125, 0.140] | helps | -0.467 [-0.484, -0.450] | 6.53 | fail | box + camera 0.510 [0.499, 0.521] | -0.159 [-0.163, -0.154] |
| Qwen3.8-27B | street | 1277 | R, scale-invariant | 0.452 [0.425, 0.483] | | | | | | | | | |
| SpatialAxiom-9B | indoor | 1086 | R, E03 decoder (12, mlp) | 0.431 [0.399, 0.467] | 0.212 [0.197, 0.228] | 0.101 [0.094, 0.108] | n/a |  | -0.218 [-0.251, -0.189] | 4.28 | fail |  | n/a |
| SpatialAxiom-9B | indoor | 1086 | A1, scene type erased (12, mlp) | 0.620 [0.584, 0.658] | 0.212 [0.197, 0.228] | 0.101 [0.094, 0.108] | -0.189 [-0.208, -0.172] | hurts | -0.408 [-0.444, -0.374] | 6.15 | fail |  | n/a |
| SpatialAxiom-9B | indoor | 1086 | A2, within the image (12, ridge) | 0.138 [0.129, 0.149] | 0.101 [0.090, 0.112] | 0.061 [0.054, 0.069] | 0.050 [0.037, 0.064] | helps | -0.037 [-0.047, -0.027] | 2.26 | fail | 2D cues 0.198 [0.183, 0.214] | 0.060 [0.043, 0.076] |
| SpatialAxiom-9B | indoor | 1086 | A3, camera given (12, ridge) | 0.399 [0.367, 0.433] | 0.212 [0.197, 0.228] | 0.101 [0.094, 0.108] | 0.032 [0.006, 0.060] | helps | -0.185 [-0.220, -0.153] | 3.96 | fail | box + camera 0.973 [0.896, 1.051] | 0.573 [0.519, 0.630] |
| SpatialAxiom-9B | indoor | 1086 | R, scale-invariant | 0.188 [0.172, 0.205] | | | | | | | | | |
| SpatialAxiom-9B | indoor_synth | 1183 | R, E03 decoder (12, mlp) | 0.295 [0.275, 0.320] | 0.230 [0.210, 0.255] | 0.178 [0.155, 0.209] | n/a |  | -0.065 [-0.085, -0.045] | 1.66 | fail |  | n/a |
| SpatialAxiom-9B | indoor_synth | 1183 | A1, scene type erased (12, mlp) | 0.303 [0.278, 0.332] | 0.230 [0.210, 0.255] | 0.178 [0.155, 0.209] | -0.008 [-0.024, 0.007] | no difference | -0.083 [-0.114, -0.055] | 1.70 | fail |  | n/a |
| SpatialAxiom-9B | indoor_synth | 1183 | A2, within the image (12, ridge) | 0.173 [0.158, 0.191] | 0.141 [0.125, 0.159] | 0.096 [0.079, 0.120] | 0.035 [0.021, 0.048] | helps | -0.032 [-0.048, -0.017] | 1.80 | fail | 2D cues 0.337 [0.301, 0.379] | 0.164 [0.137, 0.195] |
| SpatialAxiom-9B | indoor_synth | 1183 | A3, camera given (12, ridge) | 0.334 [0.310, 0.359] | 0.230 [0.210, 0.255] | 0.178 [0.155, 0.209] | -0.039 [-0.061, -0.018] | hurts | -0.104 [-0.136, -0.077] | 1.87 | fail | box + camera 0.465 [0.437, 0.495] | 0.131 [0.111, 0.151] |
| SpatialAxiom-9B | indoor_synth | 1183 | R, scale-invariant | 0.208 [0.189, 0.230] | | | | | | | | | |
| SpatialAxiom-9B | two_people | 1134 | R, E03 decoder (23, mlp) | 0.276 [0.263, 0.292] | 0.221 [0.187, 0.264] | 0.222 [0.208, 0.235] | n/a |  | -0.066 [-0.098, -0.034] | 1.25 | fail |  | n/a |
| SpatialAxiom-9B | two_people | 1134 | A1, scene type erased (23, mlp) | 0.328 [0.307, 0.352] | 0.221 [0.187, 0.264] | 0.222 [0.208, 0.235] | -0.052 [-0.070, -0.035] | hurts | -0.106 [-0.134, -0.081] | 1.48 | fail |  | n/a |
| SpatialAxiom-9B | two_people | 1134 | A2, within the image (23, ridge) | 0.079 [0.072, 0.086] | 0.063 [0.048, 0.085] | 0.079 [0.072, 0.087] | 0.077 [0.067, 0.086] | helps | -0.015 [-0.027, -0.002] | 0.99 | fail | 2D cues 0.094 [0.088, 0.099] | 0.015 [0.007, 0.022] |
| SpatialAxiom-9B | two_people | 1134 | A3, camera given (23, mlp) | 0.444 [0.439, 0.449] | 0.221 [0.187, 0.264] | 0.222 [0.208, 0.235] | -0.167 [-0.182, -0.152] | hurts | -0.231 [-0.266, -0.187] | 2.00 | fail | box + camera 0.266 [0.257, 0.276] | -0.178 [-0.192, -0.164] |
| SpatialAxiom-9B | two_people | 1134 | R, scale-invariant | 0.156 [0.147, 0.165] | | | | | | | | | |
| SpatialAxiom-9B | street | 1277 | R, E03 decoder (18, mlp) | 0.826 [0.821, 0.831] | 0.261 [0.240, 0.282] | 0.102 [0.098, 0.107] | n/a |  | -0.564 [-0.587, -0.542] | 8.07 | fail |  | n/a |
| SpatialAxiom-9B | street | 1277 | A1, scene type erased (18, ridge) | 0.853 [0.848, 0.858] | 0.261 [0.240, 0.282] | 0.102 [0.098, 0.107] | -0.027 [-0.028, -0.025] | hurts | -0.589 [-0.613, -0.567] | 8.33 | fail |  | n/a |
| SpatialAxiom-9B | street | 1277 | A2, within the image (18, ridge) | 0.269 [0.251, 0.288] | 0.201 [0.178, 0.228] | 0.072 [0.067, 0.077] | 0.060 [0.049, 0.071] | helps | -0.065 [-0.093, -0.038] | 3.73 | fail | 2D cues 0.260 [0.244, 0.277] | -0.009 [-0.021, 0.003] |
| SpatialAxiom-9B | street | 1277 | A3, camera given (18, mlp) | 0.669 [0.659, 0.679] | 0.261 [0.240, 0.282] | 0.102 [0.098, 0.107] | 0.157 [0.148, 0.166] | helps | -0.405 [-0.432, -0.378] | 6.54 | fail | box + camera 0.510 [0.499, 0.521] | -0.159 [-0.165, -0.153] |
| SpatialAxiom-9B | street | 1277 | R, scale-invariant | 0.329 [0.308, 0.352] | | | | | | | | | |

**Reading** (pre-registered; 8 folds):

- A1, scene type erased (H1, scene-type shortcut): helps in 2, hurts in 5 -> **mixed**; passes E03's bar where R fails: none; helps and beats its control: 2
- A2, within the image (H2, absolute scale (relative geometry transfers)): helps in 7, hurts in 0 -> **supported**; passes E03's bar where R fails: none; helps and beats its control: 4
- A3, camera given (H3, missing camera): helps in 4, hurts in 4 -> **mixed**; passes E03's bar where R fails: [('Qwen3.8-27B', 'two_people')]; helps and beats its control: 2

## Exploratory: re-chosen layers

| model | held out | n | arm (layer, head) | error | spoken | MoGe-2 | R minus arm | vs R | spoken minus arm | arm / MoGe-2 | pass | control | control minus arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | indoor | 1086 | R, E03 decoder (32, ridge) | 0.454 [0.417, 0.493] | 0.288 [0.269, 0.306] | 0.101 [0.094, 0.108] | n/a |  | -0.167 [-0.201, -0.135] | 4.51 | fail |  | n/a |
| Qwen3.8-27B | indoor | 1086 | A1, scene type erased (32, mlp) | 0.620 [0.575, 0.669] | 0.288 [0.269, 0.306] | 0.101 [0.094, 0.108] | -0.165 [-0.192, -0.141] | hurts | -0.332 [-0.379, -0.290] | 6.15 | fail |  | n/a |
| Qwen3.8-27B | indoor | 1086 | A2, within the image (32, ridge) | 0.142 [0.133, 0.151] | 0.115 [0.106, 0.125] | 0.061 [0.054, 0.069] | 0.051 [0.038, 0.064] | helps | -0.027 [-0.037, -0.016] | 2.33 | fail | 2D cues 0.198 [0.183, 0.214] | 0.056 [0.041, 0.071] |
| Qwen3.8-27B | indoor | 1086 | A3, camera given (5, ridge) | 0.983 [0.912, 1.055] | 0.288 [0.269, 0.306] | 0.101 [0.094, 0.108] | -0.528 [-0.578, -0.479] | hurts | -0.695 [-0.762, -0.631] | 9.75 | fail | box + camera 0.973 [0.896, 1.051] | -0.010 [-0.051, 0.030] |
| Qwen3.8-27B | indoor | 1086 | R, scale-invariant | 0.193 [0.177, 0.210] | | | | | | | | | |
| Qwen3.8-27B | indoor_synth | 1183 | R, E03 decoder (36, mlp) | 0.291 [0.270, 0.315] | 0.231 [0.213, 0.252] | 0.178 [0.155, 0.209] | n/a |  | -0.060 [-0.075, -0.046] | 1.63 | fail |  | n/a |
| Qwen3.8-27B | indoor_synth | 1183 | A1, scene type erased (36, mlp) | 0.276 [0.255, 0.299] | 0.231 [0.213, 0.252] | 0.178 [0.155, 0.209] | 0.015 [0.005, 0.025] | helps | -0.045 [-0.060, -0.030] | 1.55 | fail |  | n/a |
| Qwen3.8-27B | indoor_synth | 1183 | A2, within the image (53, ridge) | 0.191 [0.174, 0.211] | 0.156 [0.141, 0.172] | 0.096 [0.079, 0.120] | 0.011 [-0.001, 0.023] | no difference | -0.036 [-0.049, -0.024] | 1.99 | fail | 2D cues 0.337 [0.301, 0.379] | 0.146 [0.117, 0.179] |
| Qwen3.8-27B | indoor_synth | 1183 | A3, camera given (5, ridge) | 0.511 [0.477, 0.549] | 0.231 [0.213, 0.252] | 0.178 [0.155, 0.209] | -0.220 [-0.249, -0.192] | hurts | -0.281 [-0.314, -0.251] | 2.87 | fail | box + camera 0.465 [0.437, 0.495] | -0.046 [-0.064, -0.029] |
| Qwen3.8-27B | indoor_synth | 1183 | R, scale-invariant | 0.203 [0.186, 0.222] | | | | | | | | | |
| Qwen3.8-27B | two_people | 1134 | R, E03 decoder (1, mlp) | 0.819 [0.795, 0.847] | 0.574 [0.546, 0.603] | 0.222 [0.208, 0.235] | n/a |  | -0.237 [-0.283, -0.198] | 3.70 | fail |  | n/a |
| Qwen3.8-27B | two_people | 1134 | A1, scene type erased (33, mlp) | 1.108 [1.053, 1.166] | 0.574 [0.546, 0.603] | 0.222 [0.208, 0.235] | -0.289 [-0.339, -0.238] | hurts | -0.495 [-0.586, -0.409] | 5.00 | fail |  | n/a |
| Qwen3.8-27B | two_people | 1134 | A2, within the image (49, ridge) | 0.059 [0.055, 0.064] | 0.051 [0.045, 0.057] | 0.079 [0.072, 0.087] | 0.944 [0.909, 0.979] | helps | -0.004 [-0.013, 0.003] | 0.74 | fail | 2D cues 0.094 [0.088, 0.099] | 0.035 [0.029, 0.040] |
| Qwen3.8-27B | two_people | 1134 | A3, camera given (61, mlp) | 0.207 [0.196, 0.217] | 0.574 [0.546, 0.603] | 0.222 [0.208, 0.235] | 0.613 [0.588, 0.640] | helps | 0.375 [0.345, 0.406] | 0.93 | PASS | box + camera 0.266 [0.257, 0.276] | 0.059 [0.046, 0.072] |
| Qwen3.8-27B | two_people | 1134 | R, scale-invariant | 1.003 [0.966, 1.039] | | | | | | | | | |
| Qwen3.8-27B | street | 1277 | R, E03 decoder (40, mlp) | 0.802 [0.795, 0.808] | 0.199 [0.187, 0.212] | 0.102 [0.098, 0.107] | n/a |  | -0.601 [-0.616, -0.586] | 7.83 | fail |  | n/a |
| Qwen3.8-27B | street | 1277 | A1, scene type erased (40, mlp) | 0.782 [0.774, 0.790] | 0.199 [0.187, 0.212] | 0.102 [0.098, 0.107] | 0.020 [0.017, 0.023] | helps | -0.579 [-0.595, -0.562] | 7.64 | fail |  | n/a |
| Qwen3.8-27B | street | 1277 | A2, within the image (45, ridge) | 0.264 [0.248, 0.281] | 0.151 [0.133, 0.170] | 0.072 [0.067, 0.077] | 0.188 [0.167, 0.211] | helps | -0.111 [-0.134, -0.088] | 3.66 | fail | 2D cues 0.260 [0.244, 0.277] | -0.004 [-0.017, 0.008] |
| Qwen3.8-27B | street | 1277 | A3, camera given (37, ridge) | 0.688 [0.681, 0.696] | 0.199 [0.187, 0.212] | 0.102 [0.098, 0.107] | 0.113 [0.106, 0.120] | helps | -0.486 [-0.503, -0.469] | 6.73 | fail | box + camera 0.510 [0.499, 0.521] | -0.178 [-0.183, -0.173] |
| Qwen3.8-27B | street | 1277 | R, scale-invariant | 0.452 [0.425, 0.483] | | | | | | | | | |
| SpatialAxiom-9B | indoor | 1086 | R, E03 decoder (12, mlp) | 0.431 [0.399, 0.467] | 0.212 [0.197, 0.228] | 0.101 [0.094, 0.108] | n/a |  | -0.218 [-0.251, -0.189] | 4.28 | fail |  | n/a |
| SpatialAxiom-9B | indoor | 1086 | A1, scene type erased (17, mlp) | 0.608 [0.571, 0.650] | 0.212 [0.197, 0.228] | 0.101 [0.094, 0.108] | -0.177 [-0.200, -0.154] | hurts | -0.398 [-0.433, -0.364] | 6.03 | fail |  | n/a |
| SpatialAxiom-9B | indoor | 1086 | A2, within the image (17, ridge) | 0.130 [0.120, 0.141] | 0.101 [0.090, 0.112] | 0.061 [0.054, 0.069] | 0.058 [0.044, 0.074] | helps | -0.030 [-0.039, -0.021] | 2.13 | fail | 2D cues 0.198 [0.183, 0.214] | 0.068 [0.052, 0.085] |
| SpatialAxiom-9B | indoor | 1086 | A3, camera given (29, ridge) | 0.946 [0.877, 1.017] | 0.212 [0.197, 0.228] | 0.101 [0.094, 0.108] | -0.515 [-0.571, -0.459] | hurts | -0.735 [-0.805, -0.668] | 9.39 | fail | box + camera 0.973 [0.896, 1.051] | 0.026 [-0.016, 0.066] |
| SpatialAxiom-9B | indoor | 1086 | R, scale-invariant | 0.188 [0.172, 0.205] | | | | | | | | | |
| SpatialAxiom-9B | indoor_synth | 1183 | R, E03 decoder (12, mlp) | 0.295 [0.275, 0.320] | 0.230 [0.210, 0.255] | 0.178 [0.155, 0.209] | n/a |  | -0.065 [-0.085, -0.045] | 1.66 | fail |  | n/a |
| SpatialAxiom-9B | indoor_synth | 1183 | A1, scene type erased (17, mlp) | 0.300 [0.277, 0.327] | 0.230 [0.210, 0.255] | 0.178 [0.155, 0.209] | -0.005 [-0.023, 0.013] | no difference | -0.089 [-0.117, -0.065] | 1.68 | fail |  | n/a |
| SpatialAxiom-9B | indoor_synth | 1183 | A2, within the image (17, ridge) | 0.191 [0.173, 0.211] | 0.141 [0.125, 0.159] | 0.096 [0.079, 0.120] | 0.017 [0.002, 0.032] | helps | -0.034 [-0.049, -0.020] | 1.98 | fail | 2D cues 0.337 [0.301, 0.379] | 0.146 [0.119, 0.177] |
| SpatialAxiom-9B | indoor_synth | 1183 | A3, camera given (12, ridge) | 0.334 [0.310, 0.359] | 0.230 [0.210, 0.255] | 0.178 [0.155, 0.209] | -0.039 [-0.061, -0.018] | hurts | -0.104 [-0.136, -0.077] | 1.87 | fail | box + camera 0.465 [0.437, 0.495] | 0.131 [0.111, 0.151] |
| SpatialAxiom-9B | indoor_synth | 1183 | R, scale-invariant | 0.208 [0.189, 0.230] | | | | | | | | | |
| SpatialAxiom-9B | two_people | 1134 | R, E03 decoder (23, mlp) | 0.276 [0.263, 0.292] | 0.221 [0.187, 0.264] | 0.222 [0.208, 0.235] | n/a |  | -0.066 [-0.098, -0.034] | 1.25 | fail |  | n/a |
| SpatialAxiom-9B | two_people | 1134 | A1, scene type erased (17, mlp) | 0.824 [0.783, 0.868] | 0.221 [0.187, 0.264] | 0.222 [0.208, 0.235] | -0.547 [-0.591, -0.507] | hurts | -0.569 [-0.631, -0.509] | 3.72 | fail |  | n/a |
| SpatialAxiom-9B | two_people | 1134 | A2, within the image (17, ridge) | 0.109 [0.102, 0.116] | 0.063 [0.048, 0.085] | 0.079 [0.072, 0.087] | 0.047 [0.041, 0.053] | helps | -0.045 [-0.060, -0.028] | 1.37 | fail | 2D cues 0.094 [0.088, 0.099] | -0.015 [-0.024, -0.006] |
| SpatialAxiom-9B | two_people | 1134 | A3, camera given (23, mlp) | 0.444 [0.439, 0.449] | 0.221 [0.187, 0.264] | 0.222 [0.208, 0.235] | -0.167 [-0.182, -0.152] | hurts | -0.231 [-0.266, -0.187] | 2.00 | fail | box + camera 0.266 [0.257, 0.276] | -0.178 [-0.192, -0.164] |
| SpatialAxiom-9B | two_people | 1134 | R, scale-invariant | 0.156 [0.147, 0.165] | | | | | | | | | |
| SpatialAxiom-9B | street | 1277 | R, E03 decoder (18, mlp) | 0.826 [0.821, 0.831] | 0.261 [0.240, 0.282] | 0.102 [0.098, 0.107] | n/a |  | -0.564 [-0.587, -0.542] | 8.07 | fail |  | n/a |
| SpatialAxiom-9B | street | 1277 | A1, scene type erased (17, ridge) | 0.849 [0.843, 0.854] | 0.261 [0.240, 0.282] | 0.102 [0.098, 0.107] | -0.022 [-0.024, -0.021] | hurts | -0.585 [-0.609, -0.563] | 8.29 | fail |  | n/a |
| SpatialAxiom-9B | street | 1277 | A2, within the image (17, ridge) | 0.268 [0.250, 0.287] | 0.201 [0.178, 0.228] | 0.072 [0.067, 0.077] | 0.061 [0.049, 0.072] | helps | -0.065 [-0.093, -0.037] | 3.72 | fail | 2D cues 0.260 [0.244, 0.277] | -0.008 [-0.020, 0.004] |
| SpatialAxiom-9B | street | 1277 | A3, camera given (21, mlp) | 0.577 [0.566, 0.588] | 0.261 [0.240, 0.282] | 0.102 [0.098, 0.107] | 0.249 [0.240, 0.258] | helps | -0.315 [-0.342, -0.287] | 5.64 | fail | box + camera 0.510 [0.499, 0.521] | -0.067 [-0.073, -0.061] |
| SpatialAxiom-9B | street | 1277 | R, scale-invariant | 0.329 [0.308, 0.352] | | | | | | | | | |

**Reading** (pre-registered; 8 folds):

- A1, scene type erased (H1, scene-type shortcut): helps in 2, hurts in 5 -> **mixed**; passes E03's bar where R fails: none; helps and beats its control: 2
- A2, within the image (H2, absolute scale (relative geometry transfers)): helps in 7, hurts in 0 -> **supported**; passes E03's bar where R fails: none; helps and beats its control: 4
- A3, camera given (H3, missing camera): helps in 3, hurts in 5 -> **mixed**; passes E03's bar where R fails: [('Qwen3.8-27B', 'two_people')]; helps and beats its control: 1

## Checks

| model | held out | R refit | E03 published | within 5% | scene type named, before erasure | after | majority share |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | indoor | 0.4544 | 0.4544 | True | 1.000 | 0.355 | 0.355 |
| Qwen3.8-27B | indoor_synth | 0.2913 | 0.2842 | True | 1.000 | 0.365 | 0.365 |
| Qwen3.8-27B | two_people | 0.8194 | 0.7433 | False | 0.843 | 0.360 | 0.360 |
| Qwen3.8-27B | street | 0.8016 | 0.8022 | True | 1.000 | 0.348 | 0.348 |
| SpatialAxiom-9B | indoor | 0.4314 | 0.4498 | True | 1.000 | 0.355 | 0.355 |
| SpatialAxiom-9B | indoor_synth | 0.2951 | 0.2959 | True | 1.000 | 0.365 | 0.365 |
| SpatialAxiom-9B | two_people | 0.2765 | 0.2778 | True | 0.999 | 0.360 | 0.360 |
| SpatialAxiom-9B | street | 0.8263 | 0.8238 | True | 1.000 | 0.348 | 0.348 |
