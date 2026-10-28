
## Qwen3.8-27B


### Probes, all test frames (pre-registered)


**depth** (abs-rel, lower is better; 480 test objects in 195 images; layers chosen on training data: V 13, L_img 48, L_img_any 36, W 36, W_noimg 28, W_mis 11)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.381 [0.343, 0.420] |  |
| vision tower | 0.161 [0.144, 0.179] |  |
| image tokens, last quarter of language layers | 0.177 [0.157, 0.197] | 0.93 [0.88, 0.98] |
| image tokens, any language layer | 0.164 [0.146, 0.183] | 0.98 [0.94, 1.03] |
| object word, with the image | 0.169 [0.152, 0.185] | 0.96 [0.91, 1.02] |
| object word, no image | 0.360 [0.326, 0.396] | 0.09 [0.05, 0.14] |
| object word, mismatched image | 0.369 [0.334, 0.406] | 0.05 [0.01, 0.09] |
| control: shuffled labels (tower layer) | 0.480 [0.433, 0.530] |  |
| control: norm-matched random features | 0.478 [0.432, 0.528] |  |

δ<1.25 (higher is better): name prior 0.404 [0.358, 0.449]; vision tower 0.775 [0.730, 0.820]; image tokens, last quarter of language layers 0.719 [0.672, 0.771]; image tokens, any language layer 0.744 [0.700, 0.791]; object word, with the image 0.762 [0.722, 0.801]; object word, no image 0.400 [0.354, 0.443]; object word, mismatched image 0.398 [0.350, 0.443]

G0 control: shuffled labels (tower layer): P minus control -0.099 [-0.136, -0.061] -> pass; control: norm-matched random features: P minus control -0.098 [-0.134, -0.060] -> pass. G1: tower gain over the prior 57.7% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 755 test objects in 310 images; layers chosen on training data: V 5, L_img 48, L_img_any 47, W 44, W_noimg 48, W_mis 27)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.22 [15.36, 17.09] |  |
| vision tower | 3.91 [3.72, 4.11] |  |
| image tokens, last quarter of language layers | 3.62 [3.45, 3.79] | 1.02 [1.01, 1.04] |
| image tokens, any language layer | 3.54 [3.39, 3.71] | 1.03 [1.01, 1.05] |
| object word, with the image | 6.98 [6.56, 7.42] | 0.75 [0.72, 0.78] |
| object word, no image | 16.09 [15.25, 16.94] | 0.01 [-0.01, 0.03] |
| object word, mismatched image | 16.17 [15.35, 16.99] | 0.00 [-0.01, 0.02] |
| control: shuffled labels (tower layer) | 18.11 [17.29, 18.96] |  |
| control: norm-matched random features | 18.12 [17.29, 18.96] |  |

G0 control: shuffled labels (tower layer): P minus control -1.893 [-2.280, -1.470] -> pass; control: norm-matched random features: P minus control -1.896 [-2.283, -1.474] -> pass. G1: tower gain over the prior 75.9% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 21, L_img 48, L_img_any 36, W 40, W_noimg 38, W_mis 9)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.360 [0.313, 0.409] |  |
| image tokens, last quarter of language layers | 0.362 [0.319, 0.410] | 0.97 [0.70, 1.26] |
| image tokens, any language layer | 0.341 [0.301, 0.382] | 1.19 [0.97, 1.67] |
| object word, with the image | 0.314 [0.273, 0.358] | 1.46 [1.11, 2.60] |
| object word, no image | 0.424 [0.368, 0.482] | 0.36 [0.01, 0.67] |
| object word, mismatched image | 0.426 [0.372, 0.484] | 0.33 [-0.02, 0.62] |
| control: shuffled labels (tower layer) | 0.647 [0.560, 0.741] |  |
| control: norm-matched random features | 0.645 [0.558, 0.739] |  |

G0 control: shuffled labels (tower layer): P minus control -0.188 [-0.260, -0.107] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 21.6% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**centre** (3D error, m, lower is better; 480 test objects in 195 images; layers chosen on training data: V 13, L_img 48, L_img_any 36, W 36, W_noimg 37, W_mis 15)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.354 [1.281, 1.437] |  |
| vision tower | 0.649 [0.607, 0.692] |  |
| image tokens, last quarter of language layers | 0.603 [0.560, 0.647] | 1.06 [1.02, 1.11] |
| image tokens, any language layer | 0.581 [0.543, 0.622] | 1.10 [1.05, 1.14] |
| object word, with the image | 0.711 [0.662, 0.763] | 0.91 [0.86, 0.96] |
| object word, no image | 1.335 [1.262, 1.419] | 0.03 [0.00, 0.05] |
| object word, mismatched image | 1.340 [1.267, 1.423] | 0.02 [-0.00, 0.04] |
| control: shuffled labels (tower layer) | 1.647 [1.566, 1.734] |  |
| control: norm-matched random features | 1.647 [1.566, 1.734] |  |

G0 control: shuffled labels (tower layer): P minus control -0.293 [-0.353, -0.233] -> pass; control: norm-matched random features: P minus control -0.293 [-0.353, -0.233] -> pass. G1: tower gain over the prior 52.1% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.92 [0.88, 0.95] (n=272) | 0.99 [0.98, 1.00] (n=399) | 0.98 [0.96, 1.00] (n=381) | 0.85 [0.77, 0.92] (n=176) | 0.96 [0.94, 0.97] (n=1228) |
| image tokens, last quarter of language layers | 0.90 [0.86, 0.93] (n=272) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.82 [0.74, 0.89] (n=176) | 0.95 [0.93, 0.96] (n=1228) |
| object word, with the image | 0.89 [0.85, 0.92] (n=272) | 0.94 [0.91, 0.97] (n=399) | 0.97 [0.95, 0.99] (n=381) | 0.83 [0.75, 0.90] (n=176) | 0.93 [0.91, 0.94] (n=1228) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.636 [0.560, 0.725] (n=375); vision tower 0.413 [0.349, 0.486] (n=375); image tokens, last quarter of language layers 0.399 [0.337, 0.470] (n=375); object word, with the image 0.437 [0.368, 0.519] (n=375)

### Probes, without MMA and grappling frames


**depth** (abs-rel, lower is better; 452 test objects in 181 images; layers chosen on training data: V 13, L_img 48, L_img_any 36, W 36, W_noimg 28, W_mis 11)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.394 [0.355, 0.434] |  |
| vision tower | 0.164 [0.147, 0.183] |  |
| image tokens, last quarter of language layers | 0.183 [0.162, 0.204] | 0.92 [0.87, 0.97] |
| image tokens, any language layer | 0.168 [0.149, 0.188] | 0.98 [0.94, 1.03] |
| object word, with the image | 0.172 [0.154, 0.189] | 0.97 [0.92, 1.02] |
| object word, no image | 0.372 [0.336, 0.409] | 0.10 [0.05, 0.14] |
| object word, mismatched image | 0.382 [0.345, 0.421] | 0.05 [0.01, 0.09] |
| control: shuffled labels (tower layer) | 0.489 [0.439, 0.543] |  |
| control: norm-matched random features | 0.487 [0.439, 0.540] |  |

δ<1.25 (higher is better): name prior 0.385 [0.339, 0.430]; vision tower 0.765 [0.718, 0.812]; image tokens, last quarter of language layers 0.704 [0.653, 0.756]; image tokens, any language layer 0.737 [0.690, 0.785]; object word, with the image 0.754 [0.712, 0.794]; object word, no image 0.381 [0.334, 0.425]; object word, mismatched image 0.378 [0.330, 0.425]

G0 control: shuffled labels (tower layer): P minus control -0.095 [-0.134, -0.056] -> pass; control: norm-matched random features: P minus control -0.094 [-0.131, -0.055] -> pass. G1: tower gain over the prior 58.3% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 727 test objects in 296 images; layers chosen on training data: V 5, L_img 48, L_img_any 47, W 44, W_noimg 48, W_mis 27)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.40 [15.50, 17.28] |  |
| vision tower | 3.98 [3.79, 4.18] |  |
| image tokens, last quarter of language layers | 3.65 [3.48, 3.82] | 1.03 [1.01, 1.04] |
| image tokens, any language layer | 3.58 [3.41, 3.75] | 1.03 [1.02, 1.05] |
| object word, with the image | 7.06 [6.63, 7.53] | 0.75 [0.72, 0.78] |
| object word, no image | 16.26 [15.40, 17.12] | 0.01 [-0.01, 0.03] |
| object word, mismatched image | 16.35 [15.50, 17.19] | 0.00 [-0.01, 0.02] |
| control: shuffled labels (tower layer) | 18.32 [17.47, 19.19] |  |
| control: norm-matched random features | 18.32 [17.47, 19.19] |  |

G0 control: shuffled labels (tower layer): P minus control -1.921 [-2.329, -1.486] -> pass; control: norm-matched random features: P minus control -1.924 [-2.329, -1.489] -> pass. G1: tower gain over the prior 75.7% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 21, L_img 48, L_img_any 36, W 40, W_noimg 38, W_mis 9)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.360 [0.313, 0.409] |  |
| image tokens, last quarter of language layers | 0.362 [0.319, 0.410] | 0.97 [0.70, 1.26] |
| image tokens, any language layer | 0.341 [0.301, 0.382] | 1.19 [0.97, 1.67] |
| object word, with the image | 0.314 [0.273, 0.358] | 1.46 [1.11, 2.60] |
| object word, no image | 0.424 [0.368, 0.482] | 0.36 [0.01, 0.67] |
| object word, mismatched image | 0.426 [0.372, 0.484] | 0.33 [-0.02, 0.62] |
| control: shuffled labels (tower layer) | 0.647 [0.560, 0.741] |  |
| control: norm-matched random features | 0.645 [0.558, 0.739] |  |

G0 control: shuffled labels (tower layer): P minus control -0.188 [-0.260, -0.107] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 21.6% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**centre** (3D error, m, lower is better; 452 test objects in 181 images; layers chosen on training data: V 13, L_img 48, L_img_any 36, W 36, W_noimg 37, W_mis 15)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.355 [1.278, 1.441] |  |
| vision tower | 0.648 [0.604, 0.695] |  |
| image tokens, last quarter of language layers | 0.608 [0.563, 0.654] | 1.06 [1.01, 1.10] |
| image tokens, any language layer | 0.579 [0.540, 0.623] | 1.10 [1.05, 1.14] |
| object word, with the image | 0.710 [0.659, 0.767] | 0.91 [0.86, 0.97] |
| object word, no image | 1.335 [1.256, 1.421] | 0.03 [0.00, 0.05] |
| object word, mismatched image | 1.340 [1.263, 1.425] | 0.02 [-0.00, 0.05] |
| control: shuffled labels (tower layer) | 1.645 [1.562, 1.733] |  |
| control: norm-matched random features | 1.645 [1.562, 1.733] |  |

G0 control: shuffled labels (tower layer): P minus control -0.290 [-0.349, -0.230] -> pass; control: norm-matched random features: P minus control -0.290 [-0.349, -0.230] -> pass. G1: tower gain over the prior 52.2% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.92 [0.88, 0.95] (n=266) | 0.99 [0.98, 1.00] (n=399) | 0.98 [0.96, 1.00] (n=381) | 0.85 [0.77, 0.92] (n=176) | 0.95 [0.94, 0.97] (n=1222) |
| image tokens, last quarter of language layers | 0.90 [0.86, 0.93] (n=266) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.82 [0.74, 0.89] (n=176) | 0.95 [0.93, 0.96] (n=1222) |
| object word, with the image | 0.89 [0.85, 0.93] (n=266) | 0.94 [0.91, 0.97] (n=399) | 0.97 [0.95, 0.99] (n=381) | 0.83 [0.75, 0.90] (n=176) | 0.93 [0.91, 0.94] (n=1222) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.625 [0.550, 0.709] (n=361); vision tower 0.406 [0.341, 0.477] (n=361); image tokens, last quarter of language layers 0.400 [0.335, 0.473] (n=361); object word, with the image 0.430 [0.360, 0.513] (n=361)

### Asked answers, all frames (same objects for every column)

Answers: 1385 images, 1385 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 100% of 480 | 0.377 [0.346, 0.410] | 0.381 [0.343, 0.420] | 0.161 [0.144, 0.179] | 0.177 [0.157, 0.197] | 0.169 [0.152, 0.185] | MoGe-2 0.146 [0.127, 0.165] |
| depth_heldout | 100% of 856 | 0.199 [0.186, 0.211] | 0.897 [0.891, 0.902] | 0.695 [0.685, 0.704] | 0.783 [0.776, 0.790] | 0.858 [0.852, 0.864] | MoGe-2 0.103 [0.098, 0.108] |
| depth_test_indoor | 100% of 338 | 0.299 [0.266, 0.330] | 0.372 [0.332, 0.418] | 0.184 [0.163, 0.207] | 0.204 [0.180, 0.231] | 0.186 [0.166, 0.208] | MoGe-2 0.101 [0.088, 0.114] |
| depth_test_two_people | 100% of 142 | 0.565 [0.512, 0.622] | 0.402 [0.334, 0.477] | 0.107 [0.087, 0.127] | 0.112 [0.093, 0.134] | 0.128 [0.109, 0.147] | MoGe-2 0.253 [0.203, 0.300] |
| xy_test | 100% of 755 | 166.98 [140.17, 195.17] | 16.22 [15.36, 17.09] | 3.91 [3.72, 4.11] | 3.62 [3.45, 3.79] | 6.98 [6.56, 7.42] | RF-DETR 1.38 [1.01, 1.84] |
| xy_heldout | 100% of 856 | 4.23 [3.46, 5.01] | 18.43 [17.73, 19.13] | 6.66 [6.34, 6.97] | 8.49 [8.15, 8.83] | 9.65 [9.14, 10.16] |  |
| xy_test_people | 100% of 275 | 170.55 [127.23, 214.81] | 13.59 [12.52, 14.66] | 4.19 [3.84, 4.55] | 3.40 [3.14, 3.71] | 6.49 [5.99, 7.01] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 100% of 338 | 0.385 [0.343, 0.430] | 0.459 [0.388, 0.541] | 0.360 [0.313, 0.409] | 0.362 [0.319, 0.410] | 0.314 [0.273, 0.358] |  |
| size_heldout | 100% of 856 | 0.158 [0.146, 0.170] | 0.680 [0.667, 0.693] | 0.664 [0.648, 0.680] | 0.636 [0.618, 0.653] | 0.683 [0.669, 0.699] |  |
| distance_test | 100% of 375 | 0.370 [0.324, 0.425] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 99% of 512 | 0.423 [0.392, 0.453] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked 0.003 [-0.031, 0.039]; asked beats the prior by 1% -> qualitative (pre-registered rule); asked δ<1.25 0.371 [0.317, 0.427]; median asked/true 1.32
- depth_heldout: prior minus asked 0.698 [0.684, 0.711]; asked beats the prior by 78% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.676 [0.642, 0.706]; median asked/true 1.00
- depth_test_indoor: prior minus asked 0.073 [0.033, 0.114]; asked beats the prior by 20% -> qualitative (pre-registered rule); asked δ<1.25 0.497 [0.432, 0.565]; median asked/true 1.22
- depth_test_two_people: prior minus asked -0.163 [-0.214, -0.113]; asked beats the prior by -40% -> qualitative (pre-registered rule); asked δ<1.25 0.070 [0.027, 0.125]; median asked/true 1.52
- xy_test: answers above 100: 34%; median error 2.98; read as 0-1000 (exploratory) 3.14 [2.66, 3.66]; native boxes 2.73 [2.25, 3.26] (n=731, 97% returned); prior minus asked -150.762 [-178.952, -124.534]; asked beats the prior by -930% -> qualitative (pre-registered rule)
- xy_heldout: answers above 100: 0%; median error 0.70; read as 0-1000 (exploratory) 4.23 [3.46, 5.01]; native boxes 4.10 [3.31, 4.91] (n=805, 94% returned); prior minus asked 14.203 [13.252, 15.170]; asked beats the prior by 77% -> a measurement (>= 20%, CI excludes 0)
- xy_test_people: answers above 100: 36%; median error 2.08; read as 0-1000 (exploratory) 1.68 [1.39, 1.96]; native boxes 1.14 [0.88, 1.42] (n=253, 92% returned); prior minus asked -156.953 [-201.041, -113.831]; asked beats the prior by -1155% -> qualitative (pre-registered rule)
- size_test: prior minus asked 0.074 [0.008, 0.155]; asked beats the prior by 16% -> qualitative (pre-registered rule)
- size_heldout: prior minus asked 0.522 [0.510, 0.535]; asked beats the prior by 77% -> a measurement (>= 20%, CI excludes 0)

### Asked answers, without MMA and grappling (same objects for every column)

Answers: 1385 images, 1385 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 100% of 452 | 0.370 [0.337, 0.404] | 0.394 [0.355, 0.434] | 0.164 [0.147, 0.183] | 0.183 [0.162, 0.204] | 0.172 [0.154, 0.189] | MoGe-2 0.138 [0.119, 0.156] |
| depth_heldout | 100% of 856 | 0.199 [0.186, 0.211] | 0.897 [0.891, 0.902] | 0.695 [0.685, 0.704] | 0.783 [0.776, 0.790] | 0.858 [0.852, 0.864] | MoGe-2 0.103 [0.098, 0.108] |
| depth_test_indoor | 100% of 338 | 0.299 [0.266, 0.330] | 0.372 [0.332, 0.418] | 0.184 [0.163, 0.207] | 0.204 [0.180, 0.231] | 0.186 [0.166, 0.208] | MoGe-2 0.101 [0.088, 0.114] |
| depth_test_two_people | 100% of 114 | 0.581 [0.523, 0.645] | 0.460 [0.380, 0.545] | 0.107 [0.084, 0.132] | 0.120 [0.096, 0.145] | 0.130 [0.108, 0.153] | MoGe-2 0.247 [0.194, 0.299] |
| xy_test | 100% of 727 | 173.24 [145.69, 202.12] | 16.40 [15.50, 17.28] | 3.98 [3.79, 4.18] | 3.65 [3.48, 3.82] | 7.06 [6.63, 7.53] | RF-DETR 1.38 [1.00, 1.83] |
| xy_heldout | 100% of 856 | 4.23 [3.46, 5.01] | 18.43 [17.73, 19.13] | 6.66 [6.34, 6.97] | 8.49 [8.15, 8.83] | 9.65 [9.14, 10.16] |  |
| xy_test_people | 100% of 275 | 170.55 [127.23, 214.81] | 13.59 [12.52, 14.66] | 4.19 [3.84, 4.55] | 3.40 [3.14, 3.71] | 6.49 [5.99, 7.01] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 100% of 338 | 0.385 [0.343, 0.430] | 0.459 [0.388, 0.541] | 0.360 [0.313, 0.409] | 0.362 [0.319, 0.410] | 0.314 [0.273, 0.358] |  |
| size_heldout | 100% of 856 | 0.158 [0.146, 0.170] | 0.680 [0.667, 0.693] | 0.664 [0.648, 0.680] | 0.636 [0.618, 0.653] | 0.683 [0.669, 0.699] |  |
| distance_test | 100% of 361 | 0.343 [0.307, 0.383] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 99% of 512 | 0.423 [0.392, 0.453] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked 0.024 [-0.009, 0.059]; asked beats the prior by 6% -> qualitative (pre-registered rule); asked δ<1.25 0.381 [0.323, 0.439]; median asked/true 1.30
- depth_heldout: prior minus asked 0.698 [0.684, 0.711]; asked beats the prior by 78% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.676 [0.642, 0.706]; median asked/true 1.00
- depth_test_indoor: prior minus asked 0.073 [0.033, 0.114]; asked beats the prior by 20% -> qualitative (pre-registered rule); asked δ<1.25 0.497 [0.432, 0.565]; median asked/true 1.22
- depth_test_two_people: prior minus asked -0.121 [-0.169, -0.071]; asked beats the prior by -26% -> qualitative (pre-registered rule); asked δ<1.25 0.035 [0.008, 0.071]; median asked/true 1.52
- xy_test: answers above 100: 35%; median error 3.15; read as 0-1000 (exploratory) 3.09 [2.63, 3.60]; native boxes 2.75 [2.26, 3.29] (n=703, 97% returned); prior minus asked -156.843 [-185.612, -129.462]; asked beats the prior by -956% -> qualitative (pre-registered rule)
- xy_heldout: answers above 100: 0%; median error 0.70; read as 0-1000 (exploratory) 4.23 [3.46, 5.01]; native boxes 4.10 [3.31, 4.91] (n=805, 94% returned); prior minus asked 14.203 [13.252, 15.170]; asked beats the prior by 77% -> a measurement (>= 20%, CI excludes 0)
- xy_test_people: answers above 100: 36%; median error 2.08; read as 0-1000 (exploratory) 1.68 [1.39, 1.96]; native boxes 1.14 [0.88, 1.42] (n=253, 92% returned); prior minus asked -156.953 [-201.041, -113.831]; asked beats the prior by -1155% -> qualitative (pre-registered rule)
- size_test: prior minus asked 0.074 [0.008, 0.155]; asked beats the prior by 16% -> qualitative (pre-registered rule)
- size_heldout: prior minus asked 0.522 [0.510, 0.535]; asked beats the prior by 77% -> a measurement (>= 20%, CI excludes 0)

### Yes/no questions, all frames (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.819 [0.758, 0.881] | 142 | 0.32 |
| people/above | 0.900 [0.852, 0.949] | 144 | 0.47 |
| people/all | 0.863 [0.823, 0.902] | 286 | 0.40 |
| indoor/closer | 0.834 [0.784, 0.882] | 228 | 0.61 |
| indoor/left_of | 0.820 [0.771, 0.867] | 257 | 0.35 |
| indoor/above | 0.848 [0.800, 0.891] | 230 | 0.35 |
| indoor/bigger | 0.854 [0.796, 0.909] | 176 | 0.49 |
| indoor/all | 0.842 [0.814, 0.868] | 891 | 0.44 |
| two_people/closer | 0.856 [0.793, 0.912] | 140 | 0.46 |
| two_people/above | 0.500 [0.500, 0.500] | 20 (n<30: no gate) | 0.00 |
| two_people/all | 0.815 [0.755, 0.871] | 160 | 0.40 |
| street/closer | 0.776 [0.735, 0.815] | 400 | 0.47 |
| street/left_of | 0.763 [0.726, 0.800] | 380 | 0.28 |
| street/above | 0.591 [0.528, 0.667] | 75 | 0.08 |
| street/bigger | 0.978 [0.962, 0.992] | 393 | 0.54 |
| street/all | 0.827 [0.808, 0.847] | 1248 | 0.41 |
| pooled_test/closer | 0.846 [0.807, 0.883] | 368 | 0.55 |
| pooled_test/left_of | 0.820 [0.780, 0.858] | 399 | 0.34 |
| pooled_test/above | 0.854 [0.819, 0.887] | 394 | 0.38 |
| pooled_test/bigger | 0.854 [0.796, 0.909] | 176 | 0.49 |
| pooled_test/all | 0.843 [0.821, 0.864] | 1337 | 0.43 |
| pooled/closer | 0.811 [0.783, 0.838] | 768 | 0.51 |
| pooled/left_of | 0.791 [0.765, 0.818] | 779 | 0.31 |
| pooled/above | 0.818 [0.784, 0.849] | 469 | 0.33 |
| pooled/bigger | 0.940 [0.918, 0.961] | 569 | 0.52 |
| pooled/all | 0.835 [0.821, 0.850] | 2585 | 0.42 |
| naming/people | 0.974 [0.952, 0.992] | 230 | 0.50 |
| naming/indoor | 0.911 [0.873, 0.944] | 248 | 0.58 |
| naming/street | 0.818 [0.780, 0.855] | 400 | 0.50 |
| naming/pooled | 0.885 [0.863, 0.905] | 878 | 0.52 |

### Yes/no questions, without MMA and grappling (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.819 [0.758, 0.881] | 142 | 0.32 |
| people/above | 0.900 [0.852, 0.949] | 144 | 0.47 |
| people/all | 0.863 [0.823, 0.902] | 286 | 0.40 |
| indoor/closer | 0.834 [0.784, 0.882] | 228 | 0.61 |
| indoor/left_of | 0.820 [0.771, 0.867] | 257 | 0.35 |
| indoor/above | 0.848 [0.800, 0.891] | 230 | 0.35 |
| indoor/bigger | 0.854 [0.796, 0.909] | 176 | 0.49 |
| indoor/all | 0.842 [0.814, 0.868] | 891 | 0.44 |
| two_people/closer | 0.882 [0.819, 0.937] | 119 | 0.47 |
| two_people/above | 0.500 [0.500, 0.500] | 17 (n<30: no gate) | 0.00 |
| two_people/all | 0.828 [0.766, 0.888] | 136 | 0.41 |
| street/closer | 0.776 [0.735, 0.815] | 400 | 0.47 |
| street/left_of | 0.763 [0.726, 0.800] | 380 | 0.28 |
| street/above | 0.591 [0.528, 0.667] | 75 | 0.08 |
| street/bigger | 0.978 [0.962, 0.992] | 393 | 0.54 |
| street/all | 0.827 [0.808, 0.847] | 1248 | 0.41 |
| pooled_test/closer | 0.854 [0.816, 0.890] | 347 | 0.56 |
| pooled_test/left_of | 0.820 [0.780, 0.858] | 399 | 0.34 |
| pooled_test/above | 0.854 [0.819, 0.887] | 391 | 0.38 |
| pooled_test/bigger | 0.854 [0.796, 0.909] | 176 | 0.49 |
| pooled_test/all | 0.844 [0.823, 0.866] | 1313 | 0.43 |
| pooled/closer | 0.814 [0.786, 0.841] | 747 | 0.51 |
| pooled/left_of | 0.791 [0.765, 0.818] | 779 | 0.31 |
| pooled/above | 0.818 [0.784, 0.849] | 466 | 0.33 |
| pooled/bigger | 0.940 [0.918, 0.961] | 569 | 0.52 |
| pooled/all | 0.836 [0.822, 0.851] | 2561 | 0.42 |
| naming/people | 0.974 [0.952, 0.992] | 230 | 0.50 |
| naming/indoor | 0.911 [0.873, 0.944] | 248 | 0.58 |
| naming/street | 0.818 [0.780, 0.855] | 400 | 0.50 |
| naming/pooled | 0.885 [0.863, 0.905] | 878 | 0.52 |

## SpatialAxiom-9B


### Probes, all test frames (pre-registered)


**depth** (abs-rel, lower is better; 480 test objects in 195 images; layers chosen on training data: V 25, L_img 24, L_img_any 12, W 12, W_noimg 18, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.381 [0.343, 0.420] |  |
| vision tower | 0.156 [0.138, 0.174] |  |
| image tokens, last quarter of language layers | 0.163 [0.147, 0.180] | 0.97 [0.92, 1.01] |
| image tokens, any language layer | 0.144 [0.129, 0.160] | 1.05 [1.01, 1.10] |
| object word, with the image | 0.161 [0.146, 0.177] | 0.97 [0.92, 1.04] |
| object word, no image | 0.363 [0.328, 0.398] | 0.08 [0.04, 0.12] |
| object word, mismatched image | 0.368 [0.333, 0.406] | 0.05 [0.02, 0.08] |
| control: shuffled labels (tower layer) | 0.478 [0.432, 0.528] |  |
| control: norm-matched random features | 0.478 [0.432, 0.528] |  |

δ<1.25 (higher is better): name prior 0.404 [0.358, 0.449]; vision tower 0.775 [0.730, 0.819]; image tokens, last quarter of language layers 0.752 [0.709, 0.798]; image tokens, any language layer 0.806 [0.763, 0.846]; object word, with the image 0.775 [0.734, 0.812]; object word, no image 0.408 [0.363, 0.453]; object word, mismatched image 0.402 [0.354, 0.446]

G0 control: shuffled labels (tower layer): P minus control -0.097 [-0.134, -0.060] -> pass; control: norm-matched random features: P minus control -0.098 [-0.134, -0.060] -> pass. G1: tower gain over the prior 59.1% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 755 test objects in 310 images; layers chosen on training data: V 6, L_img 24, L_img_any 18, W 16, W_noimg 17, W_mis 6)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.22 [15.36, 17.09] |  |
| vision tower | 3.55 [3.35, 3.74] |  |
| image tokens, last quarter of language layers | 3.98 [3.79, 4.17] | 0.97 [0.95, 0.99] |
| image tokens, any language layer | 3.15 [2.99, 3.30] | 1.03 [1.02, 1.05] |
| object word, with the image | 6.77 [6.37, 7.18] | 0.75 [0.71, 0.78] |
| object word, no image | 16.00 [15.16, 16.84] | 0.02 [0.00, 0.03] |
| object word, mismatched image | 16.09 [15.27, 16.92] | 0.01 [-0.01, 0.03] |
| control: shuffled labels (tower layer) | 18.12 [17.29, 18.96] |  |
| control: norm-matched random features | 18.12 [17.29, 18.96] |  |

G0 control: shuffled labels (tower layer): P minus control -1.899 [-2.285, -1.475] -> pass; control: norm-matched random features: P minus control -1.896 [-2.283, -1.474] -> pass. G1: tower gain over the prior 78.1% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 24, L_img 24, L_img_any 11, W 20, W_noimg 12, W_mis 6)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.333 [0.292, 0.377] |  |
| image tokens, last quarter of language layers | 0.332 [0.291, 0.376] | 1.01 [0.83, 1.24] |
| image tokens, any language layer | 0.322 [0.284, 0.363] | 1.09 [0.93, 1.34] |
| object word, with the image | 0.296 [0.257, 0.335] | 1.30 [1.05, 1.87] |
| object word, no image | 0.430 [0.372, 0.489] | 0.23 [-0.03, 0.45] |
| object word, mismatched image | 0.439 [0.380, 0.502] | 0.16 [-0.09, 0.35] |
| control: shuffled labels (tower layer) | 0.647 [0.560, 0.740] |  |
| control: norm-matched random features | 0.645 [0.558, 0.739] |  |

G0 control: shuffled labels (tower layer): P minus control -0.188 [-0.261, -0.107] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 27.4% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**centre** (3D error, m, lower is better; 480 test objects in 195 images; layers chosen on training data: V 13, L_img 24, L_img_any 13, W 16, W_noimg 13, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.354 [1.281, 1.437] |  |
| vision tower | 0.644 [0.600, 0.688] |  |
| image tokens, last quarter of language layers | 0.587 [0.545, 0.631] | 1.08 [1.04, 1.12] |
| image tokens, any language layer | 0.529 [0.493, 0.568] | 1.16 [1.12, 1.21] |
| object word, with the image | 0.703 [0.652, 0.760] | 0.92 [0.86, 0.97] |
| object word, no image | 1.334 [1.260, 1.419] | 0.03 [0.00, 0.05] |
| object word, mismatched image | 1.335 [1.263, 1.418] | 0.03 [0.01, 0.04] |
| control: shuffled labels (tower layer) | 1.647 [1.566, 1.734] |  |
| control: norm-matched random features | 1.647 [1.566, 1.734] |  |

G0 control: shuffled labels (tower layer): P minus control -0.293 [-0.353, -0.232] -> pass; control: norm-matched random features: P minus control -0.293 [-0.353, -0.233] -> pass. G1: tower gain over the prior 52.5% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.93 [0.90, 0.96] (n=272) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.97, 1.00] (n=381) | 0.83 [0.74, 0.90] (n=176) | 0.96 [0.95, 0.97] (n=1228) |
| image tokens, last quarter of language layers | 0.92 [0.87, 0.95] (n=272) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.86 [0.78, 0.93] (n=176) | 0.96 [0.94, 0.97] (n=1228) |
| object word, with the image | 0.92 [0.88, 0.95] (n=272) | 0.94 [0.91, 0.97] (n=399) | 0.98 [0.96, 0.99] (n=381) | 0.85 [0.79, 0.92] (n=176) | 0.94 [0.92, 0.96] (n=1228) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.636 [0.560, 0.725] (n=375); vision tower 0.395 [0.336, 0.459] (n=375); image tokens, last quarter of language layers 0.372 [0.315, 0.435] (n=375); object word, with the image 0.453 [0.383, 0.535] (n=375)

### Probes, without MMA and grappling frames


**depth** (abs-rel, lower is better; 452 test objects in 181 images; layers chosen on training data: V 25, L_img 24, L_img_any 12, W 12, W_noimg 18, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.394 [0.355, 0.434] |  |
| vision tower | 0.161 [0.142, 0.179] |  |
| image tokens, last quarter of language layers | 0.166 [0.149, 0.184] | 0.98 [0.93, 1.02] |
| image tokens, any language layer | 0.148 [0.133, 0.165] | 1.05 [1.01, 1.10] |
| object word, with the image | 0.167 [0.151, 0.184] | 0.97 [0.91, 1.03] |
| object word, no image | 0.375 [0.339, 0.412] | 0.08 [0.04, 0.12] |
| object word, mismatched image | 0.381 [0.344, 0.419] | 0.06 [0.03, 0.09] |
| control: shuffled labels (tower layer) | 0.488 [0.439, 0.541] |  |
| control: norm-matched random features | 0.488 [0.439, 0.541] |  |

δ<1.25 (higher is better): name prior 0.385 [0.339, 0.430]; vision tower 0.763 [0.714, 0.810]; image tokens, last quarter of language layers 0.741 [0.695, 0.789]; image tokens, any language layer 0.799 [0.754, 0.839]; object word, with the image 0.761 [0.716, 0.799]; object word, no image 0.389 [0.343, 0.434]; object word, mismatched image 0.383 [0.334, 0.427]

G0 control: shuffled labels (tower layer): P minus control -0.094 [-0.132, -0.055] -> pass; control: norm-matched random features: P minus control -0.094 [-0.132, -0.055] -> pass. G1: tower gain over the prior 59.2% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 727 test objects in 296 images; layers chosen on training data: V 6, L_img 24, L_img_any 18, W 16, W_noimg 17, W_mis 6)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.40 [15.50, 17.28] |  |
| vision tower | 3.60 [3.40, 3.80] |  |
| image tokens, last quarter of language layers | 4.01 [3.81, 4.20] | 0.97 [0.95, 0.99] |
| image tokens, any language layer | 3.18 [3.02, 3.34] | 1.03 [1.02, 1.05] |
| object word, with the image | 6.80 [6.38, 7.25] | 0.75 [0.72, 0.78] |
| object word, no image | 16.17 [15.33, 17.02] | 0.02 [0.00, 0.03] |
| object word, mismatched image | 16.26 [15.41, 17.10] | 0.01 [-0.01, 0.03] |
| control: shuffled labels (tower layer) | 18.33 [17.48, 19.20] |  |
| control: norm-matched random features | 18.32 [17.47, 19.19] |  |

G0 control: shuffled labels (tower layer): P minus control -1.932 [-2.338, -1.498] -> pass; control: norm-matched random features: P minus control -1.924 [-2.329, -1.489] -> pass. G1: tower gain over the prior 78.0% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 24, L_img 24, L_img_any 11, W 20, W_noimg 12, W_mis 6)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.333 [0.292, 0.377] |  |
| image tokens, last quarter of language layers | 0.332 [0.291, 0.376] | 1.01 [0.83, 1.24] |
| image tokens, any language layer | 0.322 [0.284, 0.363] | 1.09 [0.93, 1.34] |
| object word, with the image | 0.296 [0.257, 0.335] | 1.30 [1.05, 1.87] |
| object word, no image | 0.430 [0.372, 0.489] | 0.23 [-0.03, 0.45] |
| object word, mismatched image | 0.439 [0.380, 0.502] | 0.16 [-0.09, 0.35] |
| control: shuffled labels (tower layer) | 0.647 [0.560, 0.740] |  |
| control: norm-matched random features | 0.645 [0.558, 0.739] |  |

G0 control: shuffled labels (tower layer): P minus control -0.188 [-0.261, -0.107] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 27.4% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**centre** (3D error, m, lower is better; 452 test objects in 181 images; layers chosen on training data: V 13, L_img 24, L_img_any 13, W 16, W_noimg 13, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.355 [1.278, 1.441] |  |
| vision tower | 0.647 [0.601, 0.694] |  |
| image tokens, last quarter of language layers | 0.585 [0.540, 0.630] | 1.09 [1.04, 1.13] |
| image tokens, any language layer | 0.527 [0.490, 0.567] | 1.17 [1.12, 1.22] |
| object word, with the image | 0.705 [0.650, 0.763] | 0.92 [0.86, 0.98] |
| object word, no image | 1.334 [1.255, 1.420] | 0.03 [0.00, 0.06] |
| object word, mismatched image | 1.335 [1.259, 1.420] | 0.03 [0.01, 0.04] |
| control: shuffled labels (tower layer) | 1.644 [1.562, 1.733] |  |
| control: norm-matched random features | 1.645 [1.562, 1.733] |  |

G0 control: shuffled labels (tower layer): P minus control -0.290 [-0.349, -0.230] -> pass; control: norm-matched random features: P minus control -0.290 [-0.349, -0.230] -> pass. G1: tower gain over the prior 52.3% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.93 [0.90, 0.96] (n=266) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.97, 1.00] (n=381) | 0.83 [0.74, 0.90] (n=176) | 0.96 [0.95, 0.97] (n=1222) |
| image tokens, last quarter of language layers | 0.91 [0.87, 0.95] (n=266) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.86 [0.78, 0.93] (n=176) | 0.96 [0.94, 0.97] (n=1222) |
| object word, with the image | 0.92 [0.89, 0.95] (n=266) | 0.94 [0.91, 0.97] (n=399) | 0.98 [0.96, 0.99] (n=381) | 0.85 [0.79, 0.92] (n=176) | 0.94 [0.92, 0.96] (n=1222) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.625 [0.550, 0.709] (n=361); vision tower 0.387 [0.327, 0.450] (n=361); image tokens, last quarter of language layers 0.371 [0.314, 0.435] (n=361); object word, with the image 0.446 [0.374, 0.527] (n=361)

### Asked answers, all frames (same objects for every column)

Answers: 1385 images, 1380 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 97% of 480 | 0.233 [0.208, 0.264] | 0.383 [0.345, 0.423] | 0.155 [0.137, 0.174] | 0.164 [0.147, 0.181] | 0.161 [0.145, 0.177] | MoGe-2 0.146 [0.127, 0.166] |
| depth_heldout | 100% of 856 | 0.261 [0.241, 0.283] | 0.897 [0.891, 0.902] | 0.808 [0.800, 0.815] | 0.794 [0.786, 0.802] | 0.858 [0.851, 0.864] | MoGe-2 0.103 [0.098, 0.109] |
| depth_test_indoor | 96% of 338 | 0.218 [0.196, 0.243] | 0.375 [0.333, 0.423] | 0.176 [0.154, 0.201] | 0.180 [0.159, 0.204] | 0.161 [0.142, 0.180] | MoGe-2 0.100 [0.087, 0.113] |
| depth_test_two_people | 100% of 142 | 0.267 [0.209, 0.349] | 0.402 [0.334, 0.477] | 0.107 [0.088, 0.130] | 0.127 [0.108, 0.147] | 0.160 [0.135, 0.187] | MoGe-2 0.253 [0.203, 0.300] |
| xy_test | 99% of 755 | 366.09 [343.19, 387.74] | 16.17 [15.28, 17.05] | 3.55 [3.35, 3.74] | 3.97 [3.77, 4.16] | 6.69 [6.29, 7.10] | RF-DETR 1.38 [1.01, 1.84] |
| xy_heldout | 100% of 856 | 317.52 [297.93, 336.11] | 18.44 [17.75, 19.13] | 5.77 [5.50, 6.05] | 6.24 [5.95, 6.55] | 8.64 [8.02, 9.25] |  |
| xy_test_people | 100% of 275 | 454.83 [438.20, 471.96] | 13.59 [12.52, 14.66] | 3.81 [3.46, 4.17] | 3.48 [3.20, 3.74] | 5.80 [5.38, 6.22] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 96% of 338 | 0.333 [0.297, 0.369] | 0.456 [0.384, 0.541] | 0.332 [0.290, 0.377] | 0.335 [0.292, 0.381] | 0.294 [0.256, 0.336] |  |
| size_heldout | 100% of 856 | 0.239 [0.224, 0.253] | 0.680 [0.667, 0.694] | 0.653 [0.636, 0.670] | 0.669 [0.654, 0.684] | 0.666 [0.650, 0.682] |  |
| distance_test | 98% of 375 | 0.414 [0.340, 0.503] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 97% of 512 | 0.603 [0.524, 0.702] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked 0.150 [0.113, 0.190]; asked beats the prior by 39% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.655 [0.603, 0.703]; median asked/true 1.14
- depth_heldout: prior minus asked 0.636 [0.612, 0.657]; asked beats the prior by 71% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.593 [0.558, 0.628]; median asked/true 1.06
- depth_test_indoor: prior minus asked 0.156 [0.114, 0.205]; asked beats the prior by 42% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.668 [0.611, 0.723]; median asked/true 1.14
- depth_test_two_people: prior minus asked 0.135 [0.061, 0.208]; asked beats the prior by 34% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.627 [0.529, 0.715]; median asked/true 1.14
- xy_test: answers above 100: 76%; median error 408.73; read as 0-1000 (exploratory) 13.25 [10.86, 15.69]; native boxes 11.72 [9.54, 13.81] (n=737, 98% returned); prior minus asked -349.918 [-371.761, -326.496]; asked beats the prior by -2164% -> qualitative (pre-registered rule)
- xy_heldout: answers above 100: 70%; median error 388.32; read as 0-1000 (exploratory) 5.69 [4.96, 6.44]; native boxes 6.35 [5.35, 7.38] (n=842, 98% returned); prior minus asked -299.074 [-317.894, -279.427]; asked beats the prior by -1622% -> qualitative (pre-registered rule)
- xy_test_people: answers above 100: 99%; median error 449.74; read as 0-1000 (exploratory) 3.66 [2.64, 5.15]; native boxes 3.41 [1.87, 5.28] (n=273, 99% returned); prior minus asked -441.235 [-458.466, -424.156]; asked beats the prior by -3246% -> qualitative (pre-registered rule)
- size_test: prior minus asked 0.123 [0.051, 0.207]; asked beats the prior by 27% -> a measurement (>= 20%, CI excludes 0)
- size_heldout: prior minus asked 0.441 [0.426, 0.457]; asked beats the prior by 65% -> a measurement (>= 20%, CI excludes 0)

### Asked answers, without MMA and grappling (same objects for every column)

Answers: 1385 images, 1380 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 97% of 452 | 0.233 [0.207, 0.266] | 0.397 [0.357, 0.438] | 0.160 [0.141, 0.180] | 0.167 [0.150, 0.185] | 0.166 [0.150, 0.183] | MoGe-2 0.138 [0.119, 0.157] |
| depth_heldout | 100% of 856 | 0.261 [0.241, 0.283] | 0.897 [0.891, 0.902] | 0.808 [0.800, 0.815] | 0.794 [0.786, 0.802] | 0.858 [0.851, 0.864] | MoGe-2 0.103 [0.098, 0.109] |
| depth_test_indoor | 96% of 338 | 0.218 [0.196, 0.243] | 0.375 [0.333, 0.423] | 0.176 [0.154, 0.201] | 0.180 [0.159, 0.204] | 0.161 [0.142, 0.180] | MoGe-2 0.100 [0.087, 0.113] |
| depth_test_two_people | 100% of 114 | 0.276 [0.205, 0.372] | 0.460 [0.380, 0.545] | 0.115 [0.091, 0.141] | 0.131 [0.108, 0.154] | 0.182 [0.153, 0.213] | MoGe-2 0.247 [0.194, 0.299] |
| xy_test | 99% of 727 | 364.66 [340.53, 387.05] | 16.35 [15.44, 17.24] | 3.60 [3.40, 3.80] | 4.00 [3.80, 4.19] | 6.72 [6.32, 7.14] | RF-DETR 1.38 [1.00, 1.83] |
| xy_heldout | 100% of 856 | 317.52 [297.93, 336.11] | 18.44 [17.75, 19.13] | 5.77 [5.50, 6.05] | 6.24 [5.95, 6.55] | 8.64 [8.02, 9.25] |  |
| xy_test_people | 100% of 275 | 454.83 [438.20, 471.96] | 13.59 [12.52, 14.66] | 3.81 [3.46, 4.17] | 3.48 [3.20, 3.74] | 5.80 [5.38, 6.22] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 96% of 338 | 0.333 [0.297, 0.369] | 0.456 [0.384, 0.541] | 0.332 [0.290, 0.377] | 0.335 [0.292, 0.381] | 0.294 [0.256, 0.336] |  |
| size_heldout | 100% of 856 | 0.239 [0.224, 0.253] | 0.680 [0.667, 0.694] | 0.653 [0.636, 0.670] | 0.669 [0.654, 0.684] | 0.666 [0.650, 0.682] |  |
| distance_test | 98% of 361 | 0.397 [0.323, 0.489] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 97% of 512 | 0.603 [0.524, 0.702] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked 0.164 [0.125, 0.204]; asked beats the prior by 41% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.651 [0.598, 0.701]; median asked/true 1.14
- depth_heldout: prior minus asked 0.636 [0.612, 0.657]; asked beats the prior by 71% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.593 [0.558, 0.628]; median asked/true 1.06
- depth_test_indoor: prior minus asked 0.156 [0.114, 0.205]; asked beats the prior by 42% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.668 [0.611, 0.723]; median asked/true 1.14
- depth_test_two_people: prior minus asked 0.185 [0.101, 0.263]; asked beats the prior by 40% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.605 [0.489, 0.714]; median asked/true 1.17
- xy_test: answers above 100: 76%; median error 408.84; read as 0-1000 (exploratory) 13.38 [10.96, 15.93]; native boxes 11.83 [9.63, 14.02] (n=717, 99% returned); prior minus asked -348.317 [-370.732, -324.141]; asked beats the prior by -2131% -> qualitative (pre-registered rule)
- xy_heldout: answers above 100: 70%; median error 388.32; read as 0-1000 (exploratory) 5.69 [4.96, 6.44]; native boxes 6.35 [5.35, 7.38] (n=842, 98% returned); prior minus asked -299.074 [-317.894, -279.427]; asked beats the prior by -1622% -> qualitative (pre-registered rule)
- xy_test_people: answers above 100: 99%; median error 449.74; read as 0-1000 (exploratory) 3.66 [2.64, 5.15]; native boxes 3.41 [1.87, 5.28] (n=273, 99% returned); prior minus asked -441.235 [-458.466, -424.156]; asked beats the prior by -3246% -> qualitative (pre-registered rule)
- size_test: prior minus asked 0.123 [0.051, 0.207]; asked beats the prior by 27% -> a measurement (>= 20%, CI excludes 0)
- size_heldout: prior minus asked 0.441 [0.426, 0.457]; asked beats the prior by 65% -> a measurement (>= 20%, CI excludes 0)

### Yes/no questions, all frames (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.986 [0.963, 1.000] | 142 | 0.49 |
| people/above | 0.992 [0.973, 1.000] | 144 | 0.58 |
| people/all | 0.990 [0.976, 1.000] | 286 | 0.53 |
| indoor/closer | 0.943 [0.905, 0.974] | 228 | 0.55 |
| indoor/left_of | 0.911 [0.872, 0.949] | 257 | 0.52 |
| indoor/above | 0.957 [0.928, 0.984] | 230 | 0.50 |
| indoor/bigger | 0.857 [0.797, 0.911] | 176 | 0.54 |
| indoor/all | 0.920 [0.899, 0.940] | 891 | 0.53 |
| two_people/closer | 0.917 [0.871, 0.959] | 140 | 0.57 |
| two_people/above | 0.293 [0.086, 0.508] | 20 (n<30: no gate) | 0.45 |
| two_people/all | 0.840 [0.790, 0.894] | 160 | 0.56 |
| street/closer | 0.865 [0.828, 0.899] | 400 | 0.55 |
| street/left_of | 0.901 [0.868, 0.931] | 380 | 0.49 |
| street/above | 0.964 [0.922, 1.000] | 75 | 0.48 |
| street/bigger | 0.997 [0.991, 1.000] | 393 | 0.55 |
| street/all | 0.922 [0.904, 0.939] | 1248 | 0.53 |
| pooled_test/closer | 0.930 [0.900, 0.957] | 368 | 0.56 |
| pooled_test/left_of | 0.937 [0.911, 0.963] | 399 | 0.51 |
| pooled_test/above | 0.936 [0.910, 0.959] | 394 | 0.53 |
| pooled_test/bigger | 0.857 [0.797, 0.911] | 176 | 0.54 |
| pooled_test/all | 0.925 [0.908, 0.941] | 1337 | 0.53 |
| pooled/closer | 0.896 [0.872, 0.918] | 768 | 0.55 |
| pooled/left_of | 0.919 [0.899, 0.939] | 779 | 0.50 |
| pooled/above | 0.940 [0.918, 0.962] | 469 | 0.52 |
| pooled/bigger | 0.953 [0.933, 0.972] | 569 | 0.55 |
| pooled/all | 0.923 [0.912, 0.934] | 2585 | 0.53 |
| naming/people | 0.983 [0.965, 0.996] | 230 | 0.49 |
| naming/indoor | 0.931 [0.894, 0.965] | 248 | 0.52 |
| naming/street | 0.797 [0.758, 0.834] | 400 | 0.54 |
| naming/pooled | 0.884 [0.861, 0.905] | 878 | 0.52 |

### Yes/no questions, without MMA and grappling (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.986 [0.963, 1.000] | 142 | 0.49 |
| people/above | 0.992 [0.973, 1.000] | 144 | 0.58 |
| people/all | 0.990 [0.976, 1.000] | 286 | 0.53 |
| indoor/closer | 0.943 [0.905, 0.974] | 228 | 0.55 |
| indoor/left_of | 0.911 [0.872, 0.949] | 257 | 0.52 |
| indoor/above | 0.957 [0.928, 0.984] | 230 | 0.50 |
| indoor/bigger | 0.857 [0.797, 0.911] | 176 | 0.54 |
| indoor/all | 0.920 [0.899, 0.940] | 891 | 0.53 |
| two_people/closer | 0.937 [0.892, 0.976] | 119 | 0.54 |
| two_people/above | 0.174 [0.000, 0.384] | 17 (n<30: no gate) | 0.53 |
| two_people/all | 0.841 [0.785, 0.898] | 136 | 0.54 |
| street/closer | 0.865 [0.828, 0.899] | 400 | 0.55 |
| street/left_of | 0.901 [0.868, 0.931] | 380 | 0.49 |
| street/above | 0.964 [0.922, 1.000] | 75 | 0.48 |
| street/bigger | 0.997 [0.991, 1.000] | 393 | 0.55 |
| street/all | 0.922 [0.904, 0.939] | 1248 | 0.53 |
| pooled_test/closer | 0.938 [0.909, 0.964] | 347 | 0.54 |
| pooled_test/left_of | 0.937 [0.911, 0.963] | 399 | 0.51 |
| pooled_test/above | 0.936 [0.909, 0.959] | 391 | 0.53 |
| pooled_test/bigger | 0.857 [0.797, 0.911] | 176 | 0.54 |
| pooled_test/all | 0.926 [0.910, 0.943] | 1313 | 0.53 |
| pooled/closer | 0.898 [0.874, 0.921] | 747 | 0.54 |
| pooled/left_of | 0.919 [0.899, 0.939] | 779 | 0.50 |
| pooled/above | 0.940 [0.917, 0.961] | 466 | 0.52 |
| pooled/bigger | 0.953 [0.933, 0.972] | 569 | 0.55 |
| pooled/all | 0.924 [0.912, 0.936] | 2561 | 0.53 |
| naming/people | 0.983 [0.965, 0.996] | 230 | 0.49 |
| naming/indoor | 0.931 [0.894, 0.965] | 248 | 0.52 |
| naming/street | 0.797 [0.758, 0.834] | 400 | 0.54 |
| naming/pooled | 0.884 [0.861, 0.905] | 878 | 0.52 |

## Qwen3.5-9B


### Probes, all test frames (pre-registered)


**depth** (abs-rel, lower is better; 480 test objects in 195 images; layers chosen on training data: V 25, L_img 24, L_img_any 11, W 17, W_noimg 18, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.381 [0.343, 0.420] |  |
| vision tower | 0.159 [0.140, 0.180] |  |
| image tokens, last quarter of language layers | 0.160 [0.143, 0.177] | 1.00 [0.95, 1.05] |
| image tokens, any language layer | 0.153 [0.136, 0.171] | 1.03 [0.98, 1.08] |
| object word, with the image | 0.170 [0.153, 0.188] | 0.95 [0.89, 1.01] |
| object word, no image | 0.362 [0.328, 0.397] | 0.08 [0.04, 0.13] |
| object word, mismatched image | 0.368 [0.333, 0.406] | 0.06 [0.02, 0.09] |
| control: shuffled labels (tower layer) | 0.478 [0.432, 0.528] |  |
| control: norm-matched random features | 0.478 [0.432, 0.528] |  |

δ<1.25 (higher is better): name prior 0.404 [0.358, 0.449]; vision tower 0.762 [0.716, 0.805]; image tokens, last quarter of language layers 0.775 [0.733, 0.816]; image tokens, any language layer 0.773 [0.730, 0.814]; object word, with the image 0.756 [0.717, 0.796]; object word, no image 0.400 [0.355, 0.443]; object word, mismatched image 0.400 [0.353, 0.444]

G0 control: shuffled labels (tower layer): P minus control -0.097 [-0.133, -0.060] -> pass; control: norm-matched random features: P minus control -0.097 [-0.134, -0.060] -> pass. G1: tower gain over the prior 58.1% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 755 test objects in 310 images; layers chosen on training data: V 5, L_img 24, L_img_any 14, W 16, W_noimg 19, W_mis 6)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.22 [15.36, 17.09] |  |
| vision tower | 3.59 [3.41, 3.77] |  |
| image tokens, last quarter of language layers | 4.65 [4.44, 4.85] | 0.92 [0.90, 0.93] |
| image tokens, any language layer | 3.49 [3.33, 3.66] | 1.01 [0.99, 1.02] |
| object word, with the image | 6.89 [6.48, 7.32] | 0.74 [0.71, 0.77] |
| object word, no image | 16.12 [15.28, 16.97] | 0.01 [-0.01, 0.02] |
| object word, mismatched image | 16.09 [15.28, 16.92] | 0.01 [-0.01, 0.03] |
| control: shuffled labels (tower layer) | 18.11 [17.28, 18.95] |  |
| control: norm-matched random features | 18.12 [17.29, 18.96] |  |

G0 control: shuffled labels (tower layer): P minus control -1.890 [-2.276, -1.467] -> pass; control: norm-matched random features: P minus control -1.896 [-2.283, -1.474] -> pass. G1: tower gain over the prior 77.9% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 24, L_img 24, L_img_any 20, W 20, W_noimg 10, W_mis 7)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.340 [0.298, 0.387] |  |
| image tokens, last quarter of language layers | 0.340 [0.297, 0.387] | 1.00 [0.84, 1.20] |
| image tokens, any language layer | 0.329 [0.289, 0.373] | 1.09 [0.93, 1.36] |
| object word, with the image | 0.306 [0.267, 0.349] | 1.28 [1.04, 1.85] |
| object word, no image | 0.430 [0.371, 0.492] | 0.25 [-0.06, 0.49] |
| object word, mismatched image | 0.427 [0.372, 0.484] | 0.27 [0.00, 0.49] |
| control: shuffled labels (tower layer) | 0.647 [0.560, 0.741] |  |
| control: norm-matched random features | 0.645 [0.558, 0.739] |  |

G0 control: shuffled labels (tower layer): P minus control -0.188 [-0.260, -0.107] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 26.0% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**centre** (3D error, m, lower is better; 480 test objects in 195 images; layers chosen on training data: V 13, L_img 24, L_img_any 14, W 18, W_noimg 19, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.354 [1.281, 1.437] |  |
| vision tower | 0.648 [0.605, 0.693] |  |
| image tokens, last quarter of language layers | 0.584 [0.547, 0.623] | 1.09 [1.05, 1.13] |
| image tokens, any language layer | 0.535 [0.499, 0.575] | 1.16 [1.11, 1.21] |
| object word, with the image | 0.725 [0.675, 0.779] | 0.89 [0.84, 0.94] |
| object word, no image | 1.335 [1.261, 1.419] | 0.03 [0.00, 0.05] |
| object word, mismatched image | 1.336 [1.264, 1.420] | 0.03 [0.01, 0.04] |
| control: shuffled labels (tower layer) | 1.647 [1.566, 1.734] |  |
| control: norm-matched random features | 1.647 [1.566, 1.734] |  |

G0 control: shuffled labels (tower layer): P minus control -0.293 [-0.353, -0.232] -> pass; control: norm-matched random features: P minus control -0.293 [-0.353, -0.233] -> pass. G1: tower gain over the prior 52.1% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.93 [0.89, 0.96] (n=272) | 0.99 [0.98, 1.00] (n=399) | 0.98 [0.96, 1.00] (n=381) | 0.86 [0.78, 0.93] (n=176) | 0.96 [0.95, 0.97] (n=1228) |
| image tokens, last quarter of language layers | 0.90 [0.87, 0.94] (n=272) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.83 [0.75, 0.90] (n=176) | 0.95 [0.94, 0.97] (n=1228) |
| object word, with the image | 0.91 [0.88, 0.94] (n=272) | 0.94 [0.92, 0.97] (n=399) | 0.98 [0.97, 1.00] (n=381) | 0.85 [0.78, 0.92] (n=176) | 0.94 [0.92, 0.95] (n=1228) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.636 [0.560, 0.725] (n=375); vision tower 0.417 [0.348, 0.496] (n=375); image tokens, last quarter of language layers 0.368 [0.312, 0.430] (n=375); object word, with the image 0.480 [0.399, 0.574] (n=375)

### Probes, without MMA and grappling frames


**depth** (abs-rel, lower is better; 452 test objects in 181 images; layers chosen on training data: V 25, L_img 24, L_img_any 11, W 17, W_noimg 18, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.394 [0.355, 0.434] |  |
| vision tower | 0.164 [0.144, 0.186] |  |
| image tokens, last quarter of language layers | 0.163 [0.146, 0.181] | 1.00 [0.96, 1.05] |
| image tokens, any language layer | 0.157 [0.140, 0.176] | 1.03 [0.98, 1.08] |
| object word, with the image | 0.176 [0.157, 0.195] | 0.95 [0.89, 1.01] |
| object word, no image | 0.374 [0.338, 0.411] | 0.09 [0.04, 0.13] |
| object word, mismatched image | 0.381 [0.344, 0.419] | 0.06 [0.02, 0.09] |
| control: shuffled labels (tower layer) | 0.487 [0.438, 0.541] |  |
| control: norm-matched random features | 0.488 [0.439, 0.541] |  |

δ<1.25 (higher is better): name prior 0.385 [0.339, 0.430]; vision tower 0.754 [0.707, 0.800]; image tokens, last quarter of language layers 0.765 [0.721, 0.808]; image tokens, any language layer 0.761 [0.716, 0.804]; object word, with the image 0.741 [0.698, 0.781]; object word, no image 0.381 [0.335, 0.425]; object word, mismatched image 0.381 [0.331, 0.426]

G0 control: shuffled labels (tower layer): P minus control -0.093 [-0.131, -0.055] -> pass; control: norm-matched random features: P minus control -0.094 [-0.132, -0.055] -> pass. G1: tower gain over the prior 58.3% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 727 test objects in 296 images; layers chosen on training data: V 5, L_img 24, L_img_any 14, W 16, W_noimg 19, W_mis 6)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.40 [15.50, 17.28] |  |
| vision tower | 3.65 [3.47, 3.83] |  |
| image tokens, last quarter of language layers | 4.71 [4.49, 4.92] | 0.92 [0.90, 0.93] |
| image tokens, any language layer | 3.56 [3.39, 3.73] | 1.01 [0.99, 1.02] |
| object word, with the image | 6.96 [6.53, 7.42] | 0.74 [0.71, 0.77] |
| object word, no image | 16.29 [15.44, 17.15] | 0.01 [-0.01, 0.02] |
| object word, mismatched image | 16.26 [15.42, 17.10] | 0.01 [-0.01, 0.03] |
| control: shuffled labels (tower layer) | 18.32 [17.47, 19.18] |  |
| control: norm-matched random features | 18.32 [17.47, 19.19] |  |

G0 control: shuffled labels (tower layer): P minus control -1.919 [-2.327, -1.487] -> pass; control: norm-matched random features: P minus control -1.924 [-2.329, -1.489] -> pass. G1: tower gain over the prior 77.7% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 24, L_img 24, L_img_any 20, W 20, W_noimg 10, W_mis 7)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.340 [0.298, 0.387] |  |
| image tokens, last quarter of language layers | 0.340 [0.297, 0.387] | 1.00 [0.84, 1.20] |
| image tokens, any language layer | 0.329 [0.289, 0.373] | 1.09 [0.93, 1.36] |
| object word, with the image | 0.306 [0.267, 0.349] | 1.28 [1.04, 1.85] |
| object word, no image | 0.430 [0.371, 0.492] | 0.25 [-0.06, 0.49] |
| object word, mismatched image | 0.427 [0.372, 0.484] | 0.27 [0.00, 0.49] |
| control: shuffled labels (tower layer) | 0.647 [0.560, 0.741] |  |
| control: norm-matched random features | 0.645 [0.558, 0.739] |  |

G0 control: shuffled labels (tower layer): P minus control -0.188 [-0.260, -0.107] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 26.0% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**centre** (3D error, m, lower is better; 452 test objects in 181 images; layers chosen on training data: V 13, L_img 24, L_img_any 14, W 18, W_noimg 19, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.355 [1.278, 1.441] |  |
| vision tower | 0.651 [0.606, 0.699] |  |
| image tokens, last quarter of language layers | 0.584 [0.545, 0.624] | 1.10 [1.05, 1.14] |
| image tokens, any language layer | 0.537 [0.499, 0.579] | 1.16 [1.11, 1.22] |
| object word, with the image | 0.725 [0.672, 0.781] | 0.90 [0.84, 0.95] |
| object word, no image | 1.334 [1.257, 1.420] | 0.03 [0.00, 0.06] |
| object word, mismatched image | 1.336 [1.261, 1.421] | 0.03 [0.01, 0.04] |
| control: shuffled labels (tower layer) | 1.644 [1.562, 1.733] |  |
| control: norm-matched random features | 1.645 [1.562, 1.733] |  |

G0 control: shuffled labels (tower layer): P minus control -0.290 [-0.349, -0.230] -> pass; control: norm-matched random features: P minus control -0.290 [-0.349, -0.230] -> pass. G1: tower gain over the prior 51.9% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.92 [0.89, 0.96] (n=266) | 0.99 [0.98, 1.00] (n=399) | 0.98 [0.96, 1.00] (n=381) | 0.86 [0.78, 0.93] (n=176) | 0.96 [0.95, 0.97] (n=1222) |
| image tokens, last quarter of language layers | 0.91 [0.87, 0.94] (n=266) | 0.99 [0.98, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.83 [0.75, 0.90] (n=176) | 0.95 [0.94, 0.97] (n=1222) |
| object word, with the image | 0.91 [0.88, 0.95] (n=266) | 0.94 [0.92, 0.97] (n=399) | 0.98 [0.97, 1.00] (n=381) | 0.85 [0.78, 0.92] (n=176) | 0.94 [0.92, 0.95] (n=1222) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.625 [0.550, 0.709] (n=361); vision tower 0.408 [0.340, 0.486] (n=361); image tokens, last quarter of language layers 0.366 [0.307, 0.430] (n=361); object word, with the image 0.471 [0.389, 0.565] (n=361)

### Asked answers, all frames (same objects for every column)

Answers: 1385 images, 1384 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 100% of 480 | 0.347 [0.311, 0.385] | 0.381 [0.343, 0.420] | 0.159 [0.140, 0.180] | 0.160 [0.143, 0.177] | 0.170 [0.153, 0.188] | MoGe-2 0.146 [0.127, 0.165] |
| depth_heldout | 100% of 856 | 0.359 [0.325, 0.394] | 0.897 [0.891, 0.902] | 0.823 [0.815, 0.830] | 0.820 [0.813, 0.828] | 0.884 [0.878, 0.889] | MoGe-2 0.103 [0.098, 0.109] |
| depth_test_indoor | 100% of 338 | 0.237 [0.214, 0.263] | 0.372 [0.332, 0.418] | 0.185 [0.161, 0.212] | 0.172 [0.152, 0.195] | 0.179 [0.157, 0.202] | MoGe-2 0.101 [0.088, 0.114] |
| depth_test_two_people | 100% of 142 | 0.609 [0.534, 0.686] | 0.402 [0.334, 0.477] | 0.100 [0.080, 0.121] | 0.129 [0.108, 0.151] | 0.151 [0.127, 0.176] | MoGe-2 0.253 [0.203, 0.300] |
| xy_test | 100% of 755 | 406.74 [388.95, 423.84] | 16.22 [15.36, 17.09] | 3.59 [3.41, 3.77] | 4.65 [4.44, 4.85] | 6.89 [6.48, 7.32] | RF-DETR 1.38 [1.01, 1.84] |
| xy_heldout | 100% of 856 | 343.88 [325.87, 362.27] | 18.44 [17.75, 19.13] | 6.42 [6.12, 6.72] | 6.13 [5.89, 6.38] | 8.66 [8.10, 9.24] |  |
| xy_test_people | 100% of 275 | 444.40 [428.06, 460.90] | 13.59 [12.52, 14.66] | 3.63 [3.34, 3.94] | 4.88 [4.56, 5.21] | 6.62 [6.10, 7.15] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 100% of 338 | 0.349 [0.305, 0.396] | 0.459 [0.388, 0.541] | 0.340 [0.298, 0.387] | 0.340 [0.297, 0.387] | 0.306 [0.267, 0.349] |  |
| size_heldout | 100% of 856 | 0.213 [0.192, 0.237] | 0.680 [0.667, 0.694] | 0.674 [0.657, 0.691] | 0.685 [0.669, 0.701] | 0.684 [0.670, 0.699] |  |
| distance_test | 98% of 375 | 0.385 [0.327, 0.451] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 96% of 512 | 0.957 [0.783, 1.177] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked 0.034 [-0.005, 0.076]; asked beats the prior by 9% -> qualitative (pre-registered rule); asked δ<1.25 0.469 [0.417, 0.523]; median asked/true 1.23
- depth_heldout: prior minus asked 0.537 [0.504, 0.570]; asked beats the prior by 60% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.519 [0.483, 0.555]; median asked/true 1.06
- depth_test_indoor: prior minus asked 0.134 [0.090, 0.181]; asked beats the prior by 36% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.621 [0.572, 0.670]; median asked/true 1.14
- depth_test_two_people: prior minus asked -0.207 [-0.245, -0.167]; asked beats the prior by -51% -> qualitative (pre-registered rule); asked δ<1.25 0.106 [0.053, 0.169]; median asked/true 1.48
- xy_test: answers above 100: 90%; median error 416.07; read as 0-1000 (exploratory) 8.38 [7.22, 9.63]; native boxes 3.45 [2.90, 4.03] (n=754, 100% returned); prior minus asked -390.518 [-407.770, -372.513]; asked beats the prior by -2408% -> qualitative (pre-registered rule)
- xy_heldout: answers above 100: 82%; median error 381.52; read as 0-1000 (exploratory) 6.04 [5.16, 7.01]; native boxes 4.47 [3.68, 5.32] (n=835, 98% returned); prior minus asked -325.442 [-344.074, -306.951]; asked beats the prior by -1765% -> qualitative (pre-registered rule)
- xy_test_people: answers above 100: 99%; median error 445.73; read as 0-1000 (exploratory) 4.36 [3.55, 5.19]; native boxes 1.08 [0.86, 1.30] (n=275, 100% returned); prior minus asked -430.809 [-447.595, -414.326]; asked beats the prior by -3170% -> qualitative (pre-registered rule)
- size_test: prior minus asked 0.111 [0.037, 0.197]; asked beats the prior by 24% -> a measurement (>= 20%, CI excludes 0)
- size_heldout: prior minus asked 0.467 [0.442, 0.488]; asked beats the prior by 69% -> a measurement (>= 20%, CI excludes 0)

### Asked answers, without MMA and grappling (same objects for every column)

Answers: 1385 images, 1384 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 100% of 452 | 0.341 [0.303, 0.381] | 0.394 [0.355, 0.434] | 0.164 [0.144, 0.186] | 0.163 [0.146, 0.181] | 0.176 [0.157, 0.195] | MoGe-2 0.138 [0.119, 0.156] |
| depth_heldout | 100% of 856 | 0.359 [0.325, 0.394] | 0.897 [0.891, 0.902] | 0.823 [0.815, 0.830] | 0.820 [0.813, 0.828] | 0.884 [0.878, 0.889] | MoGe-2 0.103 [0.098, 0.109] |
| depth_test_indoor | 100% of 338 | 0.237 [0.214, 0.263] | 0.372 [0.332, 0.418] | 0.185 [0.161, 0.212] | 0.172 [0.152, 0.195] | 0.179 [0.157, 0.202] | MoGe-2 0.101 [0.088, 0.114] |
| depth_test_two_people | 100% of 114 | 0.650 [0.560, 0.744] | 0.460 [0.380, 0.545] | 0.104 [0.081, 0.128] | 0.135 [0.111, 0.160] | 0.166 [0.137, 0.196] | MoGe-2 0.247 [0.194, 0.299] |
| xy_test | 100% of 727 | 407.76 [390.10, 425.05] | 16.40 [15.50, 17.28] | 3.65 [3.47, 3.83] | 4.71 [4.49, 4.92] | 6.96 [6.53, 7.42] | RF-DETR 1.38 [1.00, 1.83] |
| xy_heldout | 100% of 856 | 343.88 [325.87, 362.27] | 18.44 [17.75, 19.13] | 6.42 [6.12, 6.72] | 6.13 [5.89, 6.38] | 8.66 [8.10, 9.24] |  |
| xy_test_people | 100% of 275 | 444.40 [428.06, 460.90] | 13.59 [12.52, 14.66] | 3.63 [3.34, 3.94] | 4.88 [4.56, 5.21] | 6.62 [6.10, 7.15] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 100% of 338 | 0.349 [0.305, 0.396] | 0.459 [0.388, 0.541] | 0.340 [0.298, 0.387] | 0.340 [0.297, 0.387] | 0.306 [0.267, 0.349] |  |
| size_heldout | 100% of 856 | 0.213 [0.192, 0.237] | 0.680 [0.667, 0.694] | 0.674 [0.657, 0.691] | 0.685 [0.669, 0.701] | 0.684 [0.670, 0.699] |  |
| distance_test | 98% of 361 | 0.347 [0.303, 0.400] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 96% of 512 | 0.957 [0.783, 1.177] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked 0.053 [0.014, 0.096]; asked beats the prior by 13% -> qualitative (pre-registered rule); asked δ<1.25 0.493 [0.442, 0.548]; median asked/true 1.21
- depth_heldout: prior minus asked 0.537 [0.504, 0.570]; asked beats the prior by 60% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.519 [0.483, 0.555]; median asked/true 1.06
- depth_test_indoor: prior minus asked 0.134 [0.090, 0.181]; asked beats the prior by 36% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.621 [0.572, 0.670]; median asked/true 1.14
- depth_test_two_people: prior minus asked -0.189 [-0.229, -0.147]; asked beats the prior by -41% -> qualitative (pre-registered rule); asked δ<1.25 0.114 [0.052, 0.190]; median asked/true 1.50
- xy_test: answers above 100: 90%; median error 420.15; read as 0-1000 (exploratory) 8.38 [7.18, 9.66]; native boxes 3.36 [2.83, 3.93] (n=726, 100% returned); prior minus asked -391.358 [-408.966, -373.377]; asked beats the prior by -2387% -> qualitative (pre-registered rule)
- xy_heldout: answers above 100: 82%; median error 381.52; read as 0-1000 (exploratory) 6.04 [5.16, 7.01]; native boxes 4.47 [3.68, 5.32] (n=835, 98% returned); prior minus asked -325.442 [-344.074, -306.951]; asked beats the prior by -1765% -> qualitative (pre-registered rule)
- xy_test_people: answers above 100: 99%; median error 445.73; read as 0-1000 (exploratory) 4.36 [3.55, 5.19]; native boxes 1.08 [0.86, 1.30] (n=275, 100% returned); prior minus asked -430.809 [-447.595, -414.326]; asked beats the prior by -3170% -> qualitative (pre-registered rule)
- size_test: prior minus asked 0.111 [0.037, 0.197]; asked beats the prior by 24% -> a measurement (>= 20%, CI excludes 0)
- size_heldout: prior minus asked 0.467 [0.442, 0.488]; asked beats the prior by 69% -> a measurement (>= 20%, CI excludes 0)

### Yes/no questions, all frames (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.854 [0.794, 0.907] | 142 | 0.36 |
| people/above | 0.762 [0.708, 0.815] | 144 | 0.30 |
| people/all | 0.805 [0.764, 0.843] | 286 | 0.33 |
| indoor/closer | 0.884 [0.842, 0.922] | 228 | 0.47 |
| indoor/left_of | 0.848 [0.798, 0.895] | 257 | 0.40 |
| indoor/above | 0.804 [0.760, 0.846] | 230 | 0.30 |
| indoor/bigger | 0.818 [0.747, 0.880] | 176 | 0.40 |
| indoor/all | 0.841 [0.817, 0.867] | 891 | 0.40 |
| two_people/closer | 0.914 [0.862, 0.958] | 140 | 0.49 |
| two_people/above | 0.556 [0.500, 0.682] | 20 (n<30: no gate) | 0.05 |
| two_people/all | 0.873 [0.818, 0.920] | 160 | 0.43 |
| street/closer | 0.778 [0.736, 0.817] | 400 | 0.33 |
| street/left_of | 0.788 [0.747, 0.829] | 380 | 0.31 |
| street/above | 0.545 [0.500, 0.600] | 75 | 0.04 |
| street/bigger | 0.991 [0.981, 0.998] | 393 | 0.54 |
| street/all | 0.840 [0.819, 0.861] | 1248 | 0.37 |
| pooled_test/closer | 0.894 [0.862, 0.923] | 368 | 0.48 |
| pooled_test/left_of | 0.850 [0.811, 0.887] | 399 | 0.39 |
| pooled_test/above | 0.777 [0.741, 0.808] | 394 | 0.29 |
| pooled_test/bigger | 0.818 [0.747, 0.880] | 176 | 0.40 |
| pooled_test/all | 0.836 [0.816, 0.857] | 1337 | 0.39 |
| pooled/closer | 0.836 [0.809, 0.863] | 768 | 0.40 |
| pooled/left_of | 0.819 [0.791, 0.847] | 779 | 0.35 |
| pooled/above | 0.745 [0.711, 0.776] | 469 | 0.25 |
| pooled/bigger | 0.938 [0.915, 0.959] | 569 | 0.50 |
| pooled/all | 0.838 [0.823, 0.853] | 2585 | 0.38 |
| naming/people | 0.974 [0.951, 0.992] | 230 | 0.47 |
| naming/indoor | 0.952 [0.924, 0.976] | 248 | 0.51 |
| naming/street | 0.853 [0.817, 0.886] | 400 | 0.48 |
| naming/pooled | 0.912 [0.892, 0.931] | 878 | 0.49 |

### Yes/no questions, without MMA and grappling (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.854 [0.794, 0.907] | 142 | 0.36 |
| people/above | 0.762 [0.708, 0.815] | 144 | 0.30 |
| people/all | 0.805 [0.764, 0.843] | 286 | 0.33 |
| indoor/closer | 0.884 [0.842, 0.922] | 228 | 0.47 |
| indoor/left_of | 0.848 [0.798, 0.895] | 257 | 0.40 |
| indoor/above | 0.804 [0.760, 0.846] | 230 | 0.30 |
| indoor/bigger | 0.818 [0.747, 0.880] | 176 | 0.40 |
| indoor/all | 0.841 [0.817, 0.867] | 891 | 0.40 |
| two_people/closer | 0.975 [0.943, 1.000] | 119 | 0.48 |
| two_people/above | 0.556 [0.500, 0.682] | 17 (n<30: no gate) | 0.06 |
| two_people/all | 0.917 [0.870, 0.959] | 136 | 0.43 |
| street/closer | 0.778 [0.736, 0.817] | 400 | 0.33 |
| street/left_of | 0.788 [0.747, 0.829] | 380 | 0.31 |
| street/above | 0.545 [0.500, 0.600] | 75 | 0.04 |
| street/bigger | 0.991 [0.981, 0.998] | 393 | 0.54 |
| street/all | 0.840 [0.819, 0.861] | 1248 | 0.37 |
| pooled_test/closer | 0.913 [0.882, 0.941] | 347 | 0.48 |
| pooled_test/left_of | 0.850 [0.811, 0.887] | 399 | 0.39 |
| pooled_test/above | 0.777 [0.741, 0.808] | 391 | 0.29 |
| pooled_test/bigger | 0.818 [0.747, 0.880] | 176 | 0.40 |
| pooled_test/all | 0.840 [0.820, 0.861] | 1313 | 0.38 |
| pooled/closer | 0.843 [0.817, 0.869] | 747 | 0.40 |
| pooled/left_of | 0.819 [0.791, 0.847] | 779 | 0.35 |
| pooled/above | 0.745 [0.711, 0.776] | 466 | 0.25 |
| pooled/bigger | 0.938 [0.915, 0.959] | 569 | 0.50 |
| pooled/all | 0.840 [0.825, 0.855] | 2561 | 0.38 |
| naming/people | 0.974 [0.951, 0.992] | 230 | 0.47 |
| naming/indoor | 0.952 [0.924, 0.976] | 248 | 0.51 |
| naming/street | 0.853 [0.817, 0.886] | 400 | 0.48 |
| naming/pooled | 0.912 [0.892, 0.931] | 878 | 0.49 |

## SpaceQwen2.5-VL-3B


### Probes, all test frames (pre-registered)


**depth** (abs-rel, lower is better; 480 test objects in 195 images; layers chosen on training data: V 31, L_img 31, L_img_any 14, W 22, W_noimg 23, W_mis 26)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.381 [0.343, 0.420] |  |
| vision tower | 0.177 [0.158, 0.197] |  |
| image tokens, last quarter of language layers | 0.188 [0.167, 0.209] | 0.95 [0.90, 1.00] |
| image tokens, any language layer | 0.184 [0.166, 0.204] | 0.97 [0.92, 1.01] |
| object word, with the image | 0.213 [0.192, 0.234] | 0.83 [0.76, 0.89] |
| object word, no image | 0.363 [0.329, 0.400] | 0.08 [0.03, 0.13] |
| object word, mismatched image | 0.379 [0.342, 0.417] | 0.01 [-0.05, 0.07] |
| control: shuffled labels (tower layer) | 0.478 [0.432, 0.528] |  |
| control: norm-matched random features | 0.478 [0.432, 0.528] |  |

δ<1.25 (higher is better): name prior 0.404 [0.358, 0.449]; vision tower 0.723 [0.680, 0.766]; image tokens, last quarter of language layers 0.698 [0.652, 0.747]; image tokens, any language layer 0.713 [0.669, 0.757]; object word, with the image 0.658 [0.613, 0.700]; object word, no image 0.402 [0.358, 0.449]; object word, mismatched image 0.365 [0.318, 0.411]

G0 control: shuffled labels (tower layer): P minus control -0.098 [-0.134, -0.060] -> pass; control: norm-matched random features: P minus control -0.097 [-0.134, -0.060] -> pass. G1: tower gain over the prior 53.4% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 755 test objects in 310 images; layers chosen on training data: V 31, L_img 27, L_img_any 21, W 24, W_noimg 20, W_mis 17)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.22 [15.36, 17.09] |  |
| vision tower | 3.55 [3.39, 3.73] |  |
| image tokens, last quarter of language layers | 3.67 [3.49, 3.86] | 0.99 [0.98, 1.00] |
| image tokens, any language layer | 3.46 [3.26, 3.66] | 1.01 [0.99, 1.02] |
| object word, with the image | 6.73 [6.27, 7.20] | 0.75 [0.71, 0.78] |
| object word, no image | 16.11 [15.26, 16.96] | 0.01 [-0.01, 0.03] |
| object word, mismatched image | 16.13 [15.31, 16.96] | 0.01 [-0.01, 0.02] |
| control: shuffled labels (tower layer) | 18.14 [17.32, 18.99] |  |
| control: norm-matched random features | 18.12 [17.29, 18.96] |  |

G0 control: shuffled labels (tower layer): P minus control -1.923 [-2.304, -1.498] -> pass; control: norm-matched random features: P minus control -1.896 [-2.283, -1.474] -> pass. G1: tower gain over the prior 78.1% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 31, L_img 33, L_img_any 17, W 33, W_noimg 13, W_mis 7)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.379 [0.331, 0.431] |  |
| image tokens, last quarter of language layers | 0.385 [0.338, 0.435] | 0.93 [0.55, 1.36] |
| image tokens, any language layer | 0.382 [0.340, 0.426] | 0.97 [0.60, 1.52] |
| object word, with the image | 0.347 [0.304, 0.394] | 1.40 [0.96, 4.00] |
| object word, no image | 0.423 [0.369, 0.479] | 0.45 [0.05, 1.18] |
| object word, mismatched image | 0.435 [0.378, 0.493] | 0.31 [-0.13, 0.67] |
| control: shuffled labels (tower layer) | 0.646 [0.559, 0.740] |  |
| control: norm-matched random features | 0.645 [0.559, 0.740] |  |

G0 control: shuffled labels (tower layer): P minus control -0.187 [-0.260, -0.106] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.106] -> pass. G1: tower gain over the prior 17.4% -> FAIL. **Verdict: G1 failed: the tower does not beat the name prior by 20%.**

**centre** (3D error, m, lower is better; 480 test objects in 195 images; layers chosen on training data: V 31, L_img 31, L_img_any 19, W 23, W_noimg 19, W_mis 2)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.354 [1.281, 1.437] |  |
| vision tower | 0.651 [0.604, 0.701] |  |
| image tokens, last quarter of language layers | 0.631 [0.577, 0.684] | 1.03 [1.00, 1.07] |
| image tokens, any language layer | 0.612 [0.563, 0.662] | 1.06 [1.02, 1.09] |
| object word, with the image | 0.795 [0.746, 0.852] | 0.79 [0.74, 0.85] |
| object word, no image | 1.341 [1.266, 1.424] | 0.02 [-0.01, 0.04] |
| object word, mismatched image | 1.340 [1.267, 1.423] | 0.02 [0.01, 0.03] |
| control: shuffled labels (tower layer) | 1.647 [1.566, 1.734] |  |
| control: norm-matched random features | 1.647 [1.566, 1.734] |  |

G0 control: shuffled labels (tower layer): P minus control -0.293 [-0.353, -0.233] -> pass; control: norm-matched random features: P minus control -0.293 [-0.353, -0.233] -> pass. G1: tower gain over the prior 51.9% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.88 [0.83, 0.92] (n=272) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.79 [0.70, 0.87] (n=176) | 0.94 [0.93, 0.96] (n=1228) |
| image tokens, last quarter of language layers | 0.89 [0.84, 0.93] (n=272) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.82 [0.73, 0.90] (n=176) | 0.95 [0.93, 0.96] (n=1228) |
| object word, with the image | 0.85 [0.80, 0.89] (n=272) | 0.94 [0.91, 0.96] (n=399) | 0.96 [0.94, 0.98] (n=381) | 0.81 [0.73, 0.88] (n=176) | 0.91 [0.89, 0.93] (n=1228) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.636 [0.560, 0.725] (n=375); vision tower 0.412 [0.348, 0.484] (n=375); image tokens, last quarter of language layers 0.415 [0.352, 0.485] (n=375); object word, with the image 0.499 [0.429, 0.573] (n=375)

### Probes, without MMA and grappling frames


**depth** (abs-rel, lower is better; 452 test objects in 181 images; layers chosen on training data: V 31, L_img 31, L_img_any 14, W 22, W_noimg 23, W_mis 26)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.394 [0.355, 0.434] |  |
| vision tower | 0.182 [0.163, 0.203] |  |
| image tokens, last quarter of language layers | 0.195 [0.173, 0.217] | 0.94 [0.89, 0.99] |
| image tokens, any language layer | 0.191 [0.172, 0.212] | 0.96 [0.91, 1.00] |
| object word, with the image | 0.219 [0.198, 0.242] | 0.83 [0.76, 0.89] |
| object word, no image | 0.376 [0.339, 0.413] | 0.09 [0.03, 0.14] |
| object word, mismatched image | 0.392 [0.352, 0.432] | 0.01 [-0.05, 0.07] |
| control: shuffled labels (tower layer) | 0.488 [0.438, 0.541] |  |
| control: norm-matched random features | 0.488 [0.439, 0.541] |  |

δ<1.25 (higher is better): name prior 0.385 [0.339, 0.430]; vision tower 0.712 [0.668, 0.757]; image tokens, last quarter of language layers 0.684 [0.635, 0.733]; image tokens, any language layer 0.695 [0.649, 0.740]; object word, with the image 0.639 [0.593, 0.683]; object word, no image 0.383 [0.338, 0.428]; object word, mismatched image 0.345 [0.296, 0.393]

G0 control: shuffled labels (tower layer): P minus control -0.094 [-0.132, -0.055] -> pass; control: norm-matched random features: P minus control -0.094 [-0.132, -0.055] -> pass. G1: tower gain over the prior 53.7% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 727 test objects in 296 images; layers chosen on training data: V 31, L_img 27, L_img_any 21, W 24, W_noimg 20, W_mis 17)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.40 [15.50, 17.28] |  |
| vision tower | 3.55 [3.38, 3.73] |  |
| image tokens, last quarter of language layers | 3.70 [3.51, 3.90] | 0.99 [0.98, 1.00] |
| image tokens, any language layer | 3.51 [3.31, 3.73] | 1.00 [0.99, 1.02] |
| object word, with the image | 6.81 [6.33, 7.30] | 0.75 [0.71, 0.78] |
| object word, no image | 16.29 [15.42, 17.14] | 0.01 [-0.01, 0.03] |
| object word, mismatched image | 16.31 [15.47, 17.15] | 0.01 [-0.01, 0.02] |
| control: shuffled labels (tower layer) | 18.35 [17.50, 19.22] |  |
| control: norm-matched random features | 18.32 [17.47, 19.19] |  |

G0 control: shuffled labels (tower layer): P minus control -1.956 [-2.357, -1.524] -> pass; control: norm-matched random features: P minus control -1.924 [-2.329, -1.489] -> pass. G1: tower gain over the prior 78.3% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 31, L_img 33, L_img_any 17, W 33, W_noimg 13, W_mis 7)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.379 [0.331, 0.431] |  |
| image tokens, last quarter of language layers | 0.385 [0.338, 0.435] | 0.93 [0.55, 1.36] |
| image tokens, any language layer | 0.382 [0.340, 0.426] | 0.97 [0.60, 1.52] |
| object word, with the image | 0.347 [0.304, 0.394] | 1.40 [0.96, 4.00] |
| object word, no image | 0.423 [0.369, 0.479] | 0.45 [0.05, 1.18] |
| object word, mismatched image | 0.435 [0.378, 0.493] | 0.31 [-0.13, 0.67] |
| control: shuffled labels (tower layer) | 0.646 [0.559, 0.740] |  |
| control: norm-matched random features | 0.645 [0.559, 0.740] |  |

G0 control: shuffled labels (tower layer): P minus control -0.187 [-0.260, -0.106] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.106] -> pass. G1: tower gain over the prior 17.4% -> FAIL. **Verdict: G1 failed: the tower does not beat the name prior by 20%.**

**centre** (3D error, m, lower is better; 452 test objects in 181 images; layers chosen on training data: V 31, L_img 31, L_img_any 19, W 23, W_noimg 19, W_mis 2)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.355 [1.278, 1.441] |  |
| vision tower | 0.659 [0.609, 0.712] |  |
| image tokens, last quarter of language layers | 0.640 [0.584, 0.698] | 1.03 [0.99, 1.07] |
| image tokens, any language layer | 0.622 [0.572, 0.675] | 1.05 [1.02, 1.09] |
| object word, with the image | 0.798 [0.747, 0.858] | 0.80 [0.75, 0.86] |
| object word, no image | 1.340 [1.264, 1.425] | 0.02 [-0.01, 0.05] |
| object word, mismatched image | 1.340 [1.263, 1.426] | 0.02 [0.00, 0.04] |
| control: shuffled labels (tower layer) | 1.645 [1.562, 1.733] |  |
| control: norm-matched random features | 1.645 [1.562, 1.733] |  |

G0 control: shuffled labels (tower layer): P minus control -0.290 [-0.349, -0.230] -> pass; control: norm-matched random features: P minus control -0.290 [-0.349, -0.230] -> pass. G1: tower gain over the prior 51.3% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.88 [0.83, 0.92] (n=266) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.79 [0.70, 0.87] (n=176) | 0.94 [0.92, 0.96] (n=1222) |
| image tokens, last quarter of language layers | 0.88 [0.84, 0.93] (n=266) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.82 [0.73, 0.90] (n=176) | 0.95 [0.93, 0.96] (n=1222) |
| object word, with the image | 0.85 [0.81, 0.90] (n=266) | 0.94 [0.91, 0.96] (n=399) | 0.96 [0.94, 0.98] (n=381) | 0.81 [0.73, 0.88] (n=176) | 0.91 [0.89, 0.93] (n=1222) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.625 [0.550, 0.709] (n=361); vision tower 0.418 [0.351, 0.492] (n=361); image tokens, last quarter of language layers 0.418 [0.352, 0.490] (n=361); object word, with the image 0.500 [0.428, 0.577] (n=361)

### Asked answers, all frames (same objects for every column)

Answers: 1385 images, 1371 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 99% of 480 | 0.569 [0.542, 0.594] | 0.380 [0.342, 0.419] | 0.177 [0.158, 0.197] | 0.188 [0.167, 0.209] | 0.213 [0.192, 0.234] | MoGe-2 0.145 [0.127, 0.164] |
| depth_heldout | 100% of 856 | 0.676 [0.645, 0.709] | 0.897 [0.891, 0.902] | 0.764 [0.755, 0.772] | 0.720 [0.709, 0.731] | 0.868 [0.862, 0.873] | MoGe-2 0.103 [0.098, 0.108] |
| depth_test_indoor | 99% of 338 | 0.588 [0.557, 0.619] | 0.370 [0.329, 0.417] | 0.208 [0.184, 0.234] | 0.223 [0.196, 0.251] | 0.233 [0.206, 0.261] | MoGe-2 0.100 [0.087, 0.113] |
| depth_test_two_people | 100% of 142 | 0.522 [0.477, 0.566] | 0.402 [0.334, 0.477] | 0.104 [0.086, 0.123] | 0.106 [0.088, 0.125] | 0.166 [0.144, 0.191] | MoGe-2 0.253 [0.203, 0.300] |
| xy_test | 100% of 755 | 49.35 [43.31, 55.53] | 16.19 [15.34, 17.07] | 3.56 [3.39, 3.73] | 3.68 [3.49, 3.87] | 6.71 [6.23, 7.18] | RF-DETR 1.38 [1.01, 1.84] |
| xy_heldout | 100% of 856 | 79.90 [69.93, 90.02] | 18.41 [17.71, 19.10] | 5.52 [5.32, 5.72] | 5.80 [5.55, 6.07] | 11.46 [10.90, 12.04] |  |
| xy_test_people | 100% of 275 | 54.83 [45.04, 64.42] | 13.59 [12.52, 14.66] | 3.10 [2.85, 3.36] | 3.39 [3.11, 3.68] | 6.25 [5.65, 6.92] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 99% of 338 | 0.532 [0.468, 0.603] | 0.460 [0.389, 0.543] | 0.381 [0.332, 0.433] | 0.386 [0.339, 0.436] | 0.348 [0.304, 0.395] |  |
| size_heldout | 100% of 856 | 0.627 [0.609, 0.644] | 0.679 [0.666, 0.693] | 0.690 [0.672, 0.709] | 0.639 [0.620, 0.659] | 0.652 [0.638, 0.666] |  |
| distance_test | 61% of 375 | 0.672 [0.625, 0.716] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 61% of 512 | 0.784 [0.714, 0.860] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked -0.189 [-0.239, -0.138]; asked beats the prior by -50% -> qualitative (pre-registered rule); asked δ<1.25 0.088 [0.062, 0.116]; median asked/true 0.40
- depth_heldout: prior minus asked 0.221 [0.187, 0.253]; asked beats the prior by 25% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.111 [0.089, 0.134]; median asked/true 0.35
- depth_test_indoor: prior minus asked -0.218 [-0.272, -0.160]; asked beats the prior by -59% -> qualitative (pre-registered rule); asked δ<1.25 0.078 [0.048, 0.110]; median asked/true 0.38
- depth_test_two_people: prior minus asked -0.120 [-0.221, -0.007]; asked beats the prior by -30% -> qualitative (pre-registered rule); asked δ<1.25 0.113 [0.065, 0.169]; median asked/true 0.44
- xy_test: native boxes 4.36 [3.73, 5.08] (n=666, 88% returned); prior minus asked -33.160 [-39.115, -27.219]; asked beats the prior by -205% -> qualitative (pre-registered rule)
- xy_heldout: native boxes 6.42 [5.48, 7.43] (n=807, 94% returned); prior minus asked -61.491 [-71.458, -51.527]; asked beats the prior by -334% -> qualitative (pre-registered rule)
- xy_test_people: native boxes 1.88 [1.20, 2.76] (n=190, 69% returned); prior minus asked -41.236 [-50.806, -31.818]; asked beats the prior by -303% -> qualitative (pre-registered rule)
- size_test: prior minus asked -0.071 [-0.162, 0.030]; asked beats the prior by -16% -> qualitative (pre-registered rule)
- size_heldout: prior minus asked 0.053 [0.032, 0.073]; asked beats the prior by 8% -> qualitative (pre-registered rule)

### Asked answers, without MMA and grappling (same objects for every column)

Answers: 1385 images, 1371 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 99% of 452 | 0.569 [0.541, 0.595] | 0.393 [0.354, 0.433] | 0.182 [0.163, 0.203] | 0.194 [0.173, 0.217] | 0.220 [0.198, 0.242] | MoGe-2 0.137 [0.119, 0.156] |
| depth_heldout | 100% of 856 | 0.676 [0.645, 0.709] | 0.897 [0.891, 0.902] | 0.764 [0.755, 0.772] | 0.720 [0.709, 0.731] | 0.868 [0.862, 0.873] | MoGe-2 0.103 [0.098, 0.108] |
| depth_test_indoor | 99% of 338 | 0.588 [0.557, 0.619] | 0.370 [0.329, 0.417] | 0.208 [0.184, 0.234] | 0.223 [0.196, 0.251] | 0.233 [0.206, 0.261] | MoGe-2 0.100 [0.087, 0.113] |
| depth_test_two_people | 100% of 114 | 0.511 [0.458, 0.563] | 0.460 [0.380, 0.545] | 0.106 [0.085, 0.129] | 0.111 [0.091, 0.132] | 0.181 [0.154, 0.210] | MoGe-2 0.247 [0.194, 0.299] |
| xy_test | 100% of 727 | 50.77 [44.59, 57.19] | 16.37 [15.47, 17.26] | 3.56 [3.38, 3.74] | 3.71 [3.52, 3.90] | 6.79 [6.31, 7.29] | RF-DETR 1.38 [1.00, 1.83] |
| xy_heldout | 100% of 856 | 79.90 [69.93, 90.02] | 18.41 [17.71, 19.10] | 5.52 [5.32, 5.72] | 5.80 [5.55, 6.07] | 11.46 [10.90, 12.04] |  |
| xy_test_people | 100% of 275 | 54.83 [45.04, 64.42] | 13.59 [12.52, 14.66] | 3.10 [2.85, 3.36] | 3.39 [3.11, 3.68] | 6.25 [5.65, 6.92] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 99% of 338 | 0.532 [0.468, 0.603] | 0.460 [0.389, 0.543] | 0.381 [0.332, 0.433] | 0.386 [0.339, 0.436] | 0.348 [0.304, 0.395] |  |
| size_heldout | 100% of 856 | 0.627 [0.609, 0.644] | 0.679 [0.666, 0.693] | 0.690 [0.672, 0.709] | 0.639 [0.620, 0.659] | 0.652 [0.638, 0.666] |  |
| distance_test | 61% of 361 | 0.683 [0.636, 0.728] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 61% of 512 | 0.784 [0.714, 0.860] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked -0.176 [-0.226, -0.122]; asked beats the prior by -45% -> qualitative (pre-registered rule); asked δ<1.25 0.091 [0.065, 0.120]; median asked/true 0.40
- depth_heldout: prior minus asked 0.221 [0.187, 0.253]; asked beats the prior by 25% -> a measurement (>= 20%, CI excludes 0); asked δ<1.25 0.111 [0.089, 0.134]; median asked/true 0.35
- depth_test_indoor: prior minus asked -0.218 [-0.272, -0.160]; asked beats the prior by -59% -> qualitative (pre-registered rule); asked δ<1.25 0.078 [0.048, 0.110]; median asked/true 0.38
- depth_test_two_people: prior minus asked -0.051 [-0.172, 0.077]; asked beats the prior by -11% -> qualitative (pre-registered rule); asked δ<1.25 0.132 [0.075, 0.195]; median asked/true 0.45
- xy_test: native boxes 4.29 [3.63, 5.01] (n=638, 88% returned); prior minus asked -34.395 [-40.598, -28.363]; asked beats the prior by -210% -> qualitative (pre-registered rule)
- xy_heldout: native boxes 6.42 [5.48, 7.43] (n=807, 94% returned); prior minus asked -61.491 [-71.458, -51.527]; asked beats the prior by -334% -> qualitative (pre-registered rule)
- xy_test_people: native boxes 1.88 [1.20, 2.76] (n=190, 69% returned); prior minus asked -41.236 [-50.806, -31.818]; asked beats the prior by -303% -> qualitative (pre-registered rule)
- size_test: prior minus asked -0.071 [-0.162, 0.030]; asked beats the prior by -16% -> qualitative (pre-registered rule)
- size_heldout: prior minus asked 0.053 [0.032, 0.073]; asked beats the prior by 8% -> qualitative (pre-registered rule)

### Yes/no questions, all frames (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.812 [0.750, 0.870] | 142 | 0.32 |
| people/above | 0.622 [0.570, 0.673] | 144 | 0.14 |
| people/all | 0.711 [0.671, 0.750] | 286 | 0.23 |
| indoor/closer | 0.656 [0.603, 0.710] | 228 | 0.24 |
| indoor/left_of | 0.789 [0.741, 0.837] | 257 | 0.31 |
| indoor/above | 0.770 [0.721, 0.817] | 230 | 0.29 |
| indoor/bigger | 0.549 [0.517, 0.582] | 176 | 0.05 |
| indoor/all | 0.701 [0.677, 0.725] | 891 | 0.24 |
| two_people/closer | 0.671 [0.603, 0.736] | 140 | 0.24 |
| two_people/above | 0.500 [0.500, 0.500] | 20 (n<30: no gate) | 0.00 |
| two_people/all | 0.652 [0.588, 0.713] | 160 | 0.21 |
| street/closer | 0.519 [0.499, 0.540] | 400 | 0.04 |
| street/left_of | 0.783 [0.742, 0.824] | 380 | 0.31 |
| street/above | 0.515 [0.500, 0.553] | 75 | 0.01 |
| street/bigger | 0.528 [0.512, 0.546] | 393 | 0.03 |
| street/all | 0.603 [0.585, 0.620] | 1248 | 0.12 |
| pooled_test/closer | 0.661 [0.618, 0.702] | 368 | 0.24 |
| pooled_test/left_of | 0.797 [0.759, 0.834] | 399 | 0.31 |
| pooled_test/above | 0.699 [0.661, 0.734] | 394 | 0.22 |
| pooled_test/bigger | 0.549 [0.517, 0.582] | 176 | 0.05 |
| pooled_test/all | 0.697 [0.677, 0.716] | 1337 | 0.23 |
| pooled/closer | 0.592 [0.567, 0.617] | 768 | 0.14 |
| pooled/left_of | 0.790 [0.762, 0.819] | 779 | 0.31 |
| pooled/above | 0.673 [0.640, 0.706] | 469 | 0.19 |
| pooled/bigger | 0.534 [0.519, 0.551] | 569 | 0.04 |
| pooled/all | 0.652 [0.637, 0.667] | 2585 | 0.18 |
| naming/people | 0.909 [0.872, 0.941] | 230 | 0.41 |
| naming/indoor | 0.903 [0.867, 0.935] | 248 | 0.43 |
| naming/street | 0.715 [0.674, 0.758] | 400 | 0.30 |
| naming/pooled | 0.819 [0.793, 0.843] | 878 | 0.37 |

### Yes/no questions, without MMA and grappling (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.812 [0.750, 0.870] | 142 | 0.32 |
| people/above | 0.622 [0.570, 0.673] | 144 | 0.14 |
| people/all | 0.711 [0.671, 0.750] | 286 | 0.23 |
| indoor/closer | 0.656 [0.603, 0.710] | 228 | 0.24 |
| indoor/left_of | 0.789 [0.741, 0.837] | 257 | 0.31 |
| indoor/above | 0.770 [0.721, 0.817] | 230 | 0.29 |
| indoor/bigger | 0.549 [0.517, 0.582] | 176 | 0.05 |
| indoor/all | 0.701 [0.677, 0.725] | 891 | 0.24 |
| two_people/closer | 0.684 [0.606, 0.757] | 119 | 0.25 |
| two_people/above | 0.500 [0.500, 0.500] | 17 (n<30: no gate) | 0.00 |
| two_people/all | 0.657 [0.585, 0.728] | 136 | 0.22 |
| street/closer | 0.519 [0.499, 0.540] | 400 | 0.04 |
| street/left_of | 0.783 [0.742, 0.824] | 380 | 0.31 |
| street/above | 0.515 [0.500, 0.553] | 75 | 0.01 |
| street/bigger | 0.528 [0.512, 0.546] | 393 | 0.03 |
| street/all | 0.603 [0.585, 0.620] | 1248 | 0.12 |
| pooled_test/closer | 0.664 [0.620, 0.705] | 347 | 0.24 |
| pooled_test/left_of | 0.797 [0.759, 0.834] | 399 | 0.31 |
| pooled_test/above | 0.698 [0.661, 0.734] | 391 | 0.22 |
| pooled_test/bigger | 0.549 [0.517, 0.582] | 176 | 0.05 |
| pooled_test/all | 0.698 [0.678, 0.718] | 1313 | 0.23 |
| pooled/closer | 0.591 [0.566, 0.617] | 747 | 0.14 |
| pooled/left_of | 0.790 [0.762, 0.819] | 779 | 0.31 |
| pooled/above | 0.673 [0.640, 0.706] | 466 | 0.19 |
| pooled/bigger | 0.534 [0.519, 0.551] | 569 | 0.04 |
| pooled/all | 0.652 [0.637, 0.667] | 2561 | 0.18 |
| naming/people | 0.909 [0.872, 0.941] | 230 | 0.41 |
| naming/indoor | 0.903 [0.867, 0.935] | 248 | 0.43 |
| naming/street | 0.715 [0.674, 0.758] | 400 | 0.30 |
| naming/pooled | 0.819 [0.793, 0.843] | 878 | 0.37 |

## Qwen2.5-VL-3B


### Probes, all test frames (pre-registered)


**depth** (abs-rel, lower is better; 480 test objects in 195 images; layers chosen on training data: V 31, L_img 31, L_img_any 14, W 23, W_noimg 25, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.381 [0.343, 0.420] |  |
| vision tower | 0.177 [0.158, 0.197] |  |
| image tokens, last quarter of language layers | 0.185 [0.164, 0.205] | 0.96 [0.92, 1.01] |
| image tokens, any language layer | 0.184 [0.166, 0.203] | 0.97 [0.92, 1.01] |
| object word, with the image | 0.207 [0.187, 0.228] | 0.85 [0.79, 0.92] |
| object word, no image | 0.365 [0.330, 0.402] | 0.08 [0.03, 0.12] |
| object word, mismatched image | 0.371 [0.335, 0.408] | 0.05 [0.01, 0.08] |
| control: shuffled labels (tower layer) | 0.478 [0.432, 0.528] |  |
| control: norm-matched random features | 0.478 [0.432, 0.528] |  |

δ<1.25 (higher is better): name prior 0.404 [0.358, 0.449]; vision tower 0.727 [0.685, 0.769]; image tokens, last quarter of language layers 0.727 [0.683, 0.773]; image tokens, any language layer 0.710 [0.667, 0.754]; object word, with the image 0.654 [0.611, 0.698]; object word, no image 0.392 [0.347, 0.436]; object word, mismatched image 0.408 [0.359, 0.454]

G0 control: shuffled labels (tower layer): P minus control -0.098 [-0.134, -0.060] -> pass; control: norm-matched random features: P minus control -0.097 [-0.134, -0.060] -> pass. G1: tower gain over the prior 53.4% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 755 test objects in 310 images; layers chosen on training data: V 31, L_img 27, L_img_any 21, W 24, W_noimg 20, W_mis 17)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.22 [15.36, 17.09] |  |
| vision tower | 3.56 [3.39, 3.73] |  |
| image tokens, last quarter of language layers | 3.60 [3.43, 3.79] | 1.00 [0.98, 1.01] |
| image tokens, any language layer | 3.40 [3.21, 3.61] | 1.01 [1.00, 1.02] |
| object word, with the image | 6.45 [5.99, 6.92] | 0.77 [0.73, 0.80] |
| object word, no image | 16.13 [15.28, 16.98] | 0.01 [-0.01, 0.02] |
| object word, mismatched image | 16.13 [15.32, 16.96] | 0.01 [-0.01, 0.02] |
| control: shuffled labels (tower layer) | 18.14 [17.32, 18.99] |  |
| control: norm-matched random features | 18.12 [17.29, 18.96] |  |

G0 control: shuffled labels (tower layer): P minus control -1.923 [-2.304, -1.499] -> pass; control: norm-matched random features: P minus control -1.896 [-2.283, -1.474] -> pass. G1: tower gain over the prior 78.1% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 31, L_img 33, L_img_any 17, W 31, W_noimg 13, W_mis 7)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.379 [0.331, 0.430] |  |
| image tokens, last quarter of language layers | 0.381 [0.334, 0.429] | 0.97 [0.68, 1.52] |
| image tokens, any language layer | 0.379 [0.338, 0.424] | 0.99 [0.65, 1.53] |
| object word, with the image | 0.333 [0.295, 0.375] | 1.57 [1.10, 4.75] |
| object word, no image | 0.423 [0.370, 0.480] | 0.44 [0.05, 1.13] |
| object word, mismatched image | 0.434 [0.378, 0.493] | 0.31 [-0.10, 0.69] |
| control: shuffled labels (tower layer) | 0.646 [0.559, 0.741] |  |
| control: norm-matched random features | 0.645 [0.559, 0.740] |  |

G0 control: shuffled labels (tower layer): P minus control -0.187 [-0.260, -0.106] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 17.5% -> FAIL. **Verdict: G1 failed: the tower does not beat the name prior by 20%.**

**centre** (3D error, m, lower is better; 480 test objects in 195 images; layers chosen on training data: V 31, L_img 31, L_img_any 19, W 23, W_noimg 19, W_mis 2)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.354 [1.281, 1.437] |  |
| vision tower | 0.651 [0.603, 0.701] |  |
| image tokens, last quarter of language layers | 0.627 [0.574, 0.681] | 1.03 [1.00, 1.07] |
| image tokens, any language layer | 0.611 [0.562, 0.661] | 1.06 [1.03, 1.09] |
| object word, with the image | 0.788 [0.740, 0.844] | 0.80 [0.75, 0.86] |
| object word, no image | 1.345 [1.270, 1.428] | 0.01 [-0.01, 0.04] |
| object word, mismatched image | 1.340 [1.267, 1.423] | 0.02 [0.01, 0.03] |
| control: shuffled labels (tower layer) | 1.647 [1.566, 1.734] |  |
| control: norm-matched random features | 1.647 [1.566, 1.734] |  |

G0 control: shuffled labels (tower layer): P minus control -0.293 [-0.353, -0.233] -> pass; control: norm-matched random features: P minus control -0.293 [-0.353, -0.233] -> pass. G1: tower gain over the prior 51.9% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.88 [0.83, 0.92] (n=272) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.78 [0.69, 0.87] (n=176) | 0.94 [0.92, 0.96] (n=1228) |
| image tokens, last quarter of language layers | 0.88 [0.84, 0.92] (n=272) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.82 [0.73, 0.90] (n=176) | 0.95 [0.93, 0.96] (n=1228) |
| object word, with the image | 0.85 [0.80, 0.89] (n=272) | 0.95 [0.93, 0.97] (n=399) | 0.97 [0.95, 0.99] (n=381) | 0.85 [0.78, 0.92] (n=176) | 0.92 [0.91, 0.94] (n=1228) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.636 [0.560, 0.725] (n=375); vision tower 0.414 [0.349, 0.486] (n=375); image tokens, last quarter of language layers 0.419 [0.355, 0.490] (n=375); object word, with the image 0.493 [0.420, 0.571] (n=375)

### Probes, without MMA and grappling frames


**depth** (abs-rel, lower is better; 452 test objects in 181 images; layers chosen on training data: V 31, L_img 31, L_img_any 14, W 23, W_noimg 25, W_mis 3)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.394 [0.355, 0.434] |  |
| vision tower | 0.182 [0.163, 0.203] |  |
| image tokens, last quarter of language layers | 0.191 [0.170, 0.213] | 0.96 [0.91, 1.01] |
| image tokens, any language layer | 0.190 [0.171, 0.211] | 0.96 [0.92, 1.01] |
| object word, with the image | 0.214 [0.193, 0.236] | 0.85 [0.78, 0.92] |
| object word, no image | 0.378 [0.341, 0.415] | 0.08 [0.03, 0.13] |
| object word, mismatched image | 0.384 [0.347, 0.423] | 0.05 [0.01, 0.08] |
| control: shuffled labels (tower layer) | 0.488 [0.439, 0.541] |  |
| control: norm-matched random features | 0.488 [0.439, 0.541] |  |

δ<1.25 (higher is better): name prior 0.385 [0.339, 0.430]; vision tower 0.717 [0.674, 0.760]; image tokens, last quarter of language layers 0.712 [0.665, 0.759]; image tokens, any language layer 0.692 [0.647, 0.736]; object word, with the image 0.644 [0.600, 0.689]; object word, no image 0.372 [0.327, 0.416]; object word, mismatched image 0.387 [0.337, 0.434]

G0 control: shuffled labels (tower layer): P minus control -0.094 [-0.132, -0.055] -> pass; control: norm-matched random features: P minus control -0.094 [-0.132, -0.055] -> pass. G1: tower gain over the prior 53.7% -> pass. **Verdict: falsified (L_img, W keeps >= 0.8 of the tower gain).**

**xy** (% of diagonal, lower is better; 727 test objects in 296 images; layers chosen on training data: V 31, L_img 27, L_img_any 21, W 24, W_noimg 20, W_mis 17)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 16.40 [15.50, 17.28] |  |
| vision tower | 3.56 [3.38, 3.73] |  |
| image tokens, last quarter of language layers | 3.63 [3.45, 3.82] | 0.99 [0.98, 1.01] |
| image tokens, any language layer | 3.46 [3.26, 3.67] | 1.01 [1.00, 1.02] |
| object word, with the image | 6.54 [6.06, 7.04] | 0.77 [0.73, 0.80] |
| object word, no image | 16.30 [15.44, 17.15] | 0.01 [-0.01, 0.02] |
| object word, mismatched image | 16.31 [15.48, 17.15] | 0.01 [-0.01, 0.02] |
| control: shuffled labels (tower layer) | 18.35 [17.50, 19.22] |  |
| control: norm-matched random features | 18.32 [17.47, 19.19] |  |

G0 control: shuffled labels (tower layer): P minus control -1.956 [-2.357, -1.524] -> pass; control: norm-matched random features: P minus control -1.924 [-2.329, -1.489] -> pass. G1: tower gain over the prior 78.3% -> pass. **Verdict: partial.**

**size** (abs-rel, lower is better; 338 test objects in 124 images; layers chosen on training data: V 31, L_img 33, L_img_any 17, W 31, W_noimg 13, W_mis 7)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 0.459 [0.388, 0.541] |  |
| vision tower | 0.379 [0.331, 0.430] |  |
| image tokens, last quarter of language layers | 0.381 [0.334, 0.429] | 0.97 [0.68, 1.52] |
| image tokens, any language layer | 0.379 [0.338, 0.424] | 0.99 [0.65, 1.53] |
| object word, with the image | 0.333 [0.295, 0.375] | 1.57 [1.10, 4.75] |
| object word, no image | 0.423 [0.370, 0.480] | 0.44 [0.05, 1.13] |
| object word, mismatched image | 0.434 [0.378, 0.493] | 0.31 [-0.10, 0.69] |
| control: shuffled labels (tower layer) | 0.646 [0.559, 0.741] |  |
| control: norm-matched random features | 0.645 [0.559, 0.740] |  |

G0 control: shuffled labels (tower layer): P minus control -0.187 [-0.260, -0.106] -> pass; control: norm-matched random features: P minus control -0.186 [-0.258, -0.105] -> pass. G1: tower gain over the prior 17.5% -> FAIL. **Verdict: G1 failed: the tower does not beat the name prior by 20%.**

**centre** (3D error, m, lower is better; 452 test objects in 181 images; layers chosen on training data: V 31, L_img 31, L_img_any 19, W 23, W_noimg 19, W_mis 2)

| readout | error [95% CI] | R (share of the tower's gain over the prior) [95% CI] |
|---|---|---|
| name prior | 1.355 [1.278, 1.441] |  |
| vision tower | 0.659 [0.609, 0.712] |  |
| image tokens, last quarter of language layers | 0.635 [0.580, 0.693] | 1.03 [1.00, 1.07] |
| image tokens, any language layer | 0.621 [0.571, 0.675] | 1.05 [1.02, 1.09] |
| object word, with the image | 0.792 [0.741, 0.851] | 0.81 [0.76, 0.86] |
| object word, no image | 1.345 [1.268, 1.429] | 0.01 [-0.01, 0.04] |
| object word, mismatched image | 1.340 [1.262, 1.426] | 0.02 [0.01, 0.04] |
| control: shuffled labels (tower layer) | 1.645 [1.562, 1.733] |  |
| control: norm-matched random features | 1.645 [1.562, 1.733] |  |

G0 control: shuffled labels (tower layer): P minus control -0.290 [-0.349, -0.230] -> pass; control: norm-matched random features: P minus control -0.290 [-0.349, -0.230] -> pass. G1: tower gain over the prior 51.3% -> pass.

**Relations derived from the probes** (balanced accuracy, chance 0.5; test pairs)

| readout | closer | left of | above | bigger | pooled |
|---|---|---|---|---|---|
| vision tower | 0.88 [0.83, 0.92] (n=266) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.78 [0.69, 0.87] (n=176) | 0.94 [0.92, 0.96] (n=1222) |
| image tokens, last quarter of language layers | 0.88 [0.83, 0.92] (n=266) | 0.99 [0.97, 1.00] (n=399) | 0.99 [0.98, 1.00] (n=381) | 0.82 [0.73, 0.90] (n=176) | 0.95 [0.93, 0.96] (n=1222) |
| object word, with the image | 0.86 [0.81, 0.90] (n=266) | 0.95 [0.93, 0.97] (n=399) | 0.97 [0.95, 0.99] (n=381) | 0.85 [0.78, 0.92] (n=176) | 0.93 [0.91, 0.94] (n=1222) |

**Distance between two objects from the probes' 3D centres** (abs-rel; test pairs): name prior 0.625 [0.550, 0.709] (n=361); vision tower 0.419 [0.352, 0.493] (n=361); image tokens, last quarter of language layers 0.422 [0.356, 0.494] (n=361); object word, with the image 0.495 [0.422, 0.575] (n=361)

### Asked answers, all frames (same objects for every column)

Answers: 1385 images, 1249 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 93% of 480 | 0.546 [0.510, 0.580] | 0.384 [0.346, 0.425] | 0.177 [0.157, 0.198] | 0.183 [0.162, 0.205] | 0.208 [0.186, 0.230] | MoGe-2 0.151 [0.132, 0.171] |
| depth_heldout | 99% of 856 | 1.020 [0.905, 1.143] | 0.897 [0.891, 0.902] | 0.765 [0.756, 0.773] | 0.705 [0.693, 0.716] | 0.868 [0.862, 0.873] | MoGe-2 0.103 [0.098, 0.109] |
| depth_test_indoor | 90% of 338 | 0.635 [0.603, 0.665] | 0.376 [0.334, 0.426] | 0.211 [0.185, 0.239] | 0.221 [0.193, 0.250] | 0.229 [0.201, 0.260] | MoGe-2 0.104 [0.091, 0.118] |
| depth_test_two_people | 100% of 142 | 0.354 [0.294, 0.428] | 0.402 [0.334, 0.477] | 0.104 [0.087, 0.122] | 0.103 [0.086, 0.121] | 0.162 [0.139, 0.188] | MoGe-2 0.253 [0.203, 0.300] |
| xy_test | 92% of 755 | 170.53 [163.17, 177.51] | 15.63 [14.73, 16.54] | 3.52 [3.35, 3.70] | 3.54 [3.35, 3.73] | 6.21 [5.77, 6.66] | RF-DETR 1.38 [1.01, 1.84] |
| xy_heldout | 99% of 856 | 223.45 [216.54, 230.14] | 18.41 [17.70, 19.10] | 5.49 [5.29, 5.69] | 5.89 [5.65, 6.15] | 11.45 [10.89, 12.03] |  |
| xy_test_people | 91% of 275 | 139.23 [128.68, 149.94] | 12.95 [11.83, 14.04] | 3.06 [2.81, 3.32] | 3.27 [3.00, 3.56] | 5.76 [5.19, 6.37] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 90% of 338 | 0.574 [0.506, 0.643] | 0.465 [0.390, 0.555] | 0.391 [0.341, 0.448] | 0.391 [0.341, 0.443] | 0.341 [0.301, 0.384] |  |
| size_heldout | 99% of 856 | 0.574 [0.497, 0.684] | 0.682 [0.669, 0.696] | 0.691 [0.674, 0.710] | 0.627 [0.608, 0.646] | 0.690 [0.677, 0.703] |  |
| distance_test | 93% of 375 | 0.621 [0.549, 0.701] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 90% of 512 | 1.589 [1.353, 1.853] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked -0.161 [-0.214, -0.106]; asked beats the prior by -42% -> qualitative (pre-registered rule); asked δ<1.25 0.159 [0.123, 0.197]; median asked/true 0.48
- depth_heldout: prior minus asked -0.123 [-0.247, -0.007]; asked beats the prior by -14% -> qualitative (pre-registered rule); asked δ<1.25 0.163 [0.138, 0.190]; median asked/true 1.24
- depth_test_indoor: prior minus asked -0.259 [-0.318, -0.193]; asked beats the prior by -69% -> qualitative (pre-registered rule); asked δ<1.25 0.049 [0.026, 0.073]; median asked/true 0.33
- depth_test_two_people: prior minus asked 0.048 [-0.043, 0.137]; asked beats the prior by 12% -> qualitative (pre-registered rule); asked δ<1.25 0.394 [0.306, 0.478]; median asked/true 1.02
- xy_test: native boxes 4.96 [4.08, 5.90] (n=521, 69% returned); prior minus asked -154.904 [-161.699, -147.590]; asked beats the prior by -991% -> qualitative (pre-registered rule)
- xy_heldout: native boxes 4.94 [4.09, 5.78] (n=765, 89% returned); prior minus asked -205.036 [-211.984, -197.964]; asked beats the prior by -1114% -> qualitative (pre-registered rule)
- xy_test_people: native boxes 1.78 [0.95, 2.73] (n=61, 22% returned); prior minus asked -126.279 [-137.328, -115.584]; asked beats the prior by -975% -> qualitative (pre-registered rule)
- size_test: prior minus asked -0.109 [-0.205, -0.004]; asked beats the prior by -23% -> qualitative (pre-registered rule)
- size_heldout: prior minus asked 0.108 [-0.002, 0.189]; asked beats the prior by 16% -> qualitative (pre-registered rule)

### Asked answers, without MMA and grappling (same objects for every column)

Answers: 1385 images, 1249 parsed as whole JSON.

| quantity / split | answered | asked: text | name prior | probe: tower | probe: late image tokens | probe: word | specialist |
|---|---|---|---|---|---|---|---|
| depth_test | 92% of 452 | 0.566 [0.531, 0.602] | 0.399 [0.360, 0.442] | 0.182 [0.161, 0.205] | 0.190 [0.168, 0.214] | 0.215 [0.193, 0.239] | MoGe-2 0.143 [0.124, 0.162] |
| depth_heldout | 99% of 856 | 1.020 [0.905, 1.143] | 0.897 [0.891, 0.902] | 0.765 [0.756, 0.773] | 0.705 [0.693, 0.716] | 0.868 [0.862, 0.873] | MoGe-2 0.103 [0.098, 0.109] |
| depth_test_indoor | 90% of 338 | 0.635 [0.603, 0.665] | 0.376 [0.334, 0.426] | 0.211 [0.185, 0.239] | 0.221 [0.193, 0.250] | 0.229 [0.201, 0.260] | MoGe-2 0.104 [0.091, 0.118] |
| depth_test_two_people | 100% of 114 | 0.382 [0.312, 0.469] | 0.460 [0.380, 0.545] | 0.106 [0.086, 0.128] | 0.108 [0.089, 0.129] | 0.177 [0.151, 0.206] | MoGe-2 0.247 [0.194, 0.299] |
| xy_test | 92% of 727 | 168.20 [161.01, 175.45] | 15.80 [14.87, 16.73] | 3.52 [3.34, 3.70] | 3.57 [3.37, 3.77] | 6.29 [5.84, 6.76] | RF-DETR 1.38 [1.00, 1.83] |
| xy_heldout | 99% of 856 | 223.45 [216.54, 230.14] | 18.41 [17.70, 19.10] | 5.49 [5.29, 5.69] | 5.89 [5.65, 6.15] | 11.45 [10.89, 12.03] |  |
| xy_test_people | 91% of 275 | 139.23 [128.68, 149.94] | 12.95 [11.83, 14.04] | 3.06 [2.81, 3.32] | 3.27 [3.00, 3.56] | 5.76 [5.19, 6.37] | RF-DETR 1.34 [0.88, 1.94] |
| size_test | 90% of 338 | 0.574 [0.506, 0.643] | 0.465 [0.390, 0.555] | 0.391 [0.341, 0.448] | 0.391 [0.341, 0.443] | 0.341 [0.301, 0.384] |  |
| size_heldout | 99% of 856 | 0.574 [0.497, 0.684] | 0.682 [0.669, 0.696] | 0.691 [0.674, 0.710] | 0.627 [0.608, 0.646] | 0.690 [0.677, 0.703] |  |
| distance_test | 93% of 361 | 0.551 [0.508, 0.595] | n/a | n/a | n/a | n/a |  |
| distance_heldout | 90% of 512 | 1.589 [1.353, 1.853] | n/a | n/a | n/a | n/a |  |
- depth_test: prior minus asked -0.167 [-0.221, -0.110]; asked beats the prior by -42% -> qualitative (pre-registered rule); asked δ<1.25 0.144 [0.108, 0.182]; median asked/true 0.45
- depth_heldout: prior minus asked -0.123 [-0.247, -0.007]; asked beats the prior by -14% -> qualitative (pre-registered rule); asked δ<1.25 0.163 [0.138, 0.190]; median asked/true 1.24
- depth_test_indoor: prior minus asked -0.259 [-0.318, -0.193]; asked beats the prior by -69% -> qualitative (pre-registered rule); asked δ<1.25 0.049 [0.026, 0.073]; median asked/true 0.33
- depth_test_two_people: prior minus asked 0.079 [-0.028, 0.186]; asked beats the prior by 17% -> qualitative (pre-registered rule); asked δ<1.25 0.395 [0.300, 0.484]; median asked/true 1.03
- xy_test: native boxes 4.85 [3.97, 5.80] (n=493, 68% returned); prior minus asked -152.403 [-159.721, -145.239]; asked beats the prior by -965% -> qualitative (pre-registered rule)
- xy_heldout: native boxes 4.94 [4.09, 5.78] (n=765, 89% returned); prior minus asked -205.036 [-211.984, -197.964]; asked beats the prior by -1114% -> qualitative (pre-registered rule)
- xy_test_people: native boxes 1.78 [0.95, 2.73] (n=61, 22% returned); prior minus asked -126.279 [-137.328, -115.584]; asked beats the prior by -975% -> qualitative (pre-registered rule)
- size_test: prior minus asked -0.109 [-0.205, -0.004]; asked beats the prior by -23% -> qualitative (pre-registered rule)
- size_heldout: prior minus asked 0.108 [-0.002, 0.189]; asked beats the prior by 16% -> qualitative (pre-registered rule)

### Yes/no questions, all frames (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.931 [0.888, 0.967] | 142 | 0.44 |
| people/above | 0.789 [0.721, 0.853] | 144 | 0.51 |
| people/all | 0.861 [0.822, 0.897] | 286 | 0.47 |
| indoor/closer | 0.736 [0.677, 0.793] | 228 | 0.70 |
| indoor/left_of | 0.903 [0.859, 0.942] | 257 | 0.46 |
| indoor/above | 0.774 [0.716, 0.829] | 230 | 0.62 |
| indoor/bigger | 0.744 [0.674, 0.812] | 176 | 0.40 |
| indoor/all | 0.798 [0.769, 0.827] | 891 | 0.55 |
| two_people/closer | 0.575 [0.522, 0.629] | 140 | 0.86 |
| two_people/above | 0.455 [0.350, 0.500] | 20 (n<30: no gate) | 0.05 |
| two_people/all | 0.566 [0.501, 0.633] | 160 | 0.76 |
| street/closer | 0.646 [0.595, 0.695] | 400 | 0.36 |
| street/left_of | 0.871 [0.837, 0.903] | 380 | 0.44 |
| street/above | 0.602 [0.497, 0.701] | 75 | 0.28 |
| street/bigger | 0.872 [0.839, 0.902] | 393 | 0.44 |
| street/all | 0.785 [0.762, 0.807] | 1248 | 0.40 |
| pooled_test/closer | 0.668 [0.627, 0.712] | 368 | 0.76 |
| pooled_test/left_of | 0.912 [0.881, 0.941] | 399 | 0.45 |
| pooled_test/above | 0.762 [0.716, 0.806] | 394 | 0.55 |
| pooled_test/bigger | 0.744 [0.674, 0.812] | 176 | 0.40 |
| pooled_test/all | 0.781 [0.757, 0.806] | 1337 | 0.56 |
| pooled/closer | 0.667 [0.629, 0.703] | 768 | 0.55 |
| pooled/left_of | 0.892 [0.869, 0.914] | 779 | 0.44 |
| pooled/above | 0.742 [0.699, 0.782] | 469 | 0.51 |
| pooled/bigger | 0.832 [0.800, 0.863] | 569 | 0.43 |
| pooled/all | 0.783 [0.766, 0.800] | 2585 | 0.48 |
| naming/people | 0.957 [0.925, 0.982] | 230 | 0.47 |
| naming/indoor | 0.927 [0.894, 0.958] | 248 | 0.51 |
| naming/street | 0.762 [0.720, 0.804] | 400 | 0.45 |
| naming/pooled | 0.860 [0.835, 0.884] | 878 | 0.47 |

### Yes/no questions, without MMA and grappling (balanced accuracy for relations, accuracy for naming; chance 0.5)

| set | score [95% CI] | n | share answered yes |
|---|---|---|---|
| people/left_of | 0.931 [0.888, 0.967] | 142 | 0.44 |
| people/above | 0.789 [0.721, 0.853] | 144 | 0.51 |
| people/all | 0.861 [0.822, 0.897] | 286 | 0.47 |
| indoor/closer | 0.736 [0.677, 0.793] | 228 | 0.70 |
| indoor/left_of | 0.903 [0.859, 0.942] | 257 | 0.46 |
| indoor/above | 0.774 [0.716, 0.829] | 230 | 0.62 |
| indoor/bigger | 0.744 [0.674, 0.812] | 176 | 0.40 |
| indoor/all | 0.798 [0.769, 0.827] | 891 | 0.55 |
| two_people/closer | 0.571 [0.530, 0.615] | 119 | 0.92 |
| two_people/above | 0.438 [0.300, 0.500] | 17 (n<30: no gate) | 0.06 |
| two_people/all | 0.543 [0.474, 0.612] | 136 | 0.82 |
| street/closer | 0.646 [0.595, 0.695] | 400 | 0.36 |
| street/left_of | 0.871 [0.837, 0.903] | 380 | 0.44 |
| street/above | 0.602 [0.497, 0.701] | 75 | 0.28 |
| street/bigger | 0.872 [0.839, 0.902] | 393 | 0.44 |
| street/all | 0.785 [0.762, 0.807] | 1248 | 0.40 |
| pooled_test/closer | 0.670 [0.627, 0.713] | 347 | 0.78 |
| pooled_test/left_of | 0.912 [0.881, 0.941] | 399 | 0.45 |
| pooled_test/above | 0.760 [0.713, 0.804] | 391 | 0.55 |
| pooled_test/bigger | 0.744 [0.674, 0.812] | 176 | 0.40 |
| pooled_test/all | 0.782 [0.759, 0.806] | 1313 | 0.56 |
| pooled/closer | 0.667 [0.629, 0.703] | 747 | 0.55 |
| pooled/left_of | 0.892 [0.869, 0.914] | 779 | 0.44 |
| pooled/above | 0.740 [0.697, 0.781] | 466 | 0.51 |
| pooled/bigger | 0.832 [0.800, 0.863] | 569 | 0.43 |
| pooled/all | 0.784 [0.766, 0.801] | 2561 | 0.48 |
| naming/people | 0.957 [0.925, 0.982] | 230 | 0.47 |
| naming/indoor | 0.927 [0.894, 0.958] | 248 | 0.51 |
| naming/street | 0.762 [0.720, 0.804] | 400 | 0.45 |
| naming/pooled | 0.860 [0.835, 0.884] | 878 | 0.47 |

## Paired model comparisons (asked; same items; difference = first minus second)


**SpatialAxiom-9B vs Qwen3.5-9B**, all frames

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.233 [0.208, 0.264] | 0.350 [0.313, 0.387] | -0.117 [-0.153, -0.080] (n=467) |
| depth_heldout | 0.261 [0.241, 0.283] | 0.359 [0.325, 0.394] | -0.099 [-0.140, -0.057] (n=854) |
| size_test | 0.333 [0.297, 0.369] | 0.349 [0.304, 0.399] | -0.016 [-0.058, 0.020] (n=325) |
| size_heldout | 0.239 [0.224, 0.253] | 0.213 [0.192, 0.237] | 0.025 [0.004, 0.045] (n=854) |
| xy_test | 366.09 [343.19, 387.74] | 406.05 [388.39, 423.07] | -39.96 [-61.52, -19.55] (n=751) |
| xy_rescaled_exploratory_test | 13.25 [10.86, 15.69] | 8.37 [7.20, 9.60] | 4.88 [2.74, 7.14] (n=751) |
| xy_heldout | 317.52 [297.93, 336.11] | 343.88 [325.87, 362.27] | -26.37 [-47.60, -4.87] (n=854) |
| xy_rescaled_exploratory_heldout | 5.69 [4.96, 6.44] | 6.04 [5.16, 7.01] | -0.35 [-1.05, 0.31] (n=854) |
| relation | 0.923 [0.912, 0.934] | 0.838 [0.823, 0.853] | 0.085 [0.069, 0.100] (n=2585) |
| naming | 0.884 [0.861, 0.905] | 0.912 [0.892, 0.931] | -0.028 [-0.045, -0.011] (n=878) |

**SpatialAxiom-9B vs Qwen3.5-9B**, without MMA and grappling

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.233 [0.207, 0.266] | 0.344 [0.305, 0.385] | -0.111 [-0.149, -0.071] (n=439) |
| depth_heldout | 0.261 [0.241, 0.283] | 0.359 [0.325, 0.394] | -0.099 [-0.140, -0.057] (n=854) |
| size_test | 0.333 [0.297, 0.369] | 0.349 [0.304, 0.399] | -0.016 [-0.058, 0.020] (n=325) |
| size_heldout | 0.239 [0.224, 0.253] | 0.213 [0.192, 0.237] | 0.025 [0.004, 0.045] (n=854) |
| xy_test | 364.66 [340.53, 387.05] | 407.04 [389.18, 424.83] | -42.38 [-64.13, -21.49] (n=723) |
| xy_rescaled_exploratory_test | 13.38 [10.96, 15.93] | 8.37 [7.18, 9.66] | 5.01 [2.77, 7.37] (n=723) |
| xy_heldout | 317.52 [297.93, 336.11] | 343.88 [325.87, 362.27] | -26.37 [-47.60, -4.87] (n=854) |
| xy_rescaled_exploratory_heldout | 5.69 [4.96, 6.44] | 6.04 [5.16, 7.01] | -0.35 [-1.05, 0.31] (n=854) |
| relation | 0.924 [0.912, 0.936] | 0.840 [0.825, 0.855] | 0.084 [0.068, 0.099] (n=2561) |
| naming | 0.884 [0.861, 0.905] | 0.912 [0.892, 0.931] | -0.028 [-0.045, -0.011] (n=878) |

**SpatialAxiom-9B vs Qwen3.8-27B**, all frames

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.233 [0.208, 0.264] | 0.381 [0.349, 0.413] | -0.147 [-0.179, -0.114] (n=467) |
| depth_heldout | 0.261 [0.242, 0.283] | 0.199 [0.186, 0.211] | 0.062 [0.046, 0.081] (n=852) |
| size_test | 0.333 [0.297, 0.369] | 0.385 [0.341, 0.432] | -0.052 [-0.086, -0.019] (n=325) |
| size_heldout | 0.238 [0.224, 0.253] | 0.158 [0.146, 0.170] | 0.081 [0.069, 0.094] (n=852) |
| xy_test | 366.09 [343.19, 387.74] | 165.25 [139.47, 192.46] | 200.84 [164.41, 236.62] (n=751) |
| xy_rescaled_exploratory_test | 13.25 [10.86, 15.69] | 3.12 [2.65, 3.64] | 10.12 [7.91, 12.44] (n=751) |
| xy_heldout | 317.22 [297.55, 335.77] | 4.23 [3.45, 5.01] | 312.99 [293.57, 331.53] (n=852) |
| xy_rescaled_exploratory_heldout | 5.68 [4.94, 6.43] | 4.23 [3.45, 5.01] | 1.46 [0.88, 2.08] (n=852) |
| relation | 0.923 [0.912, 0.934] | 0.835 [0.821, 0.850] | 0.088 [0.073, 0.104] (n=2585) |
| naming | 0.884 [0.861, 0.905] | 0.885 [0.863, 0.905] | -0.001 [-0.021, 0.019] (n=878) |

**SpatialAxiom-9B vs Qwen3.8-27B**, without MMA and grappling

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.233 [0.207, 0.266] | 0.373 [0.341, 0.408] | -0.140 [-0.173, -0.106] (n=439) |
| depth_heldout | 0.261 [0.242, 0.283] | 0.199 [0.186, 0.211] | 0.062 [0.046, 0.081] (n=852) |
| size_test | 0.333 [0.297, 0.369] | 0.385 [0.341, 0.432] | -0.052 [-0.086, -0.019] (n=325) |
| size_heldout | 0.238 [0.224, 0.253] | 0.158 [0.146, 0.170] | 0.081 [0.069, 0.094] (n=852) |
| xy_test | 364.66 [340.53, 387.05] | 171.48 [144.71, 200.08] | 193.19 [155.99, 229.71] (n=723) |
| xy_rescaled_exploratory_test | 13.38 [10.96, 15.93] | 3.07 [2.62, 3.58] | 10.30 [8.06, 12.68] (n=723) |
| xy_heldout | 317.22 [297.55, 335.77] | 4.23 [3.45, 5.01] | 312.99 [293.57, 331.53] (n=852) |
| xy_rescaled_exploratory_heldout | 5.68 [4.94, 6.43] | 4.23 [3.45, 5.01] | 1.46 [0.88, 2.08] (n=852) |
| relation | 0.924 [0.912, 0.936] | 0.836 [0.822, 0.851] | 0.088 [0.072, 0.104] (n=2561) |
| naming | 0.884 [0.861, 0.905] | 0.885 [0.863, 0.905] | -0.001 [-0.021, 0.019] (n=878) |

**SpaceQwen2.5-VL-3B vs Qwen2.5-VL-3B**, all frames

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.567 [0.540, 0.592] | 0.545 [0.510, 0.580] | 0.021 [-0.017, 0.061] (n=443) |
| depth_heldout | 0.676 [0.644, 0.708] | 1.020 [0.905, 1.143] | -0.344 [-0.468, -0.229] (n=846) |
| size_test | 0.515 [0.452, 0.580] | 0.576 [0.510, 0.647] | -0.061 [-0.144, 0.024] (n=301) |
| size_heldout | 0.625 [0.608, 0.643] | 0.574 [0.497, 0.684] | 0.051 [-0.058, 0.130] (n=846) |
| xy_test | 49.75 [43.29, 56.38] | 170.23 [162.87, 177.24] | -120.48 [-129.11, -111.83] (n=694) |
| xy_heldout | 80.15 [70.14, 90.32] | 223.45 [216.54, 230.14] | -143.29 [-153.41, -133.09] (n=846) |
| relation | 0.652 [0.637, 0.667] | 0.783 [0.766, 0.800] | -0.131 [-0.150, -0.114] (n=2585) |
| naming | 0.819 [0.793, 0.843] | 0.860 [0.835, 0.884] | -0.041 [-0.061, -0.021] (n=878) |

**SpaceQwen2.5-VL-3B vs Qwen2.5-VL-3B**, without MMA and grappling

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.567 [0.538, 0.594] | 0.566 [0.531, 0.602] | 0.001 [-0.038, 0.041] (n=415) |
| depth_heldout | 0.676 [0.644, 0.708] | 1.020 [0.905, 1.143] | -0.344 [-0.468, -0.229] (n=846) |
| size_test | 0.515 [0.452, 0.580] | 0.576 [0.510, 0.647] | -0.061 [-0.144, 0.024] (n=301) |
| size_heldout | 0.625 [0.608, 0.643] | 0.574 [0.497, 0.684] | 0.051 [-0.058, 0.130] (n=846) |
| xy_test | 51.30 [44.64, 58.12] | 167.87 [160.59, 175.13] | -116.57 [-125.19, -107.80] (n=666) |
| xy_heldout | 80.15 [70.14, 90.32] | 223.45 [216.54, 230.14] | -143.29 [-153.41, -133.09] (n=846) |
| relation | 0.652 [0.637, 0.667] | 0.784 [0.766, 0.801] | -0.132 [-0.150, -0.114] (n=2561) |
| naming | 0.819 [0.793, 0.843] | 0.860 [0.835, 0.884] | -0.041 [-0.061, -0.021] (n=878) |

**SpaceQwen2.5-VL-3B vs Qwen3.8-27B**, all frames

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.569 [0.542, 0.594] | 0.378 [0.347, 0.411] | 0.190 [0.143, 0.237] (n=477) |
| depth_heldout | 0.675 [0.645, 0.708] | 0.199 [0.186, 0.211] | 0.477 [0.442, 0.513] (n=854) |
| size_test | 0.532 [0.468, 0.603] | 0.384 [0.342, 0.429] | 0.148 [0.078, 0.221] (n=335) |
| size_heldout | 0.626 [0.609, 0.644] | 0.158 [0.146, 0.170] | 0.469 [0.449, 0.488] (n=854) |
| xy_test | 49.35 [43.31, 55.53] | 167.64 [140.98, 195.90] | -118.29 [-145.19, -91.89] (n=752) |
| xy_heldout | 79.51 [69.76, 89.45] | 4.23 [3.46, 5.01] | 75.28 [65.46, 85.12] (n=854) |
| relation | 0.652 [0.637, 0.667] | 0.835 [0.821, 0.850] | -0.183 [-0.202, -0.167] (n=2585) |
| naming | 0.819 [0.793, 0.843] | 0.885 [0.863, 0.905] | -0.066 [-0.089, -0.043] (n=878) |

**SpaceQwen2.5-VL-3B vs Qwen3.8-27B**, without MMA and grappling

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.569 [0.541, 0.595] | 0.371 [0.338, 0.405] | 0.198 [0.148, 0.245] (n=449) |
| depth_heldout | 0.675 [0.645, 0.708] | 0.199 [0.186, 0.211] | 0.477 [0.442, 0.513] (n=854) |
| size_test | 0.532 [0.468, 0.603] | 0.384 [0.342, 0.429] | 0.148 [0.078, 0.221] (n=335) |
| size_heldout | 0.626 [0.609, 0.644] | 0.158 [0.146, 0.170] | 0.469 [0.449, 0.488] (n=854) |
| xy_test | 50.77 [44.59, 57.19] | 173.95 [146.28, 202.58] | -123.19 [-150.85, -96.20] (n=724) |
| xy_heldout | 79.51 [69.76, 89.45] | 4.23 [3.46, 5.01] | 75.28 [65.46, 85.12] (n=854) |
| relation | 0.652 [0.637, 0.667] | 0.836 [0.822, 0.851] | -0.184 [-0.203, -0.167] (n=2561) |
| naming | 0.819 [0.793, 0.843] | 0.885 [0.863, 0.905] | -0.066 [-0.089, -0.043] (n=878) |

**Qwen3.5-9B vs Qwen3.8-27B**, all frames

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.347 [0.311, 0.385] | 0.377 [0.346, 0.410] | -0.030 [-0.055, -0.004] (n=480) |
| depth_heldout | 0.360 [0.325, 0.395] | 0.199 [0.186, 0.211] | 0.161 [0.125, 0.197] (n=852) |
| size_test | 0.349 [0.305, 0.396] | 0.385 [0.343, 0.430] | -0.036 [-0.079, 0.006] (n=338) |
| size_heldout | 0.214 [0.192, 0.237] | 0.158 [0.146, 0.170] | 0.056 [0.038, 0.077] (n=852) |
| xy_test | 406.74 [388.95, 423.84] | 166.98 [140.17, 195.17] | 239.76 [209.46, 270.01] (n=755) |
| xy_rescaled_exploratory_test | 8.38 [7.22, 9.63] | 3.14 [2.66, 3.66] | 5.24 [4.15, 6.39] (n=755) |
| xy_heldout | 344.22 [326.02, 362.65] | 4.23 [3.45, 5.01] | 339.99 [322.02, 358.26] (n=852) |
| xy_rescaled_exploratory_heldout | 6.03 [5.16, 7.01] | 4.23 [3.45, 5.01] | 1.80 [1.13, 2.58] (n=852) |
| relation | 0.838 [0.823, 0.853] | 0.835 [0.821, 0.850] | 0.003 [-0.011, 0.015] (n=2585) |
| naming | 0.912 [0.892, 0.931] | 0.885 [0.863, 0.905] | 0.027 [0.010, 0.046] (n=878) |

**Qwen3.5-9B vs Qwen3.8-27B**, without MMA and grappling

| measure | first | second | difference [95% CI] |
|---|---|---|---|
| depth_test | 0.341 [0.303, 0.381] | 0.370 [0.337, 0.404] | -0.029 [-0.055, -0.002] (n=452) |
| depth_heldout | 0.360 [0.325, 0.395] | 0.199 [0.186, 0.211] | 0.161 [0.125, 0.197] (n=852) |
| size_test | 0.349 [0.305, 0.396] | 0.385 [0.343, 0.430] | -0.036 [-0.079, 0.006] (n=338) |
| size_heldout | 0.214 [0.192, 0.237] | 0.158 [0.146, 0.170] | 0.056 [0.038, 0.077] (n=852) |
| xy_test | 407.76 [390.10, 425.05] | 173.24 [145.69, 202.12] | 234.52 [203.10, 265.99] (n=727) |
| xy_rescaled_exploratory_test | 8.38 [7.18, 9.66] | 3.09 [2.63, 3.60] | 5.29 [4.19, 6.43] (n=727) |
| xy_heldout | 344.22 [326.02, 362.65] | 4.23 [3.45, 5.01] | 339.99 [322.02, 358.26] (n=852) |
| xy_rescaled_exploratory_heldout | 6.03 [5.16, 7.01] | 4.23 [3.45, 5.01] | 1.80 [1.13, 2.58] (n=852) |
| relation | 0.840 [0.825, 0.855] | 0.836 [0.822, 0.851] | 0.004 [-0.009, 0.017] (n=2561) |
| naming | 0.912 [0.892, 0.931] | 0.885 [0.863, 0.905] | 0.027 [0.010, 0.046] (n=878) |
