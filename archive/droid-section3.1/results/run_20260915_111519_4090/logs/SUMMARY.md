# Section 3.1 re-run on NVIDIA GeForce RTX 4090

run: `20260915_111519_4090`

| config | episodes | frames | run min | fps | GDINO s | SAM s | Cutie s | 3D s | DINO s | load+prep s | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| giraffe | 10/10 | 3142 | 8.3 | 6.34 | 42 | 4 | 43 | 8 | 12 | 151 | 7.21 |

<!-- giraffe: classes ['yellow.pineapple.', 'toy.giraffe.'], backbone dino-vit-b16, pr shapes [(228, 3, 768), (250, 3, 768), (279, 3, 768)]... -->
| pineapple_llm | 10/10 | 1282 | 3.3 | 6.57 | 5 | 4 | 19 | 6 | 5 | 40 | 7.21 |

<!-- pineapple_llm: classes ['pineapple.', 'bowl.'], backbone dino-vit-b16, pr shapes [(103, 3, 768), (118, 3, 768), (121, 3, 768)]... -->
| giraffe_dinov2 | 10/10 | 3142 | 8.3 | 6.30 | 40 | 4 | 43 | 8 | 14 | 152 | 7.22 |

<!-- giraffe_dinov2: classes ['yellow.pineapple.', 'toy.giraffe.'], backbone dinov2-vit-b14, pr shapes [(228, 3, 768), (250, 3, 768), (279, 3, 768)]... -->

GPU over run: mean util 10%, peak mem 7.8 GB, mean power 71 W (328 samples)
