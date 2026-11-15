# Performance

How fast ToRc runs on the machines it was built and tested on: one section per model, one table per machine. Each
cell is **read / write** in tokens per second: read is the prompt (prefill), write is the reply (decode). The columns
are the prompt length, from a short question to 128k tokens of text.

These are indicative. Every machine differs (memory speed, cooling, what else is running, drivers), so results may
vary. [RESULTS.md](RESULTS.md) explains how to measure your own machine and contribute your numbers, if you wish.

All values were measured through `torc-server`, warm (the first request, which may be slower, thrown away), with the
server's own timings, 128-token replies and the texts in `tools/bench/`. NPU rows are from the second session on,
once the NPU has its usage file. llama.cpp is shown on the CPU only, as a like-for-like comparison: ToRc does not put
experts on the GPU yet (it is on the roadmap), and its GPU tier is configured to take a share of a long prompt's
attention only. The two-machine tables for the Snapdragons are ToRc only, since llama.cpp's RPC is not a directly
comparable setup. Write speed will depend on how predictable the reply is: code, prose and so on will produce
different speeds.

## Qwen3.8 Flash Next on two boxes!!!

Read / write, tokens per second. A 107 GB model on two mini PCs with 64 GB of RAM between them: the experts that
don't fit are read from the SSDs as they are needed.

**Two Snapdragon X mini PCs (X1-26-100, 8 cores, 32 GB each) over USB4**

| | short | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|
| CPU | 64.4 / 21.1 | 81 / 17 | 53.7 / 17.9 | 66.7 / 17.5 | 56 / 13.7 |

## Qwen3.6-35B-A3B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k | 128k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 169 / 54 | 173 / 62 | 156 / 48 | 134 / 44 | 117 / 30 | 96 / 29 |  |
| CPU + NPU | 255 / 51 | 282 / 60 | 257 / 52 | 211 / 45 | 169 / 30 | 118 / 29 |  |
| CPU + GPU (Adreno X1-45) | 181 / 54 | 179 / 60 | 163 / 53 | 151 / 46 | 131 / 34 | 108 / 29 |  |
| CPU + NPU + GPU (Adreno X1-45) | 267 / 52 | 283 / 59 | 262 / 49 | 223 / 41 | 189 / 34 | 136 / 29 | 48 / 11 |

**Two Snapdragon X mini PCs over USB4: the same, 32 GB each**

| | short | 1k | 4k | 8k | 16k | 32k | 128k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 263 / 55 | 282 / 59 | 308 / 52 | 275 / 52 | 235 / 32 | 177 / 29 |  |
| CPU + NPU | 358 / 47 | 374 / 61 | 406 / 50 | 357 / 46 | 286 / 36 | 207 / 30 |  |
| CPU + GPU (Adreno X1-45) | 272 / 49 | 285 / 59 | 307 / 48 | 282 / 45 | 247 / 34 | 191 / 29 |  |
| CPU + NPU + GPU (Adreno X1-45) | 352 / 51 | 378 / 59 | 406 / 49 | 370 / 50 | 310 / 32 | 229 / 30 | 86.7 / 11.3 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 169 / 33.9 | 174 / 33.1 | 165 / 34.4 | 148 / 45.1 | 128 / 22.7 | 100 / 16.6 |
| CPU + GPU (RTX 4070 Laptop) | 173 / 32.3 | 180 / 34.3 | 169 / 35.8 | 154 / 42.5 | 149 / 22.9 | 145 / 17.9 |

**Desktop: Core i5-12600K (6P + 4E), 64 GB DDR5-5800, RTX 3060 12 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 190 / 40 | 195 / 40 | 182 / 41 | 164 / 51 | 140 / 28 | 107 / 22 |
| CPU + GPU (RTX 3060) | 192 / 39 | 197 / 41 | 183 / 43 | 170 / 51 | 161 / 27 | 147 / 22 |
| llama.cpp (CPU) | 112 / 24 | 112 / 23 | 107 / 21 | 101 / 18 | 90 / 14 | 74 / 9 |

## Qwen3-30B-A3B-Instruct-2507 (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 177 / 43.2 | 177 / 42.9 | 125 / 32.8 | 89.9 / 25.6 | 58.2 / 17.9 | 33.7 / 11 |
| CPU + NPU | 195 / 44 | 201 / 44 | 135 / 33 | 94 / 26 | 59 / 18 | 34 / 11 |
| llama.cpp (CPU) | 127 / 30.6 | 119 / 28.1 | 71.4 / 13.2 | 45.7 / 5.1 | 26.2 / 2 | 14.2 / 0.9 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 179 / 24.7 | 174 / 24.6 | 126 / 21.6 | 92.5 / 18.6 | 60.9 / 14.7 | 36.2 / 10.4 |
| CPU + GPU (RTX 4070 Laptop) | 179 / 24.8 | 177 / 24.6 | 128 / 21.6 | 130 / 18.5 | 105 / 14.7 | 72.9 / 10.5 |

## Gemma 4 26B-A4B (QAT Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 164 / 34.8 | 170 / 33.7 | 143 / 29.9 | 129 / 28.9 | 112 / 25.5 | 87.7 / 20.4 |
| CPU + NPU | 223 / 35 | 227 / 34 | 186 / 31 | 163 / 29 | 137 / 26 | 102 / 21 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 142 / 20.8 | 146 / 20.3 | 124 / 19.4 | 114 / 18.3 | 102 / 17.1 | 82.6 / 14.6 |
| CPU + GPU (RTX 4070 Laptop) | 133 / 20.9 | 146 / 20.3 | 118 / 19.4 |  |  |  |

## gpt-oss-20b (MXFP4)

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 110 / 17.6 | 114 / 17.6 | 107 / 16.6 | 99.9 / 16.3 | 83.9 / 15.2 | 65.3 / 13 |
| CPU + GPU (RTX 4070 Laptop) | 118 / 17.7 | 123 / 17.5 | 113 / 17.1 | 106 / 16.2 | 103 / 15.1 | 102 / 13.1 |
| llama.cpp (CPU) | 105 / 20 | 107 / 19 | 101 / 16 | 91 / 12 | 75 / 10 | 54 / 3 |

## GLM-4.7-Flash (Q4_0)

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 102 / 18.1 | 98.5 / 17.7 | 63 / 14.7 | 42.8 / 11.6 | 26.4 / 8.3 | 14.9 / 5.2 |
| CPU + GPU (RTX 4070 Laptop) | 105 / 17.7 | 101 / 17.7 | 65.9 / 14.7 | 60.5 / 11.5 | 45.3 / 8.2 | 28.7 / 5.3 |

## Moonlight 16B-A3B (Q4_0)

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 139 / 24.7 | 138 / 24.9 | 101 / 21.6 | 74.5 / 17.9 | 49.5 / 14.1 | 29.6 / 9.7 |
| CPU + GPU (RTX 4070 Laptop) | 142 / 24.7 | 139 / 24.8 | 104 / 21.8 | 96.6 / 18.3 | 69.3 / 13.6 | 50.1 / 9.7 |

## ERNIE-4.5-21B-A3B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 183 / 40.1 | 183 / 39.2 | 158 / 32.1 | 131 / 26.4 | 97.6 / 19.7 | 64.1 / 13 |
| CPU + NPU | 198 / 39.9 | 198 / 39 | 171 / 32 | 142 / 26.3 | 105 / 19.7 | 69.2 / 13 |
| CPU + GPU (Adreno X1-45) | 184 / 39.8 | 184 / 39.1 | 160 / 31.9 | 135 / 26.4 | 101 / 19.7 | 68 / 13 |
| CPU + NPU + GPU (Adreno X1-45) | 199 / 39.8 | 199 / 39.1 | 171 / 32.1 | 142 / 26.6 | 105 / 19.7 | 69.6 / 13 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 164 / 21.9 | 162 / 21.8 | 144 / 19.9 | 123 / 16.9 | 97.2 / 15.2 | 67.7 / 11.6 |

## LFM2.5-8B-A1B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 430 / 81.9 | 440 / 84.3 | 424 / 80.7 | 387 / 74 | 331 / 63.8 | 255 / 50.8 |
| llama.cpp (CPU) | 317 / 75.7 | 313 / 74.2 | 284 / 57 | 245 / 33.5 | 194 / 17.4 | 135 / 9.1 |

## Qwen3.5-9B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 102 / 20.1 | 102 / 20.2 | 101 / 19.6 | 95.9 / 18.8 | 87.7 / 17.6 | 74.1 / 15.3 |
| CPU + NPU | 240 / 20.2 |  | 243 / 19.7 |  | 179 / 17.7 |  |
| CPU + GPU (Adreno X1-45) | 103 / 20.4 |  | 102 / 19.7 |  | 88.4 / 17.7 |  |
| CPU + NPU + GPU (Adreno X1-45) | 238 / 20.2 |  | 242 / 19.6 |  | 179 / 17.7 |  |
| llama.cpp (CPU) | 116 / 17.7 | 114 / 17.4 | 105 / 13.8 | 92.4 / 10.1 | 74.3 / 7 | 53 / 4.2 |

## Qwen3.5-4B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 181 / 36.2 | 183 / 36.1 | 173 / 34.2 | 159 / 32 | 138 / 28.5 | 108 / 23 |
| CPU + NPU | 360 / 34.2 |  | 350 / 32.6 |  | 230 / 27.4 |  |
| CPU + GPU (Adreno X1-45) | 181 / 34.3 |  | 174 / 32.7 |  | 138 / 27.3 |  |
| CPU + NPU + GPU (Adreno X1-45) | 379 / 32.8 |  | 349 / 32.6 |  | 230 / 27.3 |  |
| llama.cpp (CPU) | 194 / 27.9 | 192 / 27.2 | 164 / 19 | 135 / 12.7 | 99.4 / 8.1 | 64.8 / 4.6 |

## Qwen3.5-2B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 449 / 68.3 | 436 / 74.5 | 426 / 72.6 | 392 / 68.8 | 343 / 62.6 | 272 / 52 |
| CPU + NPU | 888 / 65.4 |  | 838 / 70 |  | 570 / 60.2 |  |
| CPU + GPU (Adreno X1-45) | 451 / 65.7 |  | 422 / 69.6 |  | 344 / 58.8 |  |
| CPU + NPU + GPU (Adreno X1-45) | 884 / 65.6 |  | 837 / 69.7 |  | 568 / 60.1 |  |
| llama.cpp (CPU) | 505 / 56.3 | 493 / 54.9 | 422 / 47.6 | 353 / 37.7 | 265 / 27.1 | 177 / 17.2 |

## Qwen3.5-0.8B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 923 / 152 | 920 / 154 | 803 / 143 | 690 / 128 | 551 / 107 | 390 / 79.2 |
| CPU + NPU | 1329 / 145 |  | 1116 / 138 |  | 698 / 105 |  |
| CPU + GPU (Adreno X1-45) | 915 / 146 |  | 789 / 137 |  | 549 / 105 |  |
| CPU + NPU + GPU (Adreno X1-45) | 1344 / 147 |  | 1148 / 138 |  | 687 / 105 |  |
| llama.cpp (CPU) | 931 / 92.6 | 914 / 88.4 | 675 / 71.5 | 508 / 51.4 | 345 / 33.6 | 210 / 19.6 |

