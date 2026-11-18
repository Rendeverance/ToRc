# Performance

How fast ToRc runs on the machines it was built and tested on: one section per model, one table per machine. Each
cell is **read / write** in tokens per second: read is the prompt (prefill), write is the reply (decode). The columns
are the prompt length, from a short question to 128k tokens of text.

These are indicative. Every machine differs (memory speed, cooling, what else is running, drivers), so results may
vary. [RESULTS.md](RESULTS.md) explains how to measure your own machine and contribute your numbers, if you wish.

All values were measured through `torc-server`, warm (the first request, which may be slower, thrown away), with the
server's own timings, 128-token replies (1,024 at 32k and up, except on the laptop) and the texts in `tools/bench/`.
Each figure is the median of several warm requests (the 128k figures are single requests). NPU rows are from the
second session on, once the NPU has its usage file. llama.cpp is shown on the CPU only, as a
like-for-like comparison: ToRc does not put experts on the GPU yet (it is on the roadmap), and its GPU tier is
configured to take a share of a long prompt's attention only. The two-machine tables for the Snapdragons are ToRc only,
since llama.cpp's RPC is not a directly comparable setup. Write speed will depend on how predictable the reply is:
code, prose and so on will produce different speeds.

## Qwen3.8 Flash Next on two boxes!!!

Read / write, tokens per second. A 107 GB model on two mini PCs with 64 GB of RAM between them: the experts that
don't fit are read from the SSDs as they are needed.

**Two Snapdragon X mini PCs (X1-26-100, 8 cores, 32 GB each) over USB4**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU + NPU | 63.9 / 21 | 68.6 / 18.1 | 78.7 / 17.2 | 53.7 / 17.9 | 66.7 / 17.5 | 53.8 / 16.7 |

## Qwen3.6-35B-A3B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k | 128k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 169 / 54 | 173 / 62 | 156 / 48 | 134 / 44 | 117 / 30 | 96 / 29 | 39.8 / 10.2 |
| CPU + NPU | 255 / 51 | 282 / 60 | 257 / 52 | 211 / 45 | 169 / 30 | 118 / 29 | 43.1 / 9.5 |
| CPU + GPU (Adreno X1-45) | 181 / 54 | 179 / 60 | 163 / 53 | 151 / 46 | 131 / 34 | 108 / 29 | 45.2 / 10.1 |
| CPU + NPU + GPU (Adreno X1-45) | 267 / 52 | 283 / 59 | 262 / 49 | 223 / 41 | 189 / 34 | 136 / 29 | 48 / 11 |

**Two Snapdragon X mini PCs over USB4: the same, 32 GB each**

| | short | 1k | 4k | 8k | 16k | 32k | 128k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 263 / 55 | 282 / 59 | 308 / 52 | 275 / 52 | 235 / 32 | 177 / 29 | 72.0 / 10.4 |
| CPU + NPU | 358 / 47 | 374 / 61 | 406 / 50 | 357 / 46 | 286 / 36 | 207 / 30 | 77.2 / 10.1 |
| CPU + GPU (Adreno X1-45) | 272 / 49 | 285 / 59 | 307 / 48 | 282 / 45 | 247 / 34 | 191 / 29 | 81.3 / 10.2 |
| CPU + NPU + GPU (Adreno X1-45) | 352 / 51 | 378 / 59 | 406 / 49 | 370 / 50 | 310 / 32 | 229 / 30 | 86.7 / 11.3 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, Radeon 780M (integrated), RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 172 / 34.2 | 178 / 32.4 | 167 / 35.7 | 151 / 51.9 | 130 / 26.4 | 100 / 22.8 |
| CPU + GPU (RTX 4070 Laptop) | 170 / 33.4 | 177 / 34.7 | 166 / 39.4 | 152 / 51.9 | 147 / 27.0 | 144 / 23.3 |
| CPU + GPU (Radeon 780M, integrated) | 171 / 34.3 | 177 / 32.8 | 166 / 35.4 | 159 / 51.8 | 149 / 25.2 | - |
| llama.cpp (CPU) | 119 / 19.9 | 124 / 19.9 | 122 / 17.9 | 112 / 15.9 | 98.3 / 10.1 | 80.1 / 4.7 |

**Desktop: Core i5-12600K (6P + 4E), 64 GB DDR5-5800, RTX 3060 12 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 190 / 40 | 195 / 40 | 182 / 41 | 164 / 51 | 140 / 28 | 107 / 22 |
| CPU + GPU (RTX 3060) | 192 / 39 | 197 / 41 | 183 / 43 | 170 / 51 | 161 / 27 | 147 / 22 |
| llama.cpp (CPU) | 112 / 24 | 112 / 23 | 107 / 21 | 101 / 18 | 90 / 14 | 74 / 9 |

**Workstation: Core i9-13900K (8P + 16E), 96 GB DDR5-6000, 8 threads for both engines**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 252 / 43.8 | 259 / 52.7 | 243 / 44.2 | 221 / 47.9 | 191 / 43.4 | 148 / 31.7 |
| CPU, drafts off | 253 / 32.7 | 264 / 32.7 | 248 / 32 | 224 / 31.1 | 194 / 29.4 | 149 / 26.1 |
| CPU + GPU (Radeon AI PRO R9700*) | 253 / 42.6 | 262 / 52.6 | 246 / 47.2 | 250 / 51.5 | 252 / 36.4 | 245 / 32.4 |
| llama.cpp (CPU) | 143 / 26.5 | 154 / 26.4 | 143 / 24.2 | 133 / 21.1 | 116 / 15.6 | 91.4 / 7.5 |
| llama.cpp (CPU, drafts on) | 151 / 25.3 | 152 / 34.1 | 143 / 23.6 | 132 / 23.4 | 113 / 16.1 | 88.4 / 4.1 |

*The R9700 takes a share of attention only; the experts stay on the CPU.

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
| llama.cpp (CPU) | 117 / 22.2 | 119 / 21.3 | 92.8 / 12.4 | 59.9 / 6.4 | 43.1 / 2.8 | 25.3 / 1 |

## Gemma 4 26B-A4B (QAT Q4_0)

A note on memory: 25 of Gemma 4's 30 layers read only the last 1024 tokens, but ToRc keeps every position's keys and
values for them, so that a chat can be edited or regenerated from any point without reading it again. That is about
205 KB a token: 6.7 GB at a 32k prompt, on top of the model. A long prompt therefore needs more memory with Gemma than
with the other models here.

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 164 / 34.8 | 170 / 33.7 | 143 / 29.9 | 129 / 28.9 | 112 / 25.5 | 87.7 / 20.4 |
| CPU + NPU | 223 / 35 | 227 / 34 | 186 / 31 | 163 / 29 | 137 / 26 | 102 / 21 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 142 / 20.9 | 146 / 20.3 | 124 / 19.4 | 114 / 18.4 | 102 / 17.1 | 82.6 / 14.6 |
| CPU + GPU (RTX 4070 Laptop) | 133 / 20.9 | 146 / 20.3 | 118 / 19.4 | 125 / 17.9 | 122 / 16.7 | 110 / 14.6 |

**Workstation: Core i9-13900K (8P + 16E), 96 GB DDR5-6000, 8 threads for both engines**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 202 / 29.8 | 204 / 28.8 | 183 / 27.6 | 169 / 26.5 | 150 / 23.9 | 122 / 22.2 |
| CPU + GPU (Radeon AI PRO R9700*) | 203 / 29.8 | 206 / 29.1 | 184 / 27.6 | 183 / 26.4 | 181 / 24.6 | 168 / 20.5 |

## gpt-oss-20b (MXFP4)

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 110 / 17.6 | 114 / 17.6 | 107 / 16.6 | 99.9 / 16.3 | 83.9 / 15.2 | 65.3 / 13 |
| CPU + GPU (RTX 4070 Laptop) | 118 / 17.7 | 123 / 17.5 | 113 / 17.1 | 106 / 16.2 | 103 / 15.1 | 102 / 13.1 |
| llama.cpp (CPU) | 101 / 19.5 | 102 / 19 | 98.4 / 16.3 | 88.5 / 13.5 | 72.9 / 9.8 | 52.6 / 3.6 |

## GLM-4.7-Flash (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 120 / 27.4 | 113 / 26 | 66.8 / 18.9 | 42.8 / 12.8 | 25.4 / 7.9 | 14 / 4.5 |
| CPU + GPU (Adreno X1-45) | 118 / 27.5 | 112 / 26.1 | 66.5 / 18.9 | 42.7 / 12.9 | 25.3 / 7.9 | 14 / 4.5 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 102 / 18.1 | 98.5 / 17.7 | 63 / 14.7 | 42.8 / 11.6 | 26.4 / 8.3 | 14.9 / 5.2 |
| CPU + GPU (RTX 4070 Laptop) | 105 / 17.7 | 101 / 17.7 | 65.9 / 14.7 | 60.5 / 11.5 | 45.3 / 8.2 | 28.7 / 5.3 |
| llama.cpp (CPU) | 98.5 / 17.5 | 91.3 / 16.5 | 66 / 9.9 | 46.3 / 6.6 | 29.3 / 4.4 | 17.1 / 2.1 |

## Moonlight 16B-A3B (Q4_0)

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 139 / 24.7 | 138 / 24.9 | 101 / 21.6 | 74.5 / 17.9 | 49.5 / 14.1 | 29.6 / 9.7 |
| CPU + GPU (RTX 4070 Laptop) | 142 / 24.7 | 139 / 24.8 | 104 / 21.8 | 96.6 / 18.3 | 69.3 / 13.6 | 50.1 / 9.7 |
| llama.cpp (CPU) | 122 / 28.6 | 144 / 27.5 | 110 / 19.3 | 82.3 / 12.5 |  |  |

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
| CPU + GPU (RTX 4070 Laptop) | 172 / 22 | 168 / 21.8 | 148 / 19.2 | 127 / 17.8 | 91 / 12.9 | 68 / 11.5 |
| llama.cpp (CPU) | 106 / 21.8 | 101 / 21.2 | 105 / 16.3 | 92 / 12.4 | 69.8 / 5.2 | 47.9 / 2.2 |

## LFM2.5-8B-A1B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 430 / 81.9 | 440 / 84.3 | 424 / 80.7 | 387 / 74 | 331 / 63.8 | 255 / 50.8 |
| CPU + GPU (Adreno X1-45) | 433 / 77.2 | 438 / 85.5 | 419 / 80.3 | 387 / 73.9 | 340 / 64.1 | 264 / 50.9 |
| llama.cpp (CPU) | 317 / 75.7 | 313 / 74.2 | 284 / 57 | 245 / 33.5 | 194 / 17.4 | 135 / 9.1 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 403 / 47.3 | 368 / 47.3 | 371 / 45.8 | 335 / 43.8 | 299 / 38.1 | 232 / 33.3 |
| CPU + GPU (RTX 4070 Laptop) | 414 / 47.4 | 396 / 47.3 | 370 / 45.4 | 352 / 43 | 342 / 39.7 | 338 / 34.5 |
| llama.cpp (CPU) | 252 / 49.4 | 250 / 48.5 | 248 / 41.5 | 233 / 33 | 206 / 22.7 | 169 / 9.4 |

## Qwen3.5-9B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 102 / 20.1 | 102 / 20.2 | 101 / 19.6 | 95.9 / 18.8 | 87.7 / 17.6 | 74.1 / 15.3 |
| CPU + NPU | 238 / 19.7 | 252 / 20.2 | 241 / 19.6 | 216 / 18.5 | 178 / 17.6 | 131 / 15.3 |
| CPU + GPU (Adreno X1-45) | 103 / 20.1 | 105 / 19.7 | 101 / 19.5 | 96.4 / 18.8 | 88.1 / 17.1 | 74.7 / 15.3 |
| CPU + NPU + GPU (Adreno X1-45) | 238 / 20.2 | 253 / 20.1 | 241 / 19.5 | 215 / 18.8 | 178 / 17.6 | 131 / 15.3 |
| llama.cpp (CPU) | 116 / 17.7 | 114 / 17.4 | 105 / 13.8 | 92.4 / 10.1 | 74.3 / 7 | 53 / 4.2 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 83 / 10.7 | 83.3 / 10.6 | 81.7 / 10.6 | 78.9 / 10.2 | 73.7 / 10.1 | 64.9 / 9.4 |
| CPU + GPU (RTX 4070 Laptop) | 83.8 / 10.7 | 83.9 / 10.7 | 82 / 10.6 | 79.7 / 10.3 | 76.2 / 10.1 | 75.2 / 9.5 |
| llama.cpp (CPU) | 79.4 / 10.6 | 81.6 / 10.5 | 79.4 / 9.9 | 76.2 / 8.3 | 70.2 / 5.9 | 61 / 3.8 |

## Qwen3.5-4B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 181 / 36.2 | 183 / 36.1 | 173 / 34.2 | 159 / 32 | 138 / 28.5 | 108 / 23 |
| CPU + NPU | 378 / 34.5 | 382 / 36.2 | 348 / 33.8 | 296 / 30.6 | 229 / 28.3 | 157 / 22.9 |
| CPU + GPU (Adreno X1-45) | 181 / 36 | 183 / 36.1 | 173 / 34.1 | 159 / 31.8 | 138 / 28.5 | 108 / 22.9 |
| CPU + NPU + GPU (Adreno X1-45) | 359 / 35.8 | 381 / 35.7 | 346 / 34 | 295 / 32 | 229 / 27.3 | 156 / 23 |
| llama.cpp (CPU) | 194 / 27.9 | 192 / 27.2 | 164 / 19 | 135 / 12.7 | 99.4 / 8.1 | 64.8 / 4.6 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 152 / 18.4 | 147 / 18.3 | 145 / 18 | 137 / 17.2 | 121 / 16.7 | 98.8 / 14.4 |
| CPU + GPU (RTX 4070 Laptop) | 154 / 18.5 | 154 / 18.6 | 148 / 17.5 | 140 / 17.6 | 134 / 16.7 | 131 / 15.1 |
| llama.cpp (CPU) | 143 / 18.1 | 144 / 17.9 | 137 / 16.3 | 126 / 12.3 | 112 / 7.6 | 89.4 / 4.4 |

## Qwen3.5-2B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 449 / 68.3 | 436 / 74.5 | 426 / 72.6 | 392 / 68.8 | 343 / 62.6 | 272 / 52 |
| CPU + NPU | 938 / 66.5 | 970 / 66.8 | 873 / 70.9 | 749 / 67.2 | 586 / 61.3 | 406 / 51.1 |
| CPU + GPU (Adreno X1-45) | 447 / 67.3 | 436 / 73.3 | 424 / 70.6 | 390 / 67.1 | 342 / 61.1 | 272 / 50.8 |
| CPU + NPU + GPU (Adreno X1-45) | 927 / 69.9 | 977 / 67 | 853 / 70.9 | 743 / 67.1 | 587 / 61.1 | 407 / 50.7 |
| llama.cpp (CPU) | 505 / 56.3 | 493 / 54.9 | 422 / 47.6 | 353 / 37.7 | 265 / 27.1 | 177 / 17.2 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 378 / 39.2 | 382 / 39.4 | 361 / 39 | 343 / 38 | 307 / 36.3 | 254 / 33 |
| CPU + GPU (RTX 4070 Laptop) | 413 / 39.3 | 417 / 39.3 | 402 / 38.6 | 379 / 37.5 | 347 / 35.8 | 316 / 32.1 |
| llama.cpp (CPU) | 353 / 38.7 | 343 / 38.6 | 339 / 36.2 | 321 / 33.2 | 287 / 23.9 | 238 / 12.9 |

## Qwen3.5-0.8B (Q4_0)

**One Snapdragon X mini PC: X1-26-100 (8 Oryon cores), 32 GB, Hexagon NPU, Adreno X1-45**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 923 / 152 | 920 / 154 | 803 / 143 | 690 / 128 | 551 / 107 | 390 / 79.2 |
| CPU + NPU | 1464 / 147 | 1472 / 150 | 1246 / 139 | 1007 / 124 | 735 / 106 | 474 / 80.3 |
| CPU + GPU (Adreno X1-45) | 919 / 148 | 916 / 149 | 800 / 138 | 682 / 125 | 551 / 105 | 390 / 78.9 |
| CPU + NPU + GPU (Adreno X1-45) | 1481 / 148 | 1509 / 146 | 1236 / 137 | 995 / 125 | 735 / 106 | 473 / 79.8 |
| llama.cpp (CPU) | 931 / 92.6 | 914 / 88.4 | 675 / 71.5 | 508 / 51.4 | 345 / 33.6 | 210 / 19.6 |

**Laptop: Ryzen 7 7840HS (8 cores), 32 GB DDR5-5600, RTX 4070 Laptop 8 GB**

| | short | 1k | 4k | 8k | 16k | 32k |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| CPU | 909 / 92.8 | 913 / 92.3 | 820 / 87.8 | 719 / 84.3 | 580 / 77.1 | 415 / 65.3 |
| CPU + GPU (RTX 4070 Laptop) | 881 / 92.4 | 885 / 92.9 | 795 / 89.8 | 706 / 83.9 | 651 / 76 | 640 / 64.4 |
| llama.cpp (CPU) | 683 / 77.2 | 674 / 75.5 | 619 / 66.7 | 570 / 56.6 | 473 / 33.7 | 351 / 15.3 |

