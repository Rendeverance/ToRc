# Results

How fast ToRc runs on real machines, measured with the same script and texts (the notes under the table list where
my own rows differ).

I'm building a table of the best result for each machine and model, with the settings that got it. Send me yours. Every result that comes in makes the defaults better for the next person with similar hardware.

## Measure your machine

1. Build ToRc and start the server with the settings you want measured ([GUIDE.md](GUIDE.md)). Give it enough context for the biggest prompt: 20480 covers everything up to 16k.
2. Close everything else. Another program taking even one core can cost you 10%.
3. Run the bench:

   ```
   node tools/bench/torc-bench.js 8080 --model "Qwen3.6-35B-A3B Q4_0" --settings "threads 12, CPU only" --name yourname
   ```

It sends the standard prompts (a short one, then 1k, 4k, 8k and 16k tokens), prints the speeds as it goes and adds one row per size to `torc-bench.csv`. Add `--sizes alice,1k,4k,8k,16k,32k` if you have the context for 32k.

The file holds the CPU, cores, RAM size and speed, GPU, NPU, OS and power plan, your settings, and the speeds. No host name, no paths, nothing else. Open it and look before you send it.

## Send it in

Open an issue called `Results: <your CPU>, <model>` and attach the CSV.

Say in `--settings` anything that limits the machine: a power limit, a small case, a laptop on battery. A number without that is no use to anyone.

## What happens to it

The best row for each machine and model goes in the table below with your name on it if you gave one. The settings that won are what I'll start the autotuner from on similar hardware.

## How the numbers are taken

- Speeds are the server's own timings, in tokens a second. Read is the prompt, write is the reply.
- Each prompt is run once to warm up, then twice. The figure is the average of the two (the median, if you ask for more runs with `--runs`).
- Every prompt starts with a line that has the time in it, so no server can reuse a prompt it has already read.
- Replies are greedy, thinking off, 128 tokens.
- Read repeats to within a few percent. Write moves more, by several tokens a second from one reply to the next when drafts are on, so run it more than once before you compare two write figures.
- The prompts are fixed public domain texts in `tools/bench` (the openings of *Alice's Adventures in Wonderland* and *Pride and Prejudice*), so everyone reads the same thing.

The same script works against llama.cpp's server, so you can measure both on your machine and send both.

## The table

| Machine | Model | Settings | alice | 1k | 4k | 8k | 16k | 32k | By |
|---|---|---|---|---|---|---|---|---|---|
| 2 x Snapdragon X mini PC (32 GB each), USB4 link | Qwen3.6-35B-A3B Q4_0 | CPU + NPU + GPU on both | 352 / 51 | 378 / 59 | 406 / 49 | 370 / 50 | 310 / 32 | 229 / 30 | Rendeverance |
| 2 x Snapdragon X mini PC (32 GB each), USB4 link | Qwen3.6-35B-A3B Q4_0 | CPU only on both | 263 / 55 | 282 / 59 | 308 / 52 | 275 / 52 | 235 / 32 | 177 / 29 | Rendeverance |
| Snapdragon X mini PC (32 GB) | Qwen3.6-35B-A3B Q4_0 | CPU + NPU + GPU | 267 / 52 | 283 / 59 | 262 / 49 | 223 / 41 | 189 / 34 | 136 / 29 | Rendeverance |
| Snapdragon X mini PC (32 GB) | Qwen3.6-35B-A3B Q4_0 | CPU only | 169 / 54 | 173 / 62 | 156 / 48 | 134 / 44 | 117 / 30 | 96 / 29 | Rendeverance |
| Desktop, i5-12600K, 64 GB, RTX 3060 | Qwen3.6-35B-A3B Q4_0 | CPU + GPU attention on the 3060 | 192 / 39 | 197 / 41 | 183 / 43 | 170 / 51 | 161 / 27 | 147 / 22 | Rendeverance |
| Desktop, i5-12600K, 64 GB | Qwen3.6-35B-A3B Q4_0 | CPU only | 190 / 40 | 195 / 40 | 182 / 41 | 164 / 51 | 140 / 28 | 107 / 22 | Rendeverance |

Read / write, tokens a second.

Write depends on how predictable the reply is, because drafts are kept or thrown away, so treat it as a range: the
same machine and prompt can move by several tokens a second from one reply to the next. The desktop rows were run
with "Continue the story." after each text and the mini PC rows without, so compare their read columns, not their
write columns. At 16k and 32k a 128-token reply is too short to show what drafting gives: with 1,024-token replies
the desktop writes 33 at 16k and 27 at 32k. The mini PC rows at 32k already use 1,024-token replies.
