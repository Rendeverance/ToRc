# Autotuning

Each machine gets the settings measured best on it, for each model, instead of rules picked on mine. `torc-tune`
measures them once and keeps them; the server reads them at start. An environment variable always wins over a tuned
value.

## What is tuned

`torc-tune <model> [threads|auto] [ctx]` tunes four settings, in this order, each from the values already chosen:

| Setting | Candidates | Phase it moves |
|---|---|---|
| prompt block size | the built-in rule, half and double it, then on by doubling or halving while each step is faster | prompt |
| work split of the expert products | even (the default) or guided | prompt |
| pinning | each thread held to one CPU, or left to the OS | prompt and decode |
| threads a reply uses | all of them, or one a core on a part with two threads a core | decode |

The reply threads are tuned apart from the prompt's because the best count for reading a prompt is not the best for
writing a reply: decode is bound by memory, and a core's second thread only competes for it. On Linux this step is
skipped for now, since the engine can't see the cores there (README, Running).

Pinning is measured because neither way wins everywhere: pinned threads keep their caches, but a pinned thread also
waits when something else takes its CPU for a moment. The prompts are read and 64 tokens written both ways. Windows'
default (pinned) is kept unless the other way is faster over reading and writing together by the margin and slower on
neither, and the server is what uses the result (`torc-run` and `torc-chat` keep the default). `TORC_PIN` set by hand
skips the step. On Linux the step is skipped for now and the threads stay unpinned, and on macOS, where a thread
can't be pinned, there is nothing to measure.

One more value is learned by the server rather than by `torc-tune`: with the GPU tier on, the GPU's share of each
block's attention, measured on the first long prompts and kept in the same file (GUIDE, Making it fast).

Run it with the server's environment (`TORC_NPU`, and `TORC_PIPE` with the worker up for two machines): the result
is kept per layout, one machine or two, NPU or CPU, since the best block changes with both.

## How a setting is measured

1. **The real model and fixed inputs.** Three fixed texts (prose, code and a story), each read as prompts of 600
   and 1800 tokens with one decoded token after, through the same code path the server runs. A kernel benchmark
   does not see the cache, the pool or the memory traffic of the other layers, so it can pick the wrong value.
2. **Warm only.** The first run of each candidate is discarded.
3. **Alternating order.** Candidates run A B C, C B A, A B C, so a drift over the session (heat, the cache filling, a
   background task) falls on both sides equally. The score of a candidate is the median of its runs.
4. **A margin to move.** A candidate replaces the default only if it is better by more than the run-to-run spread
   measured on the default itself, by at least 2.5%, and is no slower than the default on any of the texts. Ties
   keep the default.
5. **Correctness.** Every candidate's decoded logits are compared with the default's bit for bit on the CPU: none of
   these settings may change the output, so a difference is a bug, and that candidate is not used. With the NPU on,
   each prompt must pick the same next token.

## Where results live

`%LOCALAPPDATA%\torc\tune\<machine>\<model>.json` (on Linux `$XDG_CACHE_HOME/torc/tune/...`, else `~/.cache/...`).

- **machine**: a hash of the CPU's identifier, the CPU count and the memory installed. A new CPU or more memory
  means tuning again; a new engine build does not. The identifier is Windows' (`PROCESSOR_IDENTIFIER`), so on Linux
  the key is the CPU count and the memory alone for now.
- **model**: the file's model id, so the same model at another context finds the same result.
- Each setting is kept with its chosen value, the scores it was chosen on and when, so a run can show why.

## When it runs

`torc-tune <model>`, by hand: a few minutes for a model in RAM. Start the server again afterwards; it reads the file
once.

## The NPU and the GPU

The NPU and the GPU's share of long prompts read faster, but they round differently from the CPU, so the results are
not bit-exact: not with the CPU's, and not between platforms and vendors. The effect on quality is slight, and replies
are written on the CPU (except with Gemma's whole-NPU tier, `TORC_NPU=pure`).

So when Auto-tune finishes, the app asks: on a Snapdragon X with the NPU off, whether to use the NPU, and once the GPU
switch is on, whether to keep using the GPU. Using it is the highlighted choice, CPU only the other. Once answered, it
isn't asked again; the switches on the Load screen change it any time.

## Sharing settings

The settings a model runs with on one machine can go to another. The server gives them as a profile at
`GET /torc/settings`: the values in use (the prompt block, the work split, pinning, the reply threads and the NPU's hot
experts a layer) and what chose each (a switch, the tune file or the engine's default), with the switches it was
started with, the model file it was measured on
and the hardware. `tools/bench/torc-bench.js` keeps the same settings in every row of its CSV, so a result always says
how it was run.

In the desktop app, Settings > Share settings saves the loaded model's settings as a file, and reads one back in from
a settings file or a benchmark CSV. Reading in (`POST /torc/settings`) writes the values a switch or the tune file
chose (not the engine's defaults, which differ from machine to machine) into this machine's tune file for the same
model, under the layout they were measured in (one machine or two, NPU or CPU), and the model loads again with them.
They count as tuned values: a switch still wins over them, and running `torc-tune` measures the block, the
split, pinning and the reply threads again on this machine. The hot experts a layer stay as read in (`TORC_NPU_HOT`
overrides them), and more than 16 are only taken where the whole model fits in memory with 8 GB to spare. Settings
that name a model file of another size or shape are refused. A switch holding a path or an address is never written
into a profile, and the app only takes switches back in that are plain numbers or single words and point at nothing
on the disk or the network.

## Planned

None of this is built yet:

- More settings: the thread count for the prompt, the packed-product threshold and the prefetch distance on x86,
  the two-machine layer split, and the expert cache's budget.
- The drive, for models bigger than RAM: read size, reads in flight, I/O threads and read order, tuned on the
  drive the model sits on, with that drive in the key.
- A second pass over the block and the threads, which interact.
- Checks around a run: wait for a quiet machine, repeat a candidate whose runs spread too far, and measure the
  default again at the end to catch throttling.
- The engine build's kernels in the key, so a kernel change tunes again.
- A first server start with no tuning file tuning in the background while idle.
