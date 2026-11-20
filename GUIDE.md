# ToRc user guide

> A CPU-first inference engine for big mixture-of-experts language models, written in Rust with its own model format. It runs on the hardware you already have (Windows on ARM laptops, x86-64 desktops, the Snapdragon NPU if you have one, a second machine if you have one) and on the CPU the answer is the same, bit for bit, whatever thread count or block size you run it at.

This guide is the practical side: build it, convert a model, serve it, then make it fast on your machine. The README is the overview, [SPEC.md](SPEC.md) is the reference for how each part works, and [AGENTS.md](AGENTS.md) is the same material written for an AI agent helping you set it up.

## Why ToRc

Most local inference is built around a big GPU. I wanted to run proper models, the 35B and larger mixture-of-experts ones, on machines that don't have one - a pair of Snapdragon X mini PCs, an ordinary desktop - and get every last bit of speed out of the CPU, the NPU and the memory those machines actually have.

I also wanted to trust it. On the CPU, ToRc gives the same logits at any thread count and any prompt block size, so every speed-up gets checked bit for bit against the build before it. If a change is faster but the answer changes, it doesn't go in - that's strict, I know, but it's the only way I've found to be sure a speed-up is real and not a quiet change in what the model says.

I also wanted it to fit the machine rather than the other way round - measure what the hardware can actually do, tune to that, and if one machine isn't enough, spread the model over two (which is how the Snapdragon pair came about).

## What runs

- **Models:** Qwen3.6-35B-A3B, Qwen3-30B-A3B, Qwen3.8 Flash Next (bigger than RAM, best on two machines), Qwen3-Next 80B, Qwen3.5 122B and the dense Qwen3.5 models, ERNIE-4.5-21B-A3B, GLM-4.7-Flash, Moonlight, LFM2.5-8B, Gemma 4 26B-A4B and gpt-oss. One server, `torc-server`, runs all of them.
- **Quant types:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, plus Q2_K, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ1_M through portable kernels that run on any CPU (Q2_K and Q2_0 with vector code on x86 and ARM). `torc plan <model>` tells you whether a file will run before you convert it.
- **Hardware:** ARM64 (NEON, dotprod, i8mm) and x86-64 (AVX2, and AVX-VNNI or AVX512-VNNI), on Windows and Linux. The Snapdragon X NPU for reading prompts. A second machine as a pipeline worker, over USB4 or a network.
- **Clients:** anything that speaks OpenAI chat completions, the OpenAI Responses API or Anthropic messages. There is a desktop app too, ToRc (see The desktop app, below).

For a feel of the speed, the Qwen3.6-35B (Q4_0) on one Snapdragon X mini PC with 32 GB, measured through the server with `tools/bench/torc-bench.js` as described below (warm, the server's own timings, 128 tokens out): a short prompt reads at about 300 tok/s, a 4k prompt at about 285 tok/s, an 8k prompt at about 240 tok/s and a 32k prompt at about 140 tok/s, with the NPU and GPU tiers on, and replies write at about 50 tok/s after a short prompt, falling to about 30 at 32k. That is from the second session on, once the NPU has learnt the model's most used experts (step 5 below); the first session reads 5 to 9% slower. With two of them the 4k prompt reads at about 405 tok/s, the 8k at about 370 tok/s and the 32k at about 230 tok/s. [RESULTS.md](RESULTS.md) has the full table.

## What you need

- Windows 11, ARM64 or x86-64. On x86-64 the CPU needs AVX2 (Intel Haswell, AMD Zen or later). Linux runs on bare metal too (Ubuntu on a Snapdragon X mini PC, tested by a user) as well as under WSL2.
- Enough RAM for the model, or a fast SSD for the parts that don't fit (ToRc streams experts from the drive).
- Rust (stable). On Windows on ARM, either Rust's default MSVC toolchain with the Visual Studio C++ build tools for ARM64 (the release zips are built this way), or the `aarch64-pc-windows-gnullvm` toolchain with llvm-mingw on your `PATH`.
- Node.js 18 or newer for the benchmark script. Nothing to `npm install`.
- For the NPU: a ggml build with the Hexagon backend (the GenieX CLI installs one) - point `TORC_NPU_DIR` at it if it isn't in the default folder. On Linux that is llama.cpp built with its Hexagon backend (`libggml.so`, with its backends and the Hexagon libraries beside it); the default folder there is `/usr/local/lib`. If the NPU stops opening after a run that crashed, reload the kernel's driver (`rmmod fastrpc && modprobe fastrpc`) or restart: a session that ended badly can leave it unable to open another. The driver also keeps the memory of each session's NPU buffers until the machine restarts (about 1 GB, or 3 GB with the most hot experts), so after a few sessions restart before you measure anything. A model that fits is mapped whole, so it needs its own size free as well: look at `MemAvailable` in `/proc/meminfo` before loading a big one.

## Building

```
# Windows on ARM
cargo build --release --workspace

# Windows x86-64 (AVX2 + FMA)
./build-x86.sh
```

For the GPU tier, add `--features torc-model/gpu` to the ARM line, or run `./build-x86.sh target --features torc-model/gpu`. The release zips are built with it.

On Linux the same `cargo build --release --workspace` line builds it; it needs a C++ compiler as well (g++, in Ubuntu's `build-essential`), for the NPU tier's small guard that turns an exception thrown inside ggml into an error instead of a crash.

The executables land in `target/release`. Built with llvm-mingw (gnullvm), they need its `libunwind.dll` next to them.

`cargo test --release` runs the tests, including the parity tests that check every kernel against the plain definition of its quant type.

## Getting a model

ToRc has its own model format ([FORMAT.md](FORMAT.md)), converted from GGUF (a split GGUF: name its first part):

```
torc plan model.gguf
torc convert-gguf model.gguf model.torc
torc check model.torc
torc check model.torc --against model.gguf
torc info model.torc
```

`plan` tells you in a second or two whether a model will run and, if not, why (an architecture or a weight type
ToRc doesn't run yet, or a model it runs only in its mixture-of-experts form), and how it fits your memory.
`convert-gguf` asks it first, so a model that couldn't run isn't converted. `check` verifies the hashes and every checksum, `--against` compares every tensor and expert with the GGUF byte for
byte, and `info` tells you what's in the file. ToRc doesn't ship weights - models keep their own licences.

## Running it

The server is what you'll use most:

```
torc-server model.torc 8080 auto 16384
```

That's the model, the port, the thread count (`auto` picks every core of your fastest class) and the context length. Point any OpenAI-style client at `http://127.0.0.1:8080/v1`; `GET /health` answers `{"status":"ok"}` for a monitor, with no key needed. `GET /slots` and `GET /metrics` say what the server is doing, in llama-server's formats (Prometheus text for `/metrics`), and need the key from other machines like every other route: `/metrics` has llama-server's prompt, reply and request counts under the same names (not its decode-call or draft counts), and `/slots` one slot, whose `n_decoded` counts the pieces of text sent so far, about one a token. Those three and the model list are answered straight away, even while a reply is being written. The server writes one reply at a time: another chat request waits its turn, in the order requests arrive in full, and `/metrics` counts it in `requests_deferred`. Set `TORC_WARM=wait` and it loads the model into RAM (or as much as fits) before it takes requests. It only listens on this machine unless you set `TORC_HOST`. If you do, set `TORC_API_KEY` as well: then other machines have to send that key (as `Authorization: Bearer <key>` or `x-api-key: <key>`, where your client asks for an API key), while this machine never needs it. The traffic isn't encrypted, so only open it on a network you trust. A web page from another site can't use it either way: the server refuses it, so a page in your browser can't run the model behind your back. Clients that aren't web pages (chat apps, agents, scripts) are not affected.

A few things it does for you:

- **Prompts aren't read twice.** The next turn of a chat reads only the new messages, and a chat you come back to later picks up from a saved state on disk. The README has the details.
- **Drafting.** On models with an MTP block (the Qwen ones; Gemma 4 drafts with its assistant model instead, `TORC_DRAFT`), the server drafts tokens with the model's own prediction block and checks them in one pass. It decides how many to draft on each pass from what a pass costs on your machine. At the default settings the reply is the same tokens either way, just quicker. Not every GGUF keeps the block, as some conversions leave it out: without it the model still runs, just without drafts, and `TORC_MTP_FILE` can add the block from a file of its own.
- **Sampling.** Greedy by default. On every model `temperature`, `top_k` (40 unless you set it), `top_p`, `seed`, `presence_penalty` and `frequency_penalty` work as you'd expect, and a seeded reply is the same with drafting on or off. The desktop app sends each model maker's recommended settings unless you choose others (Load a model, Replies).
- **Thinking and effort.** `"chat_template_kwargs": {"enable_thinking": false}` (or `"think": false`, or `"reasoning_effort": "none"`) turns thinking off on the models that allow it. On gpt-oss, `"reasoning_effort"` of `low`, `medium` or `high` sets how much it reasons, medium when it's left out. The app has a Thinking button beside the message box, and an Effort button for gpt-oss.
- **Reply length.** A request that doesn't set `max_tokens` (or `max_output_tokens` on the Responses API) gets up to 4096 tokens, thinking included. A thinking model can use a couple of thousand of those before it answers, so if you set your own limit, leave it room.

For a quick test of a Qwen model without a client, `torc-chat model.torc "Hello"` prints one reply (it writes the Qwen chat format only; for any other model, start the server).

## The desktop app

The zip on the Releases page is the quickest way in: unzip it, open ToRc, pick a model. To build the app yourself, the
engine first, then the app (`desktop/`, a Rust shell around the system's web view, with its page in `ui/renderer/`):

```
cargo build --release
cargo build --release --manifest-path desktop/Cargo.toml
```

On Linux the app needs WebKitGTK (on Ubuntu: `libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev`). `desktop/package/pack.sh` lays the two builds out as the zip: the app, then `engine/`, `models/`, `chats/`, `primers/`, `mcps/`, `tools/`, `data/` and `docs/` beside it. Run from that folder, the app keeps its settings, chats and models in it; run from anywhere else, it keeps its settings in your system's folder for app data. Either way, what the engine learns about the machine (Auto-tune's results, expert usage, the two-machine split) goes in ToRc's cache folder, `%LOCALAPPDATA%\torc` or `~/.cache/torc`.

Pick a model from the left pane: a `.torc` file, or a GGUF, which it converts once (it checks the architecture first and tells you the space it needs). Auto-tune measures your machine; then load and chat. It starts and stops torc-server itself, so there's nothing else to run. Settings are saved as you change them, and the ones the engine reads when it starts (the model, context, threads, NPU, GPU, the second machine, saving chats, the engine folder, port, sharing and its key, and the Advanced switches) take effect at the next Load: the side pane says when a change is waiting for one, and the model you chose last is one press away when ToRc opens. To use the engine from other machines as well, turn on Settings > Share on my network: it makes an API key for them to send (this machine never needs it), and only use it on a network you trust. Chats are saved as plain JSON files, a primer is a Markdown file, and Advanced lists every switch in [SWITCHES.md](SWITCHES.md) with a box to set it. The built-in tools (the time, a calculator, the model's own context use, and, off until you switch them on, compacting and wake-ups) are always there; beside them are either your own MCP servers (ToRc asks you before each call one of those makes) or ToolFunnel, which it installs for you and whose tools and servers the Tools screen lists. On Windows, `node desktop/test/check.js <file.torc>` runs the real app against the real engine from a folder of its own, so it never touches yours; it drives the page through WebView2's DevTools port, which the app opens on the local machine only when `TORC_DEVTOOLS_PORT` is set.

## Making it fast on your machine

ToRc's defaults are sensible, but every machine is different, and the only way to know what helps on yours is to measure it. Roughly in this order:

1. **Look at the machine first.** `hwscan` reports your cores, memory bandwidth, real clocks under load and drive speed. On a Mac and on Linux its inventory is mostly empty for now (a Mac gets the CPU's name and the memory).
2. **Tune the prompt block.** `torc-tune model.torc` tries the block sizes on your machine and saves the best one where the server finds it. Run it with the same settings you'll serve with (NPU and GPU on, the second machine up). A few minutes.
3. **Threads.** `auto` is usually right. On a hybrid Intel CPU, also try the P-cores only, one thread per core.
4. **Two ARM switches worth trying:** `TORC_Q4X4=1` and `TORC_ATTN_TILE=1`. Q4X4 is off by default on the Qwen models (Gemma turns it on itself when the NPU is off) and is faster on my Snapdragon machines with the 35B. The tiled attention path is on by default for the Qwen3.6-35B and the other Qwen3.5 / 3.6 MoE models built like it, where it makes replies at a long context a few percent faster with the same output, and off for the rest: try `TORC_ATTN_TILE=1` on them, or `0` on the 35B, and keep what is faster on yours. On any machine, also try fast attention, `TORC_ATTN_FAST=1` (Advanced settings in the app): long prompts read faster. It should stay accurate (its scores differ from the default by about 3e-5 of the largest), but I'm still testing it, so it is off by default and replies can differ a little from the default's.
5. **The NPU (Snapdragon X).** `TORC_NPU=1` reads prompts faster by putting part of the work on the NPU. It isn't bit-exact (the NPU rounds differently), so it only touches prompts, never replies. Run `npucheck` first to make sure it works. On one machine it also learns which experts your work uses most and gives those to the NPU from the next time the model loads, so the second session reads faster than the first - nothing to set up. On one machine with the Qwen3.6-35B and the other Qwen3.5 / 3.6 MoE models built like it, when the whole model sits in memory, the NPU reads in blocks of 1024 tokens and takes 32 of those experts a layer (32 only with 8 GB to spare); everything else keeps blocks of 256 and 16 experts, because a bigger block reads slower on some models. The NPU's memory belongs to the server: on Windows it is all given back the moment torc-server exits, nothing to restart. If free memory looks low, check an old torc-server isn't still running.
6. **The GPU.** In the release zips (and in a build with the GPU feature), `TORC_GPU_ATTN=1` runs part of a long prompt's attention on the GPU while the CPU does the rest. It takes the integrated GPU by default (`TORC_GPU=<part of a name>` picks another), works out how to split the work on your machine and keeps that split for next time. It only joins in once a prompt is a few thousand tokens long, and like the NPU it isn't bit-exact, so it never touches replies. `TORC_GPU_CHECK=1` compares its results with the CPU's. If the GPU turns out to be no gain over the CPU alone, ToRc switches it off for that model, says so in the log and remembers. If the card runs out of memory (a big model's keys at a long context on a small card), ToRc switches the GPU off for the rest of the run, says so in the log, and the CPU carries on. Intel integrated graphics on a desktop is that case: it is several times slower than the CPU next to it, and its driver can crash outright on some models, so if you have a graphics card name it with `TORC_GPU`, and if Intel integrated is all you have, leave this off. And because the split follows measured speed, two runs of the same prompt can give slightly different replies with the GPU on, where the model's choice is a near tie. With it off every run is identical.
7. **Large pages** (Windows, needs the "Lock pages in memory" right for your account). `TORC_KLAY_LARGE`, `TORC_EXPERTS_LARGE` and `TORC_OWN_LARGE` help on some machines, but a machine that has been up for days can't always find the memory for them, so measure. (`TORC_EXPERTS_LARGE` does nothing with `TORC_Q4X4=1` on - the repacked copies are what get read then.)
8. **A model bigger than your RAM.** Experts come off the SSD, so the cache size (`TORC_CACHE_MB`) and the read settings matter. [EXPERT-CACHE.md](EXPERT-CACHE.md) covers these (tuning them is planned, [AUTOTUNE.md](AUTOTUNE.md)), and [SWITCHES.md](SWITCHES.md) lists every environment switch.
9. **Two machines.** [TWO-BOX.md](TWO-BOX.md) - one runs the first layers, the other the rest. For a model that fits on one machine, use both for the prompt and pull the reply back to one (`TORC_PIPE_PULL=1`).

Any `TORC_*` setting in the environment overrides a tuned value, so you can always A/B one against the other.

## Measuring properly

This is the bit that took me longest to get right, so it's worth a few lines.

Measure through the server, the way you'll actually use it, and use the server's own timings. Every reply carries them, so there's no stopwatch to argue about:

- **read** - prompt tokens divided by the time it took to read the prompt
- **write** - reply tokens divided by the time it took to write them

`tools/bench/torc-bench.js` does the whole job. It sends a short prompt and then 1k, 4k, 8k and 16k tokens of fixed text (32k too with `--sizes`, and the app's Benchmark adds it when the context has room; a one-line timestamp first, so the server can't reuse the last prompt's state), warms each size up, measures it twice and writes the results to a CSV with your hardware filled in:

```
node tools/bench/torc-bench.js 8080 --model "Qwen3.6-35B-A3B Q4_0" --settings "threads 12"
```

Run it on the same machine as the server. A long prompt is worth the wait, as that's where attention starts to cost. [RESULTS.md](RESULTS.md) says how to send the CSV in. `tools/bench/lbench.js` is the quick single test: one short prompt, timed on the client, or your own text with `LB_PROMPT=myprompt.txt`.

Then:

- **Warm.** Start the server, send one request and throw it away, then measure.
- **Alternate.** Comparing two builds or two settings? Run A, B, B, A, not A, A, B, B. Machines drift as they warm up (heat, caches filling, something waking up in the background), and alternating means the drift lands on both sides rather than one.
- **Enough runs.** Write speed moves a few tok/s from one request to the next. A 2% difference needs around 40 requests a side before you can believe it.
- **Hands off.** Don't build, copy files or run anything else on the machine during a measurement, and don't leave the profiler (`TORC_PROF=1`) on - it costs a few percent on its own.

`torc-run` and the profiler are great for finding where the time goes, but they aren't the number to quote - the server is.

## Limitations

ToRc is a focused solo build, tested hard on my own machines: two Snapdragon X mini PCs, an x86 laptop and an x86 desktop. Outside of that it will have rough edges.

- **Windows first.** Linux runs on bare metal (Ubuntu on a Snapdragon X mini PC, tested by a user) and under WSL2 (Ubuntu 24.04). On Linux `auto` threads take every CPU, E-cores included: give the P-cores' count by hand on a hybrid CPU. The threads are left to the OS on Linux, and Auto-tune doesn't measure pinning there yet, but pinning them (`TORC_PIN=1`) has read and written faster on a Snapdragon X under Ubuntu, so it's worth trying.
- **One request at a time.** The server runs one conversation at once; it's built for a person, not a crowd.
- **Gemma 4** doesn't save chats on the whole-NPU tier (`TORC_NPU=pure`).
- **Text the model reads can still steer it.** Control strings like `<|im_end|>` in what you paste or what a tool returns are broken before the model sees them, so they can't fake a turn (except on ERNIE, whose turns are the plain words "User:" and "Assistant:", which no marker-breaking can protect). But a page that says "ignore the user" in plain words still works on most models, mine included. Don't give a model text from the open web and tools that can do damage in the same chat.
- **x86 has had less attention than ARM.** It builds and runs, but most of the tuning has been on the Snapdragon machines.
- **The GPU only helps with long prompts.** It takes a share of their attention (step 6 above); the rest of the model, and every reply, stays on the CPU. This isn't a GPU inference engine.
- **The switches.** There are a lot of `TORC_*` settings, many of them alternatives kept off by default. The ones that matter are in this guide; the rest are explained in the comment beside each switch, and [SPEC.md](SPEC.md) goes through the bigger ones in detail.

If you find a bug - better still, a bug and a fix - or you get it running well on hardware I haven't tried, I'd like to hear about it - I'm keen to work with others on this :)

## Licence

MIT or Apache 2.0, your pick. Just keep the copyright notice. What ToRc borrows from other projects (mostly ggml and llama.cpp, also MIT) is listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), and every crate it builds with, with its licence, in [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md). All of them are permissive.
