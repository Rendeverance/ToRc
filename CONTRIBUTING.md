# Contributing

Fixes, speed-ups and new models are all welcome, and I'd really like the help.

One thing to know before you start: ToRc runs a lot of different models through a lot of shared code, and it is very easy to make one model faster and another one slower without noticing. So there are a few rules. They're short.

## What a PR needs to say

1. **What you changed and why.**
2. **What hardware you tested on.** CPU, RAM, GPU or NPU if you used one, and the OS. One machine is fine, just say which.
3. **Which models it affects.** Not only the one you were working on. If your change is in code other models run through, it affects them too, so list them and test the ones you can.
4. **Speed before and after.** Same machine, same model, same prompts, read and write from `tools/bench/torc-bench.js` (it uses the server's own timings). Run it before, after, after, before, so a machine warming up doesn't get counted as a speed-up. [GUIDE.md](GUIDE.md) has the method under "Measuring properly".
5. **Quality before and after.**
   - On the CPU the output has to be identical, bit for bit. Run `cargo test --release`, then on the main branch `torc-golden record <models folder>` with the models you have, and on your change `torc-golden check <models folder>`, and say they passed. The hashes in `tools/golden.json` only match my exact model files, so record your own; a check that compared nothing says so and fails.
   - The same input has to give the same bits every time, too. For anything that runs on several threads, run `torc-repeat <model> <text file> 1800 512,1024,256 30`: it reads the same prompt 30 times in one process, in three block sizes, and names the first position of any run that differs from the first. A race shows up as an occasional run that differs.
   - Where the output is allowed to move (the GPU and NPU tiers, a new rounding), say by how much. `TORC_GPU_CHECK=1` prints the GPU's largest difference from the CPU. For anything else, dump the logits before and after with `torc-run` and compare them with `torc-kl`.
6. **Knock-on effects.** Anything else it touches: other models' speed, memory use, load time, the tune files, the other platform (ARM or x86). If you couldn't test something, say so and I'll test it here.

One change per PR. A speed-up and a tidy-up in the same PR can't be measured apart.

CI has to pass: `cargo fmt`, clippy with warnings as errors, the tests on Windows (x86-64 and ARM) and Linux, the engine's tests under AddressSanitizer, and comments kept within 120 columns (`.github/workflows/ci.yml`).

## Where a model-specific optimisation goes

Give it its own path. Don't speed up one model by editing a kernel the other models share.

- **GPU attention.** A shape that needs its own kernel gets its own kernel file and one line in the table that picks it, `Kern::of` in `model/src/tiers/gpu.rs`. `gpu_attn_gqa8.wgsl` and `gpu_attn_mla.wgsl` are the ones to copy. The rest (splitting the work with the CPU, slicing, the learned share, switching off a GPU that is slow or out of memory) is shared and already kept apart per GPU, kernel and shape, so you get it for free. Leave `gpu_attn.wgsl` alone unless you mean to change every model that uses it.
- **CPU kernels.** Add the new kernel next to the old one with a parity test, and select it for its weight type or shape only.
- **Anything learned per machine** goes in the tune file under its own key, so a machine never reuses a number that was measured on something else.
- **New paths are off by default**, behind a `TORC_*` switch listed in [SWITCHES.md](SWITCHES.md), until they've been measured on every model that can reach them.
- **Read a switch through `switches`** (`kernels/src/switches.rs`: `on`, `is`, `num`, `get`), never from the environment yourself. The switches are read once when the process starts, and `cargo test` fails if [SWITCHES.md](SWITCHES.md) and the code don't list the same ones.

## Adding a model family

A model with a network of its own is one folder under `model/src/models/`: its forward pass, a `chat.rs` with its prompt format and tool-call format, and a `serve.rs` that answers chat requests for it through `Serve` (`model/src/chat/serve.rs`). It goes into the server with one line, in `torc-server`'s match on the file's `general.architecture`. Nothing else has to change.

`gemma` and `gptoss` are one model each. `qwen` is the engine most models run on: the Qwen models themselves, and ERNIE 4.5, GLM-4.7-Flash, Moonlight and LFM2, whose networks it covers. Those four have folders of their own (`ernie`, `glm`, `moonlight`, `lfm2`) holding only a `chat.rs`: the model's prompt writer and its call reader, with the prompt pinned in a test. `qwen/serve.rs` picks one from the file's metadata (`Chat`). So a new model whose network the engine already runs is a folder with a `chat.rs` and one line there; a model with a network of its own is a folder with its forward pass and a `serve.rs`.

- **Write the prompt from the model's own chat template** and pin it in a test, byte for byte. A wrong template doesn't crash, the model just answers worse, so the test is the only thing that will catch it.
- **Saved chats come almost free** if the family's whole state is its attention cache: implement `state_bytes` and `set_state_bytes` and use `chat/saved.rs`, as the gemma and gptoss folders do. Prove it with a reply that's the same after a save, a restart and a load as it is read fresh.
- **Tool calls** are read out of the reply in the family's own syntax and handed to the server as `Call`s, so the server never sees the format.

## The desktop app

`desktop/` is the app's Rust shell (Tauri, with its own workspace) and `ui/renderer/` its page. `desktop/bridge.js` is the whole of what the page can ask the shell for. Before a PR that touches the app, build the engine and the app and run `node desktop/test/check.js <file.torc> --two --toolfunnel` on Windows.

## Changing shared code

Shared code is anything more than one model runs through: the generic GPU kernel, the host side of `gpu.rs`, the thresholds in `fattn.rs`, the pool, the pipeline, the tune file.

You can change it, but then every model is affected and has to be shown not to move:

1. Run `torc-gpubench` for each attention shape before and after - it needs no model: `torc-gpubench <query heads> <KV heads> <head dim> <16-bit 0|1> <ctx>`. With `TORC_GPU_CHECK=1` it also prints the kernel each shape ran on and its difference from the CPU.
2. Put both sets of numbers in the PR.
3. Before it merges I run my reference models on my own machines, ARM and x86. If any read speed drops more than 3% or any write speed more than 5%, it doesn't go in as it is.

## Comments and docs

Say what the code does and why it has to. Keep dates, benchmark numbers and the story of how you found the bug out of the comments - the numbers belong in the PR, and the story in its description.

## Results from your machine

Not a code change, but just as useful: [RESULTS.md](RESULTS.md) says how to measure your machine and send the numbers in.

Not sure where something belongs? Open an issue and ask.
