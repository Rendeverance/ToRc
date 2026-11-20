# Changelog

## 0.1.7 (20 November 2026)

Thank you to RainerSC for three pull requests: seven more weight types, so the smallest quants of the big models now
load, and faster two-bit kernels. And thank you to Practical-Trick3332 on Reddit for detailed Snapdragon measurements
and feedback which contributed to the commits in this release:

- Seven more weight types (RainerSC): Q2_K, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ1_M. Unsloth's smallest
  quants, the GSQ-RCO releases and the 2-bit GSQ-hybrid 35B now load. IQ1_S isn't supported yet.
- Faster two-bit weights (RainerSC on x86, ours on ARM), same results bit for bit: the GSQ-hybrid 35B writes at 29
  tok/s instead of 12.7 on a 13900K, and 30.5 instead of 17 on a Snapdragon X.
- `torc plan <file>` tells you in a second or two whether a model will run, and if not, why. Converting and the app
  check it first.
- The GPU attention tier switches itself off where it doesn't help (Gemma 4 on Snapdragon with the NPU), and
  remembers that between starts.
- ARM: Q3_K files read about 40% faster and write 20 to 50% faster on a Snapdragon X, same results bit for bit
  (needs I8MM: Snapdragon X, Apple M2 and later).
- gpt-oss gets an Effort button: low, medium or high.
- The app tells you when it couldn't save your settings, and a narrow window no longer hides the speeds.
- Docs: NPU memory on Windows, one machine against two, active parameters, the full list of quant types, and
  thinking and effort for API clients.
- CI: actions/checkout v5.

## 0.1.6 (19 November 2026)

Mostly for Mac users (Mac changes have not been tested fully on a Mac yet), plus monitoring fixes and a Thinking
switch in the app. A server that keeps answering monitors while the model is busy. Thanks to kilkonie on Reddit for
Mac M5 feedback and data.

- On a Mac with no Node, the app now fetches the macOS build from nodejs.org. Before, it fetched the Linux build,
  which doesn't run on a Mac, so ToolFunnel and the benchmark couldn't start.
- On a Mac the app also finds a Node that Homebrew installed. An app opened from the Finder doesn't get the PATH a
  terminal has, so before it missed it and fetched its own.
- On a Mac the engine's threads are marked as work someone is waiting for (macOS's user-initiated class), so macOS
  doesn't slow them down as background work. `TORC_QOS=0` turns it off.
- `hwscan` on a Mac reports the CPU's name and the memory.
- `GET /slots` and `GET /metrics`, in llama-server's formats: `/metrics` has its prompt, reply and request counts
  under the same names, so a monitor made for llama-server reads them. From another machine they need the API key.
- While a reply is being written, `/health`, `/slots`, `/metrics` and the model list are answered straight away.
  Before, every request waited for the reply to finish, so a monitor checking every second saw the server stall.
  Chat requests that come in meanwhile wait their turn, in the order they arrive in full, and `/metrics` counts them.
  Up to 128 connections can be open at once, as before; one more is told the server is busy. The request bodies
  held at once, being read or waiting, stop at 256 MB in all; past that a request is told the server is busy too. A
  waiting request whose client has gone by its turn is not started; a client that closed its sending side after its
  request (nc -N, some scripts) looks gone, so it gets its reply only when it didn't have to wait.
- PERFORMANCE.md: LFM2.5 on the Snapdragon mini PC with the NPU, faster than the CPU alone at every length.
- A Thinking button beside the message box: as the model's maker set it, on, or off. Before, the app always left it
  to the model, and Qwen3.6 thinks unless it is told not to. It works for the Qwen and GLM models that think; on a
  model that doesn't, it changes nothing. On gpt-oss, off is its lowest reasoning effort. ToRc runs Gemma 4 without
  thinking and LFM2 has no such switch, so the button isn't shown for them.
- The chat's Thinking folds open and close together: open one and they all open, the replies still to come too,
  and the app remembers it. Before, each had to be opened on its own.
- A model file with a weight of a type ToRc has no kernel for yet (an Unsloth IQ3_S file has IQ2_S experts, say) is
  refused when it loads, naming the weight and its type. Before, it loaded and then stopped in the middle of a reply.
- EXPERT-CACHE.md says what the cache does by default on a Mac.
- The checks every change has to pass now lint the ARM build too. The ARM-only code (the NEON kernels and the ARM
  paths around them) had 22 warnings that were never seen, now fixed, with the same bits.

## 0.1.5 (18 November 2026)

The desktop app made clearer for users, plus a couple of server additions. Thanks to RainerSC for his continued
feedback and contributions.

- Share on my network (Settings > Engine): other machines can use the engine's OpenAI and Anthropic APIs. Before,
  the app always kept its server to this machine, whatever `TORC_HOST` said. Sharing comes with an API key, made for
  you, with Copy and New key buttons: paste it where your app or agent asks for an API key. The side pane shows the
  address to use.
- API key for the server (`TORC_API_KEY`): when it is set, another machine's requests must carry it, as
  `Authorization: Bearer <key>` or `x-api-key: <key>`, the ways OpenAI and Anthropic clients send one. Requests from
  the same machine never need it. A wrong or missing key gets a clear error in that API's own shape, and another
  machine has 5 seconds to send its request's head, so one that connects and sends nothing holds the server for
  seconds, not two minutes.
- Model lists that more clients understand: `/v1/models` now gives the model a `created` time, `/v1/models/<name>`
  answers for the loaded model, `/models` (for a client that leaves out the `/v1`) and a trailing slash no longer
  get a "not found", and a model name with spaces is found when it comes %-escaped. Agent frameworks that check the
  model list before they start should now get past that check.
- `GET /health` answers {"status":"ok"}, with no key needed, for monitors and frameworks that check the server is up.
  Like every request, it is answered between replies, not during one.
- Settings are saved as you change them, and the app now informs the user which ones wait for the next Load. The
  Load screen informs the user at the top, and after a change the side pane declares what is waiting, with a Load
  again button (the terminal view's top bar notes it too). The side pane shows what the loaded model is running
  with, not a change that hasn't been loaded yet.
- When ToRc opens, the model you chose last is one press away in the side pane.
- "Listen on" on the Load screen is now "Wait for the first machine on", and only shows on the second of two
  machines, which is the only place it does anything. It was easy to take for the API's address.
- Settings > Model now says plainly that the context, threads, NPU and GPU are on the Load a model screen, and its
  button points there.

## 0.1.4 (18 November 2026)

- Snapdragon X NPU, Qwen3.6-35B (and the other Qwen3.5 / 3.6 MoE models built like it) on one machine: prompts now
  read in blocks of 1024 tokens with 32 of the most used experts a layer on the NPU, when the whole model sits in
  memory with room to spare. That reads faster with the same replies. Other models keep their settings, since a
  bigger block isn't faster on all of them. Thanks to RainerSC, whose tests on an identical box found it.
- NPU on ERNIE 4.5, GLM-4.7-Flash and LFM2.5: the NPU never got their most used experts, because their first layers
  have no experts and the usage file was read the wrong shape. The expert cache had the same mistake. Both are fixed.
- x86 (AVX2): replies on models without drafts write faster, from asking for the next weights earlier.
- Settings you can share: Settings > Share settings saves the settings a model runs with on your machine as a small
  file, and reads one back in, from a settings file or a benchmark result. Every benchmark row now keeps the settings
  it ran with.
- Advanced settings: fast attention (`TORC_ATTN_FAST`, faster long prompts, still in testing so off by default) and
  pinning (`TORC_PIN`) are near the top, each with a line on what it does.
- A cache size set by hand (`TORC_CACHE_MB`) now streams the model through that cache, even when the file would fit
  in memory. Before, it held the whole file as well.
- A usage file written with the 35B's draft layer counted is no longer refused by the NPU.
- Fixes: threads left out of a short piece of work now stay asleep, the fast attention option gives the same result
  in a block as alone, and Gemma 4 on the NPU frees the part that fails when it can't start.
- Linux NPU: the guide covers the NPU memory that is only given back on a restart, and checking free memory before
  loading a big model.

## 0.1.3 (17 November 2026)

- Linux and macOS: fixed a bug where the engine couldn't read how much RAM the machine has on these systems, which
  meant it streamed every model through a small cache. Models that fit in memory are now much faster.
  `TORC_CACHE_MB` can still be set by hand if you want to play.
- Linux on Snapdragon X: the NPU now works, with llama.cpp's Hexagon backend installed (`TORC_NPU=1`; GUIDE.md has
  the setup).
- An NPU that fails while loading is now turned off with a message, and the model carries on with the CPU, where
  before the program stopped.
- Windows: Auto-tune also measures whether pinning the threads is faster on your machine and keeps the better setting.
- Gemma 4 with a discrete GPU: long prompts now fit on an 8 GB card.
- Long prompts: replies are now faster.
- The server now announces when the context is set past the length the model was made and tested for.
- Building on Linux now needs a C++ compiler (g++), for the NPU's error guard.
- Mac (Apple Silicon): EXPERIMENTAL zip.

## 0.1.2 (16 November 2026)

- Linux and macOS: worker threads are left to the OS by default. On Windows the default is pinned. `TORC_PIN=1` pins
  them, `TORC_PIN=0` leaves them unpinned.
- The server says its replies are UTF-8, so every client shows the text right.

## 0.1.1 (16 November 2026)

- Gemma 4 and gpt-oss use their makers' sampling settings.
- Mac (Apple Silicon): EXPERIMENTAL zip.
- More machines and models in PERFORMANCE.md.
- A results template in Issues.
- SHA256SUMS for checking your download.

## 0.1.0 (16 November 2026)

The first release.
