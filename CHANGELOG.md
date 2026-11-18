# Changelog

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
