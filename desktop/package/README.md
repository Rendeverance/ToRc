     ToRc                                                                                 ,---,_          ,
         Author: Rendeverance                                                           _>   `'-.  .--'/
         Copyright (c) 2026 Rendeverance. MIT or Apache-2.0.                       .--'` ._      `/   <_
                                                                                     >,-' ._'.. ..__ . ' '-.
                                                                                 .-'   .'`         `'.     '.
                                                                                  >   / >`-.     .-'< \ , '._\
                                                                                 /    ; '-._>   <_.-' ;  '._>
                                                                                `>  ,/  /___\ /___\  \_  /
                                                                                 `.-|(|  \o_/  \o_/   |)|`
                                                                                     \;        \      ;/
                                                                                       \  .-,   )-.  /
                                                                                        /`  .'-'.  `\
                                                                                       ;_.-`.___.'-.;
    
         Big models, small machines.

# ToRc

A CPU-first inference engine for mixture-of-experts language models, with a desktop app to chat with them. This
folder is the whole program: nothing is installed, and the app keeps its settings and your chats in here.

Source, issues and the full documentation: https://github.com/Rendeverance/ToRc

## Start

1. Open `ToRc`.
2. Pick a model: a `.torc` file, or a GGUF, which the app converts once to ToRc's format (it tells you first how much
   space that takes). The `models` folder is the first place it looks.
3. Press Auto-tune once: it measures this machine and keeps the fastest settings.
4. Load the model and talk.

On Linux the app needs WebKitGTK 4.1, which most desktops already have (if yours hasn't, on Ubuntu 22.04 or later and
Debian 12 or later: `sudo apt install libwebkit2gtk-4.1-0`). `unzip` keeps the programs executable; if whatever
unpacked the zip did not, run `chmod +x ToRc engine/*` in this folder.

## What is in here

| | |
|---|---|
| `ToRc` | the app |
| `engine` | the engine: `torc-server` (the chat server the app runs), `torc-tune`, `torc-run`, `torc` (converts and checks model files) and `hwscan` |
| `engine/marks` | the engine's saved state for your last 16 chats (with their tokens), so it can carry on from one without reading it all again. Nothing is kept when Settings doesn't save chats, and deleting a chat in the app leaves its file here until newer ones push it out |
| `models` | where the app looks for models first. Models can live anywhere else too |
| `chats` | your saved chats. You can choose another folder in Settings |
| `primers` | your primers: the standing instructions a chat can start with |
| `workspace` | kept for the agent's own files, for when ToRc has file tools |
| `mcps` | MCP servers: yours, and ToolFunnel when you install it. Servers added through ToolFunnel live in its folder |
| `tools` | the Node.js the app downloads if a tool needs one and you have none |
| `data` | the app's settings and benchmark results |
| `docs` | the guides |

To move ToRc, move the folder. To remove it, delete the folder. The folder has to be one you can write to, so keep it
somewhere like Documents rather than Program Files.

## Guides

- [docs/GUIDE.md](docs/GUIDE.md): converting, serving and tuning, step by step
- [docs/AGENTS.md](docs/AGENTS.md): the same, for an AI agent helping you set ToRc up
- [docs/AUTOTUNE.md](docs/AUTOTUNE.md): what Auto-tune measures
- [docs/SWITCHES.md](docs/SWITCHES.md): every setting the engine reads
- [docs/TWO-BOX.md](docs/TWO-BOX.md): two machines on one model
- [docs/EXPERT-CACHE.md](docs/EXPERT-CACHE.md): models larger than memory
- [docs/FORMAT.md](docs/FORMAT.md): the `.torc` file format
- [docs/RESULTS.md](docs/RESULTS.md): measured speeds

## Licence

MIT or Apache-2.0, at your choice (`LICENSE-MIT`, `LICENSE-APACHE`). The libraries built into ToRc keep their own
licences: `THIRD-PARTY-NOTICES.md`, `THIRD-PARTY-LICENSES.md` (the engine) and `THIRD-PARTY-LICENSES-APP.md` (the
app), with `THIRD-PARTY-DLLS.md` in the Windows on ARM zip. Models keep their own licences too.
