# Expert cache

ToRc's cache for the routed experts of a model larger than RAM, on one machine or split over two. It is
`model/src/runtime/ecache_own.rs`, behind the interface in `model/src/runtime/ecache.rs`.

## What the engine asks of it

- `open(path, budget, io_threads, model id, ...)`: map (layer, expert) to its pack in the file (the file's EXPERTS
  records: each pack's offset, per-layer pack bytes), reserve `budget` bytes of slots, and start loading the experts
  this machine has used most (its usage file, below).
- `take(layer, ids)`: pointers to those experts' packs, loaded before it returns; good until the cache is asked again.
- `ask(layer, ids)` then `settle()`: `take` without the wait, with a flag per pack saying whether it is loaded yet, so
  the caller can work on the loaded ones while the rest come in; `settle` returns when they have.
- `read_ahead(layer, ids, max)`: start loading the next layer's likely experts into slots no request holds.
- `held(layer, ids)`: which of them are loaded or loading (the fast mode's question; it changes nothing).
- `layer_slots(layer)`, `stats()`, `usage_flush()`: slots per layer; hits, misses, bytes read and time waited; the
  session's counts added to the usage file.

## Design

1. **One arena per slot size.** One allocation per distinct pack size, page-aligned (4 KiB), so every slot takes an
   unbuffered read straight from the file, with no copy through the OS page cache.
2. **Replacement: CLOCK with a frequency hint.** Every slot carries a little credit: a use adds to it, the sweeping
   hand spends it, and experts the usage file ranks hot start with more. When a pass over the layers asks for more
   experts than an arena holds, the arena evicts by layer distance instead: the slot whose layer comes round again
   last. A request's own slots, and slots with a read in flight, are never evicted.
3. **I/O: unbuffered reads.** On Windows they are overlapped (`FILE_FLAG_NO_BUFFERING` + `FILE_FLAG_OVERLAPPED`,
   `ecache_overlapped.rs`), started by the thread that asks for them, so the compute threads never share their cores
   with reader threads. On Linux the file is opened with `O_DIRECT` and reader threads (`TORC_IO_THREADS`) make
   positional reads. Where a Linux file system refuses `O_DIRECT` (tmpfs, some network and FUSE file systems) the
   server says so once and reads through the page cache instead, which costs a copy and nothing else.
4. **Request path.** `ask` assigns victims for the misses, queues their reads and returns every slot pointer with
   a ready mask; `settle` waits for the queued reads only. Read-ahead uses the same queue and never takes a slot a
   request holds.
5. **Warm-up.** On open, the experts with the highest counts in the usage file are read in the background until the
   budget is full.
6. **Usage file.** Per machine and model file: `usage/<the first 16 bytes of the model id, in hex>` in ToRc's cache
   folder (`%LOCALAPPDATA%\torc`, else `$XDG_CACHE_HOME/torc`, else `~/.cache/torc`). Little-endian: `TORCUSE1`,
   the 32-byte model id, u32 layers, u32 experts per layer, then a u64 request count per (layer, expert), layer by
   layer. Each session adds its counts (`usage_flush`, and when the cache closes). The NPU's hot-expert files
   (`TORC_NPU_PROFILE`) use the same layout, with an all-zero model id. A model that fits in RAM has no cache, so on one
   machine with the NPU on, torc-server keeps this file itself, and the NPU's hot experts are chosen from it the next
   time the model loads.

The bytes in a slot are the file's, so a model gives the same bits with the cache, without it, and on two machines.

## Settings

`TORC_CACHE_MB=m` sets the budget (0 turns the cache off). By default the cache is on when the file does not fit in
80% of the installed memory, and its budget is the free memory less the weights that are always resident and a 6 GB
margin (on a two-machine worker, less its own layers' share and 3 GB), never under 2 GB. `TORC_IO_PIECE_KB` (read
size, default the whole slot), `TORC_IO_DEPTH` (reads in flight, default 64) and `TORC_OWN_PREFETCH=0` (no read-ahead)
tune the reads. On Windows the reads are overlapped, and `TORC_IO_THREADS` only matters with `TORC_IO=threads`.
`TORC_IO_BUFFERED=1` reads through the page cache on either system, for comparing the two.
