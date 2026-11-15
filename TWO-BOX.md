# Two machines as one: the layer pipeline

Two machines run one model: the main machine runs the first layers of every pass, the worker the rest and the
vocabulary head. Every layer runs the same code on the same inputs on whichever machine holds it, so the output is
bit for bit what one machine gives. Code: `model/src/runtime/pipe.rs`.

## Run it

Both machines need the same model file (any path). The model is split by layers, not by file: each machine maps the
whole file but only ever reads its own layers from it, so each one's memory holds its half. Where the split falls is
learned (half to start with, then whatever balanced the two machines on earlier runs of that model).

In the desktop app: on the worker, tick "This is the second machine" on the Load screen, pick the network the main
machine reaches it on under "Listen on" (it waits on that one only), and load the model; the status line shows its
address. On the main machine, put that address in the Second machine box and load the same model. From a terminal:

On the worker:

    torc-run <model.torc> serve-pipe=50052 ctx=16384

`ctx=` is the longest chat the worker will take (4096 if you leave it out). Make it at least the context the main
machine runs with, or longer chats fail on the worker. A worker reads no tokens, so it needs no tokens file, and it
runs on every core of the fastest class.

On the main machine, any torc-run command with:

    TORC_PIPE=<worker address>:50052 torc-run <model.torc> <tokens> <n> 8 block=64 ...

`torc-server` takes `TORC_PIPE` the same way, and that is how you'd normally use it: start the worker, then the
server on the main machine.

A worker has no authentication: whoever can reach its port can use it. Run it on a link you trust, and on a machine
with other networks attached set `TORC_PIPE_BIND=<the link's address on the worker>` so it listens there only.

Any address the main machine can reach will do; a fast direct link (USB4, 10 GbE) helps. The worker serves one run at
a time and resets between runs.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `TORC_PIPE=host:port` | off | use the worker |
| `TORC_PIPE_BIND=address` (worker) | every interface | the address the worker listens on |
| `TORC_PIPE_SPLIT=S` | learned | the first layer the worker runs; otherwise the balance learned on earlier runs of a model of this shape (saved in ToRc's cache folder, %LOCALAPPDATA%\torc or ~/.cache/torc, running mean), else half |
| `block=N` (torc-run) | - | prompt block size: smaller blocks keep both machines busy |
| `TORC_PIPE_PULL=1` | off | gen mode: after the prompt, pull the worker's layer states back and reply on this machine alone |
| `TORC_CACHE_LAYERS=a-b` | all | warm only those layers' experts from the profile (a machine's own half) |

## When to use which

- Model bigger than one machine: both machines for everything. Each machine caches only its half of the experts.
- Model that fits one machine: both machines for the prompt, then `TORC_PIPE_PULL=1` so the reply runs on one machine
  (no network hop per token).
- MTP drafts work over two machines for the block-path models (the worker returns the hidden rows with the logits);
  not for qwen4exp (Flash Next).

## Only the worker's half on the worker

The worker reads only its own layers, so it can hold only those: less to copy, and half the disk. Pick the split S
(the first layer the worker runs), then on the main machine:

    torc shard-plan <model.torc> S-999          (what that half holds, and its size)
    torc pack <model.torc> half.part S-999

Copy half.part to the worker and there run `torc unpack half.part <model.torc>` (a new file: it won't write over one
that exists). That writes a file of the model's full length holding only those layers' bytes (a sparse file, so the rest
takes no disk). Then, on the worker, `torc check <model.torc> --layers S-999` checks those layers against their hashes
and writes `<model.torc>.layers`, which tells the engine the file is a half and keeps the split where it was cut. Set
`TORC_PIPE_SPLIT=S` on the main machine as well. With the whole file on both machines none of this is needed, and the
app assumes the whole file.
