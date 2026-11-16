// explain.js - what the `i` buttons in Advanced say, for the switches that are real settings. SWITCHES.md is the
// reference; this is the same thing for someone meeting the switch for the first time. Each entry was checked
// against the code that reads the switch.
window.EXPLAIN = {
  TORC_MTP:
    'The most tokens drafted in one pass, on the models with an MTP block (the Qwen ones). 8 by default, 16 at ' +
    'most, and 0 turns drafting off. The server drafts with the model’s own prediction block and checks the drafts ' +
    'in one pass. At the default settings the reply is the same tokens either way.',
  TORC_MTP_FILE: 'A separate file with the MTP block in it, for a model whose own file doesn’t carry one.',
  TORC_DRAFT:
    'Gemma 4’s drafting: the path to its assistant model. Gemma drafts with that model, not with an MTP ' +
    'block, and each draft is checked by the main model.',
  TORC_DRAFTS:
    'A file where the draft planner keeps what it has learned. Only torc-run reads it; torc-server keeps ' +
    'its planner in memory, so this does nothing for the server ToRc starts.',
  TORC_DECODE_THREADS:
    'How many threads write a reply. Reading a prompt uses all of them, but writing is different: ' +
    'on a CPU with two threads per core, one per core can write faster. The tuner sets this.',
  TORC_BLOCK:
    'How many prompt tokens are read in one go. torc-tune tries the sizes on your machine and saves the best one.',
  TORC_BLOCK_LONG:
    'A second block size for long prompts, written b or b,n: prompts of n tokens or more (2048 unless ' +
    'you say) are read in blocks of b. Off by default.',
  TORC_CACHE_MB:
    'The expert cache size in MB. When a model is bigger than about 80% of your installed memory the ' +
    'experts come off the SSD, and this cache keeps the most used ones in memory. By default it takes the free ' +
    'memory less the rest of the model and a 6 GB margin. Bigger isn’t always better: too big and the machine ' +
    'starts paging. Set 0 to turn it off. EXPERT-CACHE.md has the details.',
  TORC_CACHE_LAYERS:
    'The layers this machine loads ahead, written a-b: its cache warm set and the NPU’s hot experts. On two ' +
    'machines each one takes its own half by default, so set it only to override that.',
  TORC_WARM:
    'Whether the model’s file is read into memory at load. By default it is when the model fits. 0 skips it ' +
    'for a quicker start and a slower first prompt, 1 forces it, and wait does it before the server takes requests.',
  TORC_NPU:
    'Turns the NPU on. Current support includes Snapdragon X NPUs. Intel and AMD NPU support is planned when I ' +
    'have access to suitable hardware or contributors can assist. It reads ' +
    'prompts faster by putting part of the work on the NPU. It isn’t bit-exact (the NPU rounds differently), so it ' +
    'only touches prompts, never replies. Run npucheck first to make sure it works. The Load screen’s NPU switch ' +
    'sets this for you.',
  TORC_NPU_DIR: 'Where the ggml build for the NPU is. The NPU path needs it, and GUIDE.md says how to get it.',
  TORC_GEMMA_NPU: 'Which parts of a Gemma model go to the NPU.',
  TORC_NPU_PROFILE:
    'A usage file to take the NPU’s hot experts from. On one machine you don’t need it: the engine ' +
    'learns which experts your work uses most and gives those to the NPU from the next time the model loads.',
  TORC_NPU_HOT: 'How many hot experts a layer the NPU gets. 16 by default, 0 for none.',
  TORC_GPU_ATTN:
    'Set 1 and part of a long prompt’s attention runs on the GPU while the CPU does the rest. It needs an ' +
    'engine built with the GPU feature, and only joins in once a prompt is a few thousand tokens long. Like the NPU ' +
    'it isn’t bit-exact, so it never touches replies. The Load screen’s GPU switch sets this for you.',
  TORC_GPU: 'Which GPU, by part of its name. It takes the integrated GPU by default.',
  TORC_GPU_SHARE:
    'The GPU’s share of each block’s work, 0 to 1. Left alone, the engine works out the split on your machine ' +
    'and keeps it for next time.',
  TORC_GPU_CHECK:
    'Set 1 and it runs the GPU’s tokens again on the CPU and prints the largest difference. Use it ' +
    'once on a new GPU or driver. It’s slow, so don’t leave it on.',
  TORC_Q4X4:
    'A repacked layout for Q4_0 weights on ARM machines with i8mm (Snapdragon X). Off by default on the ' +
    'Qwen models, though my two Snapdragon boxes run the 35B faster with it. Worth a try on yours.',
  TORC_ATTN_TILE: 'Tiled attention. Off by default. Like TORC_Q4X4, it helps the 35B on my Snapdragon boxes.',
  TORC_MIX_MM: 'For Flash Next models on ARM with i8mm. Set 1 and a prompt’s mixer products use SMMLA. Prompts only.',
  TORC_X86_PF_BLOCKS: 'On x86, how far ahead the kernels prefetch, in blocks. 16 by default, 0 turns it off.',
  TORC_LARGE_PAGES:
    'Large pages for the expert cache. On by default on Windows, where your account needs the "Lock ' +
    'pages in memory" right for them; without it the cache uses ordinary pages. Set 0 to turn them off.',
  TORC_EXPERTS_LARGE:
    'Large pages for the experts of a model that fits in memory. It does nothing when the expert ' +
    'cache is in use, or with TORC_Q4X4=1 on, since the repacked copies are what get read then. They help on some ' +
    'machines, but one that has been up for days can’t always find the memory for them.',
  TORC_KLAY_LARGE: 'Large pages for the laid-out copies of K-quant matrices. ARM only.',
  TORC_TAIL_SKIP:
    'Fast mode, and it changes the output. The value is a rank r, from 1: an expert that every token ' +
    'ranks r or lower and that isn’t in RAM is skipped, not fetched from the SSD. With 1, everything not in RAM ' +
    'except each token’s top expert is skipped. Off by default. Only for when you’d rather have the speed than ' +
    'the exact reply.',
  TORC_HOST:
    'The address the server listens on. This machine only by default. Set it to open the server to your ' +
    'network, and remember there’s no password on it. For torc-server run by hand: the app keeps its own server ' +
    'on this machine whatever this says.',
  TORC_PIPE_BIND:
    'On a worker: the address it listens on. By default that’s every interface, and a worker has no ' +
    'authentication, so on a shared network set it to the address of the link between your two machines.',
  TORC_IDLE_SAVE:
    'After this many seconds without a request the server saves the current chat’s state to disk, so a ' +
    'restart comes back to it without reading the prompt again. 60 by default, 0 for never.',
  TORC_IDLE_SAVE_MIN:
    'A one-message chat is only saved when idle if its prompt is at least this many tokens. 8192 by ' +
    'default. A chat with two or more turns is always saved.',
  TORC_MARKS: 'How many saved conversation states are kept on this machine. 16 by default.',
  TORC_MARK_DIR: 'The folder the saved conversation states go in.',
  TORC_TOKENIZER: 'For a model with a sliced vocabulary: its original tokenizer.',
  TORC_VOCAB_MAP: 'For a model with a sliced vocabulary: the map back to the original token ids.',
  TORC_NGRAM_IDMAP: 'For a model with a sliced vocabulary: the id map its n-gram table needs.',
  TORC_TUNE_LENS: 'The prompt lengths torc-tune measures with. Change it to tune for the lengths you actually use.',
  TORC_TUNE_TEXT: 'The text torc-tune builds its prompts from. Change it to tune on text like your own.',
  TORC_PIPE:
    'The second machine, as host:port. Start the worker there first. The Load screen’s Second machine box ' +
    'sets this for you. TWO-BOX.md has the rest.',
  TORC_PIPE_SPLIT:
    'The first layer the worker runs. Leave it and the engine uses the balance it learned on earlier ' +
    'runs of a model this shape, or half and half the first time.',
  TORC_PIPE_PULL:
    'Set 1 and both machines read the prompt, then the reply runs on this machine alone. That’s the ' +
    'one to use for a model that fits on one machine.',
  TORC_PIPE_TIMEOUT:
    'How many seconds this machine waits on the worker before it calls the link dead. 90 by ' +
    'default, 0 waits forever.',
  TORC_PIPE_FALLBACK:
    'Set 1 and if the worker can’t be reached when the model loads, the model runs on this ' +
    'machine alone. Only for a model file that holds every layer. Without it the load stops with the error.',
  TORC_IO_THREADS:
    'How many threads read experts off the SSD. On Windows the reads are overlapped by default and ' +
    'need no reader threads, so this only matters with TORC_IO=threads.',
  TORC_IO_DEPTH:
    'How many expert reads are in flight at once. 64 by default. Only matters for a model bigger than your RAM.',
  TORC_IO_PIECE_KB:
    'The size of each read off the SSD, in KiB (a multiple of 4). The default reads a whole expert ' +
    'slot at once. Some drives do better with smaller pieces.',
  TORC_IO_SORT:
    'Expert reads go out in the order they sit on the drive. Set 0 to send them in the order they were asked for.',
  TORC_OWN_LARGE:
    'Copies the weights that are always in memory into large pages. It also keeps an owned copy of ' +
    'them, which counts against the cache budget, so measure before you leave it on.',
  TORC_OWN_PREFETCH: 'Set 0 to turn the expert cache’s read-ahead off.',
  TORC_CHECKPOINT_MIN:
    'While a long prompt is read on two machines the server keeps checkpoints, so a dropped link ' +
    'costs one stretch and not the whole prompt. This is the fewest tokens worth a checkpoint, 512 by default.',
};
