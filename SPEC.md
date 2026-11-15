# ToRc build sheet (SPEC.md)

What ToRc is built from: per component, what it must do, the definitions that fix its output, and where the
mathematics comes from. Kept current with the code.

Sources ToRc builds from: ggml / llama.cpp (MIT), the papers and public-domain numerical libraries named in each
section (e.g. Cephes, S. L. Moshier), my own code, and behavioural specifications like the ones below, each written
down as a definition so that ToRc's output is fixed bit for bit by the document, not by any code.

Status words: **built** (in ToRc, from this spec), **planned**, **open** (not decided).

---

## 1. Attention (CPU)

**Must:** for every query token at position p and query head h, out = softmax(scale · q·K^T) · V over the keys and
values of positions j, with j <= p (causal) and p - j < window when the layer has a sliding window (llama.cpp's
rule: `is_masked_swa`, standard type: masked when p - j >= n_swa). Grouped queries: query head h reads KV head
h / (n_head / n_kv). The cache holds per (KV head, position) a row of head_dim numbers at 8 bits (Qwen) or 16 bits
(Gemma 4) and one f32 scale; a number's value is int · scale.

**Required property: block invariance.** A token's output has the same bits whether it runs alone (decode), in
a block of a few (an MTP verify pass) or in a long block (prefill), on any number of threads. MTP's promise that
its output is byte-identical to plain greedy decoding rests on it. Everything below that fixes an order exists
for this.

**Bit-exact definition (float scores; the path every model uses):**

1. Pages: the keys and values are cut into pages of 256 positions at absolute positions (page p holds positions
   256p .. 256p+255). Token j of a call at positions pos0..pos0+n-1 reads the pages from its window's first page
   (page 0 without a window; with a window w, floor(max(pos0 + j + 1 - w, 0) / 256)) to the page holding its own
   position, floor((pos0 + j) / 256).
2. Per (KV head, page) and per slot (token j, query head h of that KV head), over the page's rows t0..t1-1, where t1
   is the page's end or the last position anyone in the work item can see:
   - **Score of row t:** the key row converted to f32 exactly (int to f32), the query (unscaled) times it as eight
     lanes of fused multiply-adds: lane l accumulates q[i+l]·k[i+l] for i = 0, 8, 16, ... (head_dim a multiple
     of 8); the eight lanes then summed as a pairwise tree, ((L0+L1) + (L2+L3)) + ((L4+L5) + (L6+L7));
     score = (sum · scale) · kscale[t], two separate multiplies in that order.
   - **Mask:** vis = min(pos0 + j + 1 - t0, the page's rows) visible rows; with a window, rows before
     lo = pos0 + j + 1 - w score -inf, and a page wholly before lo gives the empty partial (m = -inf, l = 0,
     values 0). With a QSA selection, rows of unpicked pools score -inf, and a page with none picked is empty.
   - **Weights:** m = the largest score among the first vis rows. For each row t: d = score - m;
     w = expf(d) if t < vis and d >= -87, else 0. The sum l: eight lanes of plain (unfused) adds, lane l taking
     rows t = l, l+8, l+16, ... (rows past the page's last count 0), summed by the same tree. Each w is then
     multiplied by vscale[t] (the value's scale folded into its weight; l is taken before this).
   - **expf** (`model/src/layers/act.rs`, shared with the activations): x held to [-87, 88] (IEEE maxNum / minNum, so a
     NaN becomes -87); t = fma(x, log2 e, 1.5·2^23), k = t - 1.5·2^23 (k = x / ln 2 rounded half to even);
     r = fma(-k, ln2_hi, x), r = fma(-k, ln2_lo, r) with ln2_hi the f32 nearest ln 2 and ln2_lo = -1.9046543e-9;
     p = 1.9875691500e-4, then p = fma(p, r, c) for c = 1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2,
     1.6666665459e-1, 5.0000001201e-1 (Cephes' expf coefficients, S. L. Moshier); y = fma(r·r, p, r) + 1 (r·r
     rounded first); result = y · 2^k, 2^k built by shifting t's bits 23 places and adding 127 << 23.
   - **Values:** per output number i: acc = 0, then for t ascending acc = fma(V[t][i] as f32, w[t], acc).
   - The partial: m and l, and acc[0..head_dim].
3. **Combine** per (token, query head), over its pages in ascending order: M = the largest m (if every page is
   empty, the output is 0); total = 0, out = 0; per page: e = expf(m_p - M), total = fma(e, l_p, total),
   out[i] = fma(e, acc_p[i], out[i]); finally out[i] = out[i] / total (IEEE division).

Nothing in 1-3 depends on how the pages and slots are shared among threads, which is free.

**Built:** `model/src/layers/fattn.rs`. Every model and cache width uses it.

**Gate:** every token's output in a block bit-identical to the token alone, on any thread count (blocks_are_tokens,
and torc-run dumps at different thread counts and block sizes).

**Work split (split-K, two stages):** the first stage's work items are (KV head, page, subgroup of the KV head's
query heads, tile of the block's tokens): the subgroups are halved until there are two items a thread, and the
tokens tiled until there are four, since in a long block page p is seen only by the tokens from position 256p on.
Each key or value row is converted to f32 once and shared by every slot of its item; slots are taken eight, four,
two or one at a time so their accumulators stay in registers. The partials live in two arrays per call, the (m, l)
pairs and the value sums, indexed (page, token, query head). The second stage combines one (token, query head) row
per step. On ToRc's own pool both stages run in one round: threads take items from a shared cursor, wait at a
counter until the last item is done, then take rows from a second cursor (pool::waits_ok); under rayon each stage
is a round of its own. On ARM64 the kernels are NEON intrinsics: eight numbers as two registers of four lanes, the
tree fold as two pairwise adds, values four positions at a time by lane, the exp four lanes at a time step for step
with expf; AVX2 does the same in one register; elsewhere plain Rust.

**f32 tiles for MTP verify passes (TORC_ATTN_TILE=1, ARM):** an item of 16-63 slots (below TORC_ATTN_F32_MIN's
whole-page copy) converts its page to f32 32 positions at a time (32 KB, held in L1): every slot's scores over a key
tile, then each slot's weights over the page, then the value tiles with each slot's chain carried from tile to tile
(values_tile). Scores are per position and values chained in position order, so the bits are the same
(blocks_are_tokens with the switch on).

**Open:** a 16-bit-query path (an exact integer score); not used by ToRc's models today.

## 2. Speculative decoding (MTP)

**Qwen (mtp.rs, mtp_hc.rs):** the model's own MTP block drafts tokens; each pass verifies the token and its drafts in
one block, and the output is the main model's greedy pick at every position, so the reply is the same tokens as
without drafting. How many drafts a pass verifies is chosen by `model/src/runtime/plan.rs` (DraftPlan).

**The head laid out (TORC_KLAY_HEAD=1, off by default):** a Q6_K head in the experts' 4-row layout; q6_lay_matmul
for 1-3 vectors and gemm_q6_lay from 4 (moe.rs use_q6m; test laid_out_q6_same_bits).

**The MTP block's fill over a prompt:** the prompt's rows only feed the block's attention cache, so a fill computes
their keys and values alone (attn::attn_fill_kv), with no queries, attention or output, and the same cache bits
(TORC_MTP_FILLHASH=1 runs both fills and compares; TORC_MTP_FILL_FULL=1 runs the whole block).

**Built (Gemma 4):** `model/src/models/gemma/gemma_mtp.rs` and `model/src/models/gemma/serve.rs`, per llama.cpp's
gemma4-assistant (src/models/gemma4-assistant.cpp, MIT) and common/speculative.cpp (draft-mtp, shared memory): the
drafter reads the target's cache of its last two layers (n-2 sliding, n-1 full), its first h is the target's hidden
state after the final norm, every draft sits at the same position. Drafter matrices requantised to Q4_0 (drafts never
change output: the target's greedy pick decides every token). A round drafts up to 3 and stops at the first draft the
drafter gives less than 0.9 (TORC_DRAFT_P; "plan" = the cost-model planner).

**NPU-only tier:** the CPU drafts from a copy of the NPU's last two KV layers (Gemma::mirror_kv).

## 3. Whole model on the NPU (Snapdragon X, Hexagon)

**Built:** `model/src/models/gemma/npu_gemma.rs`: the model as one ggml graph (llama.cpp's gemma4 graph, MIT) run by the
HTP backend of the ggml DLLs GenieX ships (loaded at run time, `kernels/src/platform/npu/ggml.rs`). The HTP maps about
16 GB per session and 2.9 GB per op batch.

The experts' weight multiply is built as a 2-D broadcast, [dm, k·n] by [1, k·n]: the 3-D form, [dm, k, n] by
[1, k, n], gives wrong results on the HTP for tokens 3.. of a block.

## 4. Quantised kernels

**Built:** dot products of Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q3_K / Q4_K / Q5_K / Q6_K / IQ4_NL / IQ3_S / IQ4_XS
against Q8 activations (formats: ggml, MIT); ARM SDOT / SMMLA and x86 AVX2 / VNNI (AVX or AVX-512) kernels, mine. The
bits are pinned by `kernels/tests/parity.rs`: every path for a type gives the same bits as the others for the same row
and vector, and every product is within rounding of an f64 product of the weights dequantised by ggml's definitions.

**Activations (`kernels::Q8`: each 256 numbers quantised twice, into two arrays side by side. `k` is ggml's block_q8_K,
d, qs[256], bsums[16]; `g` is the same numbers as eight 32-groups with block_q8_0's scales, d[8], qs[256] and one sum
per group, sums[8]):** per 256 numbers, amax = the largest |x| (a NaN wins), d = amax / 127, id = 1 / d (0 when d = 0),
q = round half to even of x * id clamped to +-127, bsums = sums of 16; per 32 the same with its own amax / 127 into g's
d, qs and sums. The K-quants read `k`, the 32-weight formats `g`, so a product streams only the array its weights use; a
sum over 32 of `k` (Q4_K / Q5_K mins on x86) is two bsums added where it is used. Built: `kernels/src/quant/q8k.rs`
(NEON / AVX2; every step is per number, a max or an integer sum, so vectors keep the bits; parity test against the
scalar form).

**Weights to f32:** ggml's definitions (ggml-quants.c dequantize_row_*, MIT) for F32, F16, BF16, Q4_0, Q4_1,
Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, one product per number in the order written there:
`kernels/src/quant/dequant.rs`; the parity tests' reference.

**The experts' hidden activations:** gate and up combine as SwiGLU (SiLU of the gate times up) or, for Gemma 4, GeGLU
(GELU in its tanh form, computed as x / (1 + e^-2z)), then go through the quantiser above. The exp is
`model/src/layers/act.rs`: x held to [-87, 88], k = round-half-even(x / ln 2) by the 1.5 x 2^23 shift, r = x - k ln 2 in
two fused steps (the f32 ln 2 and its remainder), the Cephes expf polynomial in r, and 2^k written into the exponent
field. Every step is a correctly rounded IEEE operation, so the bits are the same on every machine. **qwen4exp's (ggml's
Q8_K scales):** the 256's multiplier 127 / amax and d its reciprocal; each 32's d = amax / 127 stored through fp16
(round to nearest even), multiplier 1 / d. Built in `kernels/src/quant/q8k.rs` with the f32-to-fp16 conversion,
identical on ARM and x86 (the fp16 conversion tested on every exponent's edge cases and a million random floats,
tests/gdn32.rs).

**Float dot and axpy (`kernels/src/platform/cpu/fvec.rs`):** dot = sixteen lanes of fused multiply-adds (number i in
lane i mod 16), eight more steps in lanes 0-7 if eight remain, lane l plus lane l + 8, fold8, then the last n mod 8
products fused onto the sum in order; axpy = one fused multiply-add a number. One definition for every platform (the
dims it is used on are multiples of 8).

**Q4_0 rows against one vector (decode):** per row and block of 256, lane k = sub-block k: the exact sum of nibble x
activation less 8 x the activations' sum, then acc[k] = fma(d_w[k] x d32[k], sum, acc[k]); fold8 at the end. ARM:
`kernels/src/platform/cpu/neon.rs` q4_0_dot_rows1 (a row at a time, the default; q4_0_dot_rows, two rows a pass, with
TORC_DOT=ours2), x86: `kernels/src/platform/cpu/x86dot.rs` (AVX2, two rows a pass, vpdpbusd where the CPU has AVX-VNNI).
The paths are identical (parity tests).

**Q4_0 repacked, 4 rows interleaved (ARM):** each block position of 4 rows held as their 4 fp16 scales, then per pair of
rows two 16-byte registers whose nibbles are SMMLA's operands as they stand (`q4_0_repack4`); a permutation, so the same
integers and scales. Kernels on it: `q4x4_matmul` (1-3 vectors, SDOT, lanes = the 4 rows, the 8 activation bytes
doubled) and `gemm_q4x4` (4+ vectors, SMMLA; 8 rows a pass from 16 vectors, pairs in panels). Per (row, vector,
sub-block) the order above, so identical to the row kernels (parity tests). A pass reads 4 rows as one stream with
nothing to unpack but a mask and a shift. The copy lives in RAM (`model/src/runtime/repack.rs`): on by default for Gemma
on the CPU, TORC_Q4X4=1 for Qwen, TORC_Q4X4=0 off. The table of copies is a published snapshot, so a lookup takes no
lock.

**Q8_0 and Q6_K rows against one vector (ARM):** Q8_0 as Q4_0 above without the nibbles or the bias (lane k = sub-block
k, fma of d_w x d32, fold8); Q6_K per block of 256 the exact sum of scale_g x sum (q - 32) a over its 16 groups, then
acc = fma(d_w x d, float(sum), acc), one step a block (gemm_q6k's order). Both identical to the multi-vector kernels
over one vector (parity tests). `kernels/src/platform/cpu/neon.rs` q8_0_matmul over one vector / q6k_dot_rows, default
on ARM.

**No other engine's code:** no other inference engine's code is built in; every kernel above is this workspace's, on ARM
and x86 (kernels/src/platform/cpu/x86mm.rs: the row kernels for every type above).

## 5. File formats

**ToRc layout 3** ([FORMAT.md](FORMAT.md)): each routed expert's gate, up and down sit together in one 4 KiB-aligned pack, packs
in expert order, so an expert is one read; a directory of tagged records; BLAKE3 per region and CRC-32C per tensor
and pack; and for every tensor and expert matrix, the GGUF bytes it came from, so `torc check --against` proves a
file against its source. The converter reads a split GGUF directly and never requantises.

**Planned:** hot experts grouped within a layer, kernel layouts baked in at convert (ARM's laid-out K-quant experts),
and a GGUF loader for models that fit in RAM.

## 6. Thread pool

**Must:** run a job on n threads at once, the caller being thread 0, and return when every thread has finished
it (one fork-join, no allocation per run); keep the workers hot between runs (a decode step is hundreds of runs
a few tens of microseconds apart) without burning the cores when the engine is idle; one worker per given CPU,
pinned, so the split of the work (decided by the caller, `par_for` and friends) lands on the cores it was
sized for. The numbers a job computes never depend on the pool (items are the same calls on any thread).

**Design:** a generation counter and a pointer to the job; a run stores the job, bumps the generation (sequentially
consistent), wakes any worker that went to sleep, runs its own share, then spins until every worker has marked the
generation done (a mark per worker, each on its own cache line, so no line is shared by every core). A worker spins
on the generation for a time (TORC_SPIN_US, default 200 microseconds; checked every 1024 spins), then marks itself
asleep, checks the generation once more and parks; the caller unparks the workers marked asleep after bumping the
generation (the two sequentially consistent accesses on each side make a lost wake-up impossible). A job started from
inside a job is a bug (the pool is one level), reported and aborted, not deadlocked.

**Switch:** TORC_POOL=rayon.

## 7. Tokenizer

**Must:** turn text into the model's token ids and back exactly as the reference does, from the model file's
own metadata (tokenizer.ggml.model / pre / tokens / token_type / merges), with no tokenizer code of another
engine linked in.

**Definition (sources: Hugging Face tokenizers' BPE, GPT-2's byte-to-character map, llama.cpp's
llama-vocab.cpp and unicode tables, MIT):** special tokens (types control, user-defined, unknown) are cut out of
the text first, longest first, but only when the caller asks for special tokens (the reference's rule; llama.cpp
and Hugging Face cut user-defined ones always). The rest is split into words: for `gpt2`/`qwen35`, by Qwen 3.5's
pre-tokenizer pattern (first alternative that matches: contractions; an optional non-letter non-digit then
letters and marks; one digit; an optional space then other characters then line breaks; whitespace to its last
line break; whitespace not followed by a non-space; whitespace), each word's bytes then mapped to characters by
GPT-2's table; for `gemma4`, spaces become U+2581 and the text is cut into runs of line breaks and runs of the
rest (a run of line breaks that is a token is that token). In each word the adjacent pair with the lowest merge
rank is merged, the leftmost of equals, until no pair has a rank; each piece is its token, or else its bytes'
tokens (`<0xXX>` for gemma4). Unicode categories (L, M, N) from llama.cpp's generated table
(`model/src/chat/uniflags.rs`), whitespace = Unicode White_Space.

**Built:** `model/src/chat/tok.rs` (a model kind it does not know is refused, with a message). **Gate:** identical ids
to the reference tokenizer on prose, source code and mixed text (chat markup of both families, several scripts,
emoji with joiners, combining marks, tabs, CRLF), whole and line by line, with and without special tokens; every
token's decoded bytes identical.

## 8. Experts from SSD

**Must:** run a MoE model whose experts do not fit in RAM at the speed the drive allows, with the same bits as
reading every expert from RAM. **Built:** the expert cache (`model/src/runtime/ecache_own.rs`, [EXPERT-CACHE.md](EXPERT-CACHE.md)).

**Design, each part kept only if it pays on the target drives:**
1. *Self-tuning probe* (**planned**, with the drive tuning in [AUTOTUNE.md](AUTOTUNE.md)): unbuffered reads at request sizes
   64 KiB - 8 MiB and queue depths 1-64, from several threads, over the model file itself. Out: the request size and
   depth where throughput flattens, and the latency at each, for the cache to read instead of its defaults.
2. *Slots that match the drive:* in the ToRc file each expert (gate, up, down together) is one slot, aligned and
   sized to whole requests; one expert = one read. **Built** (layout 3).
3. *Hot-pack* (**planned**): the routing profile (counts and co-selection pairs per layer: usage_flush, TORC_ROUTES)
   orders the slots within a layer so that experts chosen together sit next to each other; a miss on one then reads its
   likely partners in the same sequential run, if the probe says longer runs are cheaper per byte than separate
   requests.
4. *The cache:* RAM slots per layer (the budget from installed RAM), the profile pre-warming it at start; reads issued
   as overlapped I/O on Windows (plain positional reads on Linux so far), compute starting on whichever experts arrive
   first (the MoE sum stays in id order, so the bits do not depend on arrival). **Built** (ecache_own.rs).
5. *Prediction* (**built, off by default**: `TORC_READ_AHEAD`): the next layer's router run on the current hidden
   state as a cheap guess, prefetching only the experts it ranks with high confidence; on by default only once the
   guess is right often enough on real prompts.

**Decided:** tune per machine, with the results in ToRc's cache folder beside the other tuned values ([AUTOTUNE.md](AUTOTUNE.md)),
never in the model's folder. **Open:** the drive in the key (a model moved to another drive probes again). The
per-machine part is runtime only (request size, queue depth, how many neighbouring slots one read
takes); the file stays the same on every machine, with 4 KiB-aligned slots and the hot-pack order from the routing
profile (a property of the model, not the drive), so one file converts once and copies anywhere.
