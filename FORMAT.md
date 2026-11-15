# The ToRc model file, layout 3

A ToRc file holds one model for the engine: its metadata, its resident tensors (read on every token) and its routed
experts (read when the router picks them). It is made from a GGUF file without changing any weight: every weight byte
is a byte of the GGUF, or a zero added to widen a matrix (section 6). Little-endian throughout.

Why not read the GGUF as it is: a GGUF stores an MoE layer's experts as three 3-D tensors, so one expert is three reads
from three places. In ToRc each expert is one contiguous read, which matters when experts stream from disk, and the
file checks itself (section 7).

## 1. Regions

```
| header 4 KiB | directory | resident tensors | expert packs |
```

Each region starts on a 4 KiB boundary. The directory says what is where; the resident region holds the tensors
every token reads; the expert region holds one pack per expert.

## 2. Header (4096 bytes)

| at | type | field |
|---|---|---|
| 0 | 8 bytes | magic `TORC\r\n\x1a\n` (the bytes 54 4F 52 43 0D 0A 1A 0A) |
| 8 | u64 | the file's length in bytes |
| 16 | u32 | layout: 3 |
| 20 | u32 | regions listed: 3 |
| 24 | 48 bytes | the directory: u64 offset, u64 length, 32-byte BLAKE3 |
| 72 | 48 bytes | the resident region, the same way |
| 120 | 48 bytes | the expert region, the same way (hash zero when the model has no routed experts) |
| 168 | 32 bytes | model id: BLAKE3 over the three region hashes, directory first |
| 200 | | zeros up to 4092 |
| 4092 | u32 | CRC-32C of bytes 0..4092 |

The architecture is not repeated here: it is `general.architecture` in the metadata. A reader refuses a file with
another layout number, a bad CRC, or a size or offset that does not hold.

## 3. Directory

A run of records, each `u16 tag, u16 flags, u32 payload bytes`, then the payload, then zeros to the next multiple of 8.
Flag bit 0 means "needed": a reader that does not know the tag must refuse the file. A record without it may be
skipped. Strings are a u32 length then UTF-8 bytes.

| tag | record | count |
|---|---|---|
| 1 | METADATA: the model's key/value pairs, kept in GGUF's own encoding so no value is re-typed | one, needed |
| 2 | SOURCE: the files the model came from | at most one |
| 3 | TENSOR: one resident tensor | one per tensor, needed |
| 4 | EXPERTS: one MoE layer's experts | one per MoE layer, needed |

**SOURCE**: u16 file count; per file: u64 bytes, 32-byte BLAKE3 (zero when not hashed), string name; then string
converter (its name and version).

**TENSOR**: string name (GGUF's), u64 offset within the resident region, u64 length, u32 CRC-32C of the bytes, u32
ggml type, u32 dimension count n (1-4), then n u64 dimensions (the row length first, as GGUF has them), then the
origin (section 5).

**EXPERTS**: u64 offset of expert 0's pack within the expert region, u64 pack length, u32 layer, u32 expert count,
u32 experts used per token, u32 input width, u32 feed-forward width (gate's and up's rows; down's rows are this, or
this rounded up to 256 when only down was widened), u32 piece count (3), then the pieces gate, up, down, then one u32
CRC-32C per expert's pack. Expert e's pack starts at `first + e x pack length`.

A **piece** is one expert's matrix inside its pack: string name (the GGUF tensor it came from), u64 offset within the
pack, u64 length, u64 stride (the distance between consecutive experts' matrices in that tensor), u32 ggml type, u32
rows, u32 columns (the row length), u32 role (0 gate, 1 up, 2 down), then its origin.

## 4. Packs

A pack is everything one expert needs: its gate, up and down matrices, each starting on a 4 KiB boundary within the
pack, the pack a whole number of 4 KiB. Packs sit in expert order, back to back, layer after layer, so a pack is one
aligned read (the expert cache reads unbuffered, which needs sector-aligned offsets and lengths).

A GGUF that fuses gate and up into one tensor (`ffn_gate_up_exps`, gate rows first) gives two pieces, each naming the
fused tensor and marking which half it is.

## 5. Origin

`u64 source offset, u32 source file (0xFFFF: none), u32 change bits`. The offset is where the bytes begin in that file
(for a piece, where expert 0's do). Change bits:

| bit | meaning |
|---|---|
| 0 | rows added: zero rows after the source's (gate, up) |
| 1 | row tails added: each row extended with zeros (down) |
| 2 | the first half of a fused tensor |
| 3 | the second half of a fused tensor |

No bits set means a plain copy, which is what lets `torc check --against model.gguf` compare every tensor and piece
with the bytes it came from.

## 6. Widening

A feed-forward of width F that is not a multiple of 256 is stored at F' = 256 * ceil(F / 256), since the hidden
activations are quantised 256 at a time. This applies to routed experts, a dense branch (`ffn_gate`, `ffn_up`,
`ffn_down`) and a shared expert (`*_shexp`):

| Matrix | Stored as | The bytes added |
|---|---|---|
| gate, up | F' rows | rows F .. F'-1, all zero |
| down | F' columns | in every row, the blocks of columns F .. F'-1, all zero |

An all-zero block of any weight type used here decodes to zeros, so the added hidden units are exactly zero and every
product is unchanged. `qwen4exp` widens down only: its forward pass pads the hidden state to F' itself.

## 7. Checks

- At open: the header CRC, the directory hash, every size and offset against the file, every tensor's and piece's
  length against its type and shape.
- `torc check`: the region hashes, then every tensor's and pack's CRC-32C.
- `torc check --layers a-b`: a pipeline machine's share (section 8): the tensors and packs it holds.
- `torc check --against <gguf>`: every tensor and piece against its source bytes.

CRC-32C (Castagnoli) has instructions on both x86 (SSE4.2) and ARMv8, so a pack can be checked as it lands.

## 8. Pipeline shards

A machine running layers a..b of a model reads the directory, the resident tensors that are not other layers', and its
layers' packs. `torc pack` streams just those byte ranges and `torc unpack` writes them into a sparse file of the full
length, which opens as the whole file does. Its region hashes cannot hold (the holes read as zeros), so a shard is
checked by `torc check --layers` and marked by a `<model>.layers` file holding `a-b`.

## 9. Block kinds

The file does not store the model's structure. The reader finds each layer's blocks from its tensor names: gated
DeltaNet (`ssm_beta` / `ssm_ba`), Mamba-2 (`ssm_in`, `ssm_a`), attention (`attn_q`, `attn_qkv`, `attn_q_a`), MoE
(`ffn_gate_inp`), a dense feed-forward (`ffn_up` / `ffn_down`), hyper-connection mixing (`hc_`) and the MTP block
(`nextn.`).
