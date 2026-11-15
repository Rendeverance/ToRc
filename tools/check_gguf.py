"""A ToRc file made from a GGUF, checked byte for byte against that GGUF, with none of ToRc's Rust code.

usage: python check_gguf.py <model.torc> <model.gguf>

The GGUF is read by llama.cpp's own reader (the `gguf` package, MIT, one file: give a merged GGUF); the ToRc file by
the lines below, written from FORMAT.md alone. No weight is requantised, so every byte of the ToRc file must be one of:
  - a resident tensor's GGUF bytes, or those bytes widened with zeros to whole 256s (zero rows on gate and up, a zero
    tail on each down row), under the same name and type;
  - in an expert's pack: its slice of blk.L.ffn_{gate,up,down}_exps (or its half of ffn_gate_up_exps), widened the
    same way, at the piece's place in the pack, with zeros everywhere else in the pack;
and METADATA must be the GGUF's key/value block byte for byte. Every GGUF tensor must land somewhere. The header's
CRC-32C and the directory's BLAKE3 are not recomputed here (the Rust reader checks them at every open).
Exit status 0 only when all of that holds.
"""
import struct
import sys
import time

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader

TAG_METADATA, TAG_SOURCE, TAG_TENSOR, TAG_EXPERTS = 1, 2, 3, 4


class Cur:
    def __init__(self, b):
        self.b, self.p = b, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise SystemExit('directory record runs past its end')
        s = self.b[self.p:self.p + n]
        self.p += n
        return s

    def u(self, fmt):
        n = struct.calcsize(fmt)
        v = struct.unpack('<' + fmt, self.take(n))
        return v if len(v) > 1 else v[0]

    def str(self):
        return self.take(self.u('I')).decode()

    def origin(self):
        offset, file, changes = self.u('QII')
        return file, changes, offset


class Torc:
    def __init__(self, path):
        self.f = open(path, 'rb')
        h = self.f.read(4096)
        if h[:8] != b'TORC\r\n\x1a\n' or struct.unpack_from('<I', h, 16)[0] != 3:
            raise SystemExit(f'{path}: not a ToRc layout 3 file')
        d_off, d_n = struct.unpack_from('<2Q', h, 24)
        self.res_off = struct.unpack_from('<Q', h, 72)[0]
        self.exp_off = struct.unpack_from('<Q', h, 120)[0]
        c = Cur(self.at(d_off, d_n))
        self.metadata, self.tensors, self.experts = None, [], []
        while c.p < len(c.b):
            tag, flags, n = c.u('HHI')
            p = Cur(c.take(n))
            c.take((8 - n % 8) % 8)
            if tag == TAG_METADATA:
                self.metadata = p.b
            elif tag == TAG_TENSOR:
                name = p.str()
                off, nb, _crc, ty, nd = p.u('QQIII')
                dims = [p.u('Q') for _ in range(nd)]
                self.tensors.append((name, ty, dims, off, nb, p.origin()))
            elif tag == TAG_EXPERTS:
                first, pack_bytes = p.u('QQ')
                layer, ne, _nu, d_in, width, npc = p.u('6I')
                if npc != 3:
                    raise SystemExit(f'layer {layer}: {npc} pieces')
                pieces = []
                for role in range(3):
                    name = p.str()
                    at, nb, stride = p.u('QQQ')
                    ty, rows, cols, r = p.u('4I')
                    if r != role:
                        raise SystemExit(f'{name}: pieces out of order')
                    pieces.append((name, ty, rows, cols, at, nb, p.origin(), stride))
                self.experts.append((layer, ne, first, pack_bytes, pieces))
            elif flags & 1 and tag != TAG_SOURCE:
                raise SystemExit(f'unknown needed record {tag}')
        if self.metadata is None:
            raise SystemExit('no METADATA record')

    def at(self, off, n):
        self.f.seek(off)
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit('short read')
        return b


def row_bytes(ty, cols):
    blk, size = GGML_QUANT_SIZES[GGMLQuantizationType(ty)]
    if cols % blk:
        raise SystemExit(f'type {ty}: {cols} columns are not whole blocks')
    return cols // blk * size


def same_widened(ours, src, rows, rb, new_rb):
    """ours (new_rows x new_rb bytes) == src (rows x rb) widened with zeros: each row's tail and every extra row zero"""
    o = np.frombuffer(ours, np.uint8)
    s = np.frombuffer(src, np.uint8)
    if rb == new_rb and len(o) == len(s):
        return np.array_equal(o, s)
    if len(o) % new_rb or len(s) != rows * rb or len(o) // new_rb < rows or new_rb < rb:
        return False
    o = o.reshape(-1, new_rb)
    return np.array_equal(o[:rows, :rb], s.reshape(rows, rb)) and not o[:rows, rb:].any() and not o[rows:].any()


def main():
    t0 = time.time()
    tp, gp = sys.argv[1], sys.argv[2]
    m = Torc(tp)
    g = GGUFReader(gp, 'r')
    gf = open(gp, 'rb')
    src = {t.name: t for t in g.tensors}
    used = set()
    bad = []

    def gbytes(t, off=0, n=None):
        gf.seek(t.data_offset + off)
        return gf.read(t.n_bytes - off if n is None else n)

    # METADATA: u64 count, then the GGUF's key/value entries byte for byte (GGUF: magic, version, u64 tensors, u64 kvs)
    kv_end = min(t.field.offset for t in g.tensors)
    gf.seek(16)
    if m.metadata != gf.read(kv_end - 16):
        bad.append('METADATA differs from the GGUF key/value block')

    for name, ty, dims, off, nb, _o in m.tensors:
        s = src.get(name)
        if s is None:
            bad.append(f'{name}: not in the GGUF')
            continue
        used.add(name)
        if int(s.tensor_type) != ty:
            bad.append(f'{name}: type {ty}, GGUF type {int(s.tensor_type)}')
            continue
        sd = [int(x) for x in s.shape]
        rows = int(np.prod(sd[1:])) if len(sd) > 1 else 1
        if not same_widened(m.at(m.res_off + off, nb), gbytes(s), rows, row_bytes(ty, sd[0]), row_bytes(ty, dims[0])):
            bad.append(f'{name}: bytes differ from the GGUF')

    n_packs = 0
    for layer, ne, first, pack_bytes, pieces in m.experts:
        whole = {}
        for name, *_ in pieces:
            t = src.get(name)
            if t is None:
                bad.append(f'layer {layer}: {name} not in the GGUF')
                break
            used.add(name)
            if name not in whole:
                whole[name] = gbytes(t)
        else:
            fused = pieces[0][0] == pieces[1][0]
            for e in range(ne):
                pack = bytearray(m.at(m.exp_off + first + e * pack_bytes, pack_bytes))
                for k, (name, ty, rows, cols, at, nb, _o, _stride) in enumerate(pieces):
                    t = src[name]
                    if int(t.tensor_type) != ty:
                        bad.append(f'layer {layer} expert {e}: {name} type {ty}, GGUF {int(t.tensor_type)}')
                        continue
                    sd = [int(x) for x in t.shape]
                    per = t.n_bytes // sd[2]
                    nh, half = (2, k) if fused and k < 2 else (1, 0)
                    piece = whole[name][e * per + half * (per // nh):e * per + (half + 1) * (per // nh)]
                    if not same_widened(bytes(pack[at:at + nb]), piece, sd[1] // nh, row_bytes(ty, sd[0]),
                                        row_bytes(ty, cols)):
                        bad.append(f'layer {layer} expert {e}: {name} bytes differ from the GGUF')
                    pack[at:at + nb] = bytes(nb)
                if np.frombuffer(pack, np.uint8).any():
                    bad.append(f'layer {layer} expert {e}: non-zero bytes outside its pieces')
                n_packs += 1
                if len(bad) > 20:
                    break
        if len(bad) > 20:
            break

    for name in sorted(set(src) - used):
        bad.append(f'{name}: GGUF tensor not in the ToRc file')
    for b in bad[:30]:
        print('FAIL', b)
    print(f'{len(m.tensors)} resident tensors, {n_packs} expert packs against {len(src)} GGUF tensors: '
          f'{"OK" if not bad else f"{len(bad)} failures"} ({time.time() - t0:.0f} s)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
