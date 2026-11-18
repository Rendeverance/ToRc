#!/usr/bin/env python3
"""Write the ggml lookup tables ToRc ports into kernels/src/quant as Rust statics.

The tables come from ggml-common.h (MIT) and are copied unchanged: a translation would change the weights.
Run with the path of a ggml checkout's ggml-common.h; it rewrites the files it knows about, so a new upstream
release can be diffed instead of retyped.
"""
import re
import sys

OUT = 'kernels/src/quant'
NAMED = {'NGRID_IQ1S': 2048}

TABLES = [
    # file, rust type, rust name, ggml name, values a line, doc line
    ('iq2xxs_grid.rs', 'u64', 'IQ2XXS_GRID', 'iq2xxs_grid', 8,
     "256 entries of eight 8-bit magnitudes each, lowest byte first."),
    ('iq2xs_grid.rs', 'u64', 'IQ2XS_GRID', 'iq2xs_grid', 4,
     "512 entries of eight 8-bit magnitudes each, lowest byte first."),
    ('iq2s_grid.rs', 'u64', 'IQ2S_GRID', 'iq2s_grid', 3,
     "1024 entries of eight 8-bit magnitudes each, lowest byte first."),
    ('iq3xxs_grid.rs', 'u32', 'IQ3XXS_GRID', 'iq3xxs_grid', 8,
     "256 entries of four 8-bit magnitudes each, lowest byte first."),
    ('iq1s_grid.rs', 'u64', 'IQ1S_GRID', 'iq1s_grid', 4,
     "2048 entries of eight signed 8-bit values each, lowest byte first."),
]


def table(src, name):
    m = re.search(r'GGML_TABLE_BEGIN\((\w+),\s*' + name + r',\s*(\w+)\)(.*?)GGML_TABLE_END', src, re.S)
    if not m:
        raise SystemExit('no table named ' + name)
    ctype, count, body = m.group(1), m.group(2), m.group(3)
    vals = re.findall(r'0x[0-9a-fA-F]+|\b\d+\b', body)
    want = int(count) if count.isdigit() else NAMED[count]
    if len(vals) != want:
        raise SystemExit(f'{name}: read {len(vals)} values, expected {want}')
    return ctype, vals


def main(header):
    src = open(header).read()
    for fn, rtype, rname, gname, per, doc in TABLES:
        ctype, vals = table(src, gname)
        if (ctype == 'uint64_t') != (rtype == 'u64') or (ctype == 'uint32_t') != (rtype == 'u32'):
            raise SystemExit(f'{gname}: ggml says {ctype}, this script says {rtype}')
        out = [
            '// SPDX-License-Identifier: MIT',
            f'//! {gname}.rs - ggml\'s {gname}, unchanged (ggml/src/ggml-common.h; MIT, Copyright (c) 2023-2026 The',
            f'//! ggml authors). {doc}',
            '',
            '#[rustfmt::skip]',
            f'pub static {rname}: [{rtype}; {len(vals)}] = [',
        ]
        for i in range(0, len(vals), per):
            out.append('    ' + ', '.join(vals[i:i + per]) + ',')
        out.append('];')
        open(f'{OUT}/{fn}', 'w').write('\n'.join(out) + '\n')
        print(f'{OUT}/{fn}: {len(vals)} {rtype} values')

    _, km = table(src, 'kmask_iq2xs')
    _, ks = table(src, 'ksigns_iq2xs')
    out = [
        '// SPDX-License-Identifier: MIT',
        "//! ksigns.rs - ggml's sign tables, unchanged (ggml/src/ggml-common.h; MIT, Copyright (c) 2023-2026 The ggml",
        '//! authors): the eight bit positions of a sub-block, and the byte of signs each of the 128 seven-bit codes',
        '//! IQ2_XXS, IQ2_XS and IQ3_XXS stores picks.',
        '',
        '#[rustfmt::skip]',
        'pub static KMASK_IQ2XS: [u8; 8] = [',
        '    ' + ', '.join(km) + ',',
        '];',
        '',
        '#[rustfmt::skip]',
        'pub static KSIGNS_IQ2XS: [u8; 128] = [',
    ]
    for i in range(0, len(ks), 16):
        out.append('    ' + ', '.join(ks[i:i + 16]) + ',')
    out.append('];')
    open(f'{OUT}/ksigns.rs', 'w').write('\n'.join(out) + '\n')
    print(f'{OUT}/ksigns.rs: {len(km)} + {len(ks)} u8 values')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ggml/src/ggml-common.h')
