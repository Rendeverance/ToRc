/* ggml_ref.c - what ggml itself makes of the blocks the parity test generates.
 *
 * kernels/tests/parity.rs writes weight blocks and Q8_K vectors, this program decodes and multiplies them with the
 * format's own definitions, and the test compares the two answers. That is the check the format tables alone cannot
 * give: it does not assume this crate read ggml's formats correctly, it asks ggml.
 *
 * Built by scripts/ggml_ref.sh from a ggml checkout: the two translation units it links carry the formats
 * (ggml-quants.c) and their scalar dots (ggml-cpu/quants.c), which are the definitions every fast path copies.
 *
 * Input ("TORCREF1"): nrec records of (type, weight bytes for the whole row, blocks), the packed weights, then per
 * block the scale,
 * the 256 values and the 16 sums of 16 that go with them. Output ("TORCOUT1"): per record the dequantised floats
 * and the product.
 *
 * Q2_0's dot takes ggml's q8_0 vectors, not q8_K, and their scale is fp16 there where this crate keeps it in f32.
 * So for that type the 256 values above are split into 32-wide groups under a scale of one on both sides: the
 * weights, the unpacking and the integer dots are compared exactly, the rounding of an activation scale is not. */
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the C view of the shared header, as ggml-quants.h sets it before including it */
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

/* ggml.c owns this table and the dots for other types reach for it; the six here do not, but the link wants it, so
   here it is rather than the rest of ggml.c. */
float ggml_table_f32_f16[1 << 16];

static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
    if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {
            e = 113;
            while ((m & 0x400) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else {
        bits = sign | ((e == 0x1F ? 0xFF : e + 112) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* The rest of ggml.c is not linked here. These are the few things its other formats and its error paths reach for:
   none of the six types above touches them, so they answer plainly or stop loudly instead of quietly guessing. */
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ggml_ref: %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}

size_t ggml_type_size(int type) {
    switch (type) {
        case 10:
            return sizeof(block_q2_K);
        case 16:
            return sizeof(block_iq2_xxs);
        case 17:
            return sizeof(block_iq2_xs);
        case 18:
            return sizeof(block_iq3_xxs);
        case 22:
            return sizeof(block_iq2_s);
        case 29:
            return sizeof(block_iq1_m);
        case 42:
            return sizeof(block_q2_0);
        default:
            return 0;
    }
}

const char * ggml_type_name(int type) {
    switch (type) {
        case 10:
            return "q2_K";
        case 16:
            return "iq2_xxs";
        case 17:
            return "iq2_xs";
        case 18:
            return "iq3_xxs";
        case 22:
            return "iq2_s";
        case 29:
            return "iq1_m";
        case 42:
            return "q2_0";
        default:
            return "not linked into this harness";
    }
}

size_t ggml_row_size(int type, int64_t ne) {
    const size_t bs = ggml_type_size(type);
    return bs ? (size_t)ne / QK_K * bs : 0;
}

float ggml_table_f32_f8_e4m3[256];

static void fill_e4m3_table(void) {
    for (int i = 0; i < 256; ++i) {
        const int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 7;
        float v;
        if (e == 0) {
            v = (float)m * (1.0f / 512.0f);
        } else if (e == 0xF && m == 0x7) {
            v = NAN;
        } else {
            v = (float)(8 + m) * (float)ldexp(1.0, e - 10);
        }
        ggml_table_f32_f8_e4m3[i] = s ? -v : v;
    }
}

void dequantize_row_q2_K(const block_q2_K *x, float *y, int64_t k);
void dequantize_row_iq2_xxs(const block_iq2_xxs *x, float *y, int64_t k);
void dequantize_row_iq2_xs(const block_iq2_xs *x, float *y, int64_t k);
void dequantize_row_iq2_s(const block_iq2_s *x, float *y, int64_t k);
void dequantize_row_iq3_xxs(const block_iq3_xxs *x, float *y, int64_t k);
void dequantize_row_iq1_m(const block_iq1_m *x, float *y, int64_t k);

void ggml_vec_dot_q2_K_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by,
                                    int nrc);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
                                       size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by,
                                      int nrc);
void ggml_vec_dot_iq2_s_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by,
                                     int nrc);
void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy,
                                       size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by,
                                     int nrc);
void dequantize_row_q2_0(const block_q2_0 *x, float *y, int64_t k);
void ggml_vec_dot_q2_0_q8_0_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by,
                                    int nrc);

static uint32_t rd(FILE *f, void *p, size_t n) {
    if (fread(p, 1, n, f) != n) {
        fprintf(stderr, "ggml_ref: short input\n");
        exit(1);
    }
    return 0;
}

static uint32_t rd_u32(FILE *f) {
    uint32_t v;
    rd(f, &v, sizeof(v));
    return v;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.bin out.bin\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
    if (!in || !out) {
        fprintf(stderr, "ggml_ref: cannot open the files\n");
        return 2;
    }
    char magic[8];
    rd(in, magic, 8);
    if (memcmp(magic, "TORCREF1", 8) != 0) {
        fprintf(stderr, "ggml_ref: not a reference request\n");
        return 2;
    }
    const uint32_t nrec = rd_u32(in);
    for (int i = 0; i < (1 << 16); ++i) {
        ggml_table_f32_f16[i] = half_to_float((uint16_t)i);
    }
    fill_e4m3_table();
    fwrite("TORCOUT1", 1, 8, out);
    fwrite(&nrec, sizeof(nrec), 1, out);

    for (uint32_t r = 0; r < nrec; ++r) {
        const uint32_t ty = rd_u32(in), bytes = rd_u32(in), nb = rd_u32(in);
        uint8_t *w = malloc(bytes);
        block_q8_K *q = malloc(sizeof(*q) * nb);
        rd(in, w, bytes);
        for (uint32_t i = 0; i < nb; ++i) {
            rd(in, &q[i].d, sizeof(q[i].d));
            rd(in, q[i].qs, QK_K);
            rd(in, q[i].bsums, sizeof(q[i].bsums));
        }
        const int n = (int)nb * QK_K;
        float *dq = malloc(sizeof(float) * n);
        float dot = 0.0f;
        switch (ty) {
            case 10:
                dequantize_row_q2_K((const block_q2_K *)w, dq, n);
                ggml_vec_dot_q2_K_q8_K_generic(n, &dot, 1, w, 1, q, 1, 1);
                break;
            case 16:
                dequantize_row_iq2_xxs((const block_iq2_xxs *)w, dq, n);
                ggml_vec_dot_iq2_xxs_q8_K_generic(n, &dot, 1, w, 1, q, 1, 1);
                break;
            case 17:
                dequantize_row_iq2_xs((const block_iq2_xs *)w, dq, n);
                ggml_vec_dot_iq2_xs_q8_K_generic(n, &dot, 1, w, 1, q, 1, 1);
                break;
            case 18:
                dequantize_row_iq3_xxs((const block_iq3_xxs *)w, dq, n);
                ggml_vec_dot_iq3_xxs_q8_K_generic(n, &dot, 1, w, 1, q, 1, 1);
                break;
            case 22:
                dequantize_row_iq2_s((const block_iq2_s *)w, dq, n);
                ggml_vec_dot_iq2_s_q8_K_generic(n, &dot, 1, w, 1, q, 1, 1);
                break;
            case 29:
                dequantize_row_iq1_m((const block_iq1_m *)w, dq, n);
                ggml_vec_dot_iq1_m_q8_K_generic(n, &dot, 1, w, 1, q, 1, 1);
                break;
            case 42: {
                /* the q8_K vector of the record, re-cut into q8_0 groups under a scale of one (see the header) */
                block_q8_0 * g = calloc((size_t)nb * QK_K / QK8_0, sizeof(*g));
                for (uint32_t i = 0; i < nb * QK_K / QK8_0; ++i) {
                    g[i].d = 0x3C00; /* fp16 one */
                    memcpy(g[i].qs, q[i / (QK_K / QK8_0)].qs + (i % (QK_K / QK8_0)) * QK8_0, QK8_0);
                }
                dequantize_row_q2_0((const block_q2_0 *)w, dq, n);
                ggml_vec_dot_q2_0_q8_0_generic(n, &dot, 1, w, 1, g, 1, 1);
                free(g);
                break;
            }
            default:
                fprintf(stderr, "ggml_ref: no reference for type %u\n", ty);
                return 3;
        }
        fwrite(dq, sizeof(float), (size_t)n, out);
        fwrite(&dot, sizeof(dot), 1, out);
        free(w);
        free(q);
        free(dq);
    }
    fclose(in);
    fclose(out);
    return 0;
}
