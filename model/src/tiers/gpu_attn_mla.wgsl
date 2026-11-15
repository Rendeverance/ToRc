// gpu.rs's attention over a prompt block, for the MLA shape (one KV head, head dim 576): gpu.rs puts these constants in front
// of this file, one compiled kernel per shape:
//   HD  head dim (a multiple of 64 to 576)      G   query heads per KV head (1 to 64)
//   K16 keys and values at 16 bits (i16 with a scale per row, ERNIE, LFM2) instead of 8 (int8, the Qwen 35B)
//   KW  u32 words per cache row (HD / 4, or HD / 2 at 16 bits)
//   VC  keys per value chunk: the largest power of two whose VC x HV vec4s fit in sa (2 for MLA's 576)
// HD's largest value, 576, is MLA's absorbed head (GLM-4.7-Flash, Moonlight). Lines starting /*SINK*/ are kept only
// for a model with a sink per query head (gpt-oss), else commented out.
// Keys and values are per (KV head, position) rows with a scale each, as KvCache holds them.
//
// A workgroup takes TQ = 64 / G tokens of the block for one KV head (ROWS = TQ * G query rows sharing the same keys;
// the rest of the 64 row slots idle when G does not divide 64) and walks the keys 64 at a time with an online softmax.
// Thread (ty, tx) holds rows 4ty..4ty+3: their scores for keys 4tx..4tx+3, and their outputs for the column groups
// tx + 16c (c < NC = HD / 64), each four columns wide, in vec4 registers. Causal: token j of the block sees positions
// 0 ..= pos0 + j, and with a sliding window (p.window > 0, Gemma, gpt-oss) only the last p.window of them.
//
// Split over keys (ns > 1): workgroup z = s takes key tiles s*tps .. (s+1)*tps only and writes its unnormalised
// outputs to o's part s, with each row's running max and sum to ml; `combine` then merges the parts into fo. A small
// block has few workgroups, so splitting its keys keeps more of the GPU busy.

struct Params { n: u32, pos0: u32, cap: u32, scale: f32, nh: u32, ns: u32, tps: u32, window: u32, stride: u32,
    kp: u32, _a: u32, _b: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;       // [n][nh][HV]
@group(0) @binding(2) var<storage, read> k: array<u32>;             // [n_head_kv][cap][KW]
@group(0) @binding(3) var<storage, read> v: array<u32>;             // [n_head_kv][cap][KW]
@group(0) @binding(4) var<storage, read> ks: array<f32>;            // [n_head_kv][cap]
@group(0) @binding(5) var<storage, read> vs: array<f32>;            // [n_head_kv][cap]
@group(0) @binding(6) var<storage, read_write> o: array<vec4<f32>>; // [ns][n][nh][HV]
@group(0) @binding(7) var<storage, read_write> ml: array<vec2<f32>>; // [ns][n][nh]: max, sum (ns > 1)
@group(0) @binding(8) var<storage, read_write> fo: array<vec4<f32>>; // [n][nh][HV], combine's output
/*SINK*/@group(0) @binding(9) var<storage, read> sk: array<f32>;    // [nh]: each query head's sink logit
// the indexer's picks (qwen4exp, Flash Next; lines starting /*QSA*/): token j's flag per pool of p.kp positions at
// byte j * p.stride + pool, and nv[j] the pools the flags cover (positions past them all seen; 0xffffffff: no picks,
// every position seen)
/*QSA*/@group(0) @binding(10) var<storage, read> sel: array<u32>;
/*QSA*/@group(0) @binding(11) var<storage, read> nv: array<u32>;

const TQ: u32 = 64u / G;
const ROWS: u32 = TQ * G;
const HV: u32 = HD / 4u;        // vec4s per row
const NC: u32 = HD / 64u;       // column groups per thread
const TK: u32 = 64u;
const DC: u32 = 16u;            // dims per score chunk

// the score chunk: Q transposed [16 dims][16 row groups] then K transposed [16 dims][16 key groups]; or the value
// chunk [VC keys][HV column groups]
var<workgroup> sa: array<vec4<f32>, 512>;
// the scores, then the weights, transposed: [64 keys][16 row groups]
var<workgroup> pt: array<vec4<f32>, 1024>;
var<workgroup> rm: array<f32, 64>;
var<workgroup> rl: array<f32, 64>;
var<workgroup> ra: array<f32, 64>;

fn i8x4(w: u32) -> vec4<f32> {
    let x = bitcast<i32>(w);
    return vec4<f32>(f32(extractBits(x, 0u, 8u)), f32(extractBits(x, 8u, 8u)), f32(extractBits(x, 16u, 8u)),
        f32(extractBits(x, 24u, 8u)));
}

fn i16x2(w: u32) -> vec2<f32> {
    let x = bitcast<i32>(w);
    return vec2<f32>(f32(extractBits(x, 0u, 16u)), f32(extractBits(x, 16u, 16u)));
}

// cache row r's numbers 4c .. 4c+3, unscaled
fn kq(r: u32, c: u32) -> vec4<f32> {
    if (K16) {
        let b = r * KW + 2u * c;
        return vec4<f32>(i16x2(k[b]), i16x2(k[b + 1u]));
    }
    return i8x4(k[r * KW + c]);
}

fn vq(r: u32, c: u32) -> vec4<f32> {
    if (K16) {
        let b = r * KW + 2u * c;
        return vec4<f32>(i16x2(v[b]), i16x2(v[b + 1u]));
    }
    return i8x4(v[r * KW + c]);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let g = wg.y;
    let j0 = wg.x * TQ;
    let ty = li / 16u;
    let tx = li % 16u;
    var acc: array<vec4<f32>, 4u * NC>;   // [row 4ty + i][column group c] at i * NC + c
    for (var x = 0u; x < 4u * NC; x++) { acc[x] = vec4<f32>(0.0); }
    if (li < 64u) { rm[li] = -1e30; rl[li] = 0.0; }
    let last = p.pos0 + min(j0 + TQ, p.n);
    let ntile = (last + TK - 1u) / TK;
    let t_lo = min(wg.z * p.tps, ntile);
    let t_hi = min(t_lo + p.tps, ntile);
    let kb = g * p.cap;
    // with a window, the earliest key any row of this workgroup sees (tiles wholly before it are skipped)
    var lowvis = 0u;
    if (p.window > 0u && p.pos0 + j0 + 1u > p.window) { lowvis = p.pos0 + j0 + 1u - p.window; }
    // this thread's slot in the Q chunk load: row lr, dims 4ld..4ld+3
    let lr = li / 4u;
    let ld = li % 4u;
    let lj = j0 + lr / G;
    let lok = lr < ROWS && lj < p.n;
    let qrow = (lj * p.nh + g * G + lr % G) * HV;
    for (var kt = t_lo; kt < t_hi; kt++) {
        let k0 = kt * TK;
        if (k0 + TK <= lowvis) { continue; }
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < HD; d0 += DC) {
            workgroupBarrier();
            var qv = vec4<f32>(0.0);
            if (lok) { qv = q[qrow + d0 / 4u + ld]; }
            sa[(ld * 4u) * 16u + lr / 4u][lr % 4u] = qv.x;
            sa[(ld * 4u + 1u) * 16u + lr / 4u][lr % 4u] = qv.y;
            sa[(ld * 4u + 2u) * 16u + lr / 4u][lr % 4u] = qv.z;
            sa[(ld * 4u + 3u) * 16u + lr / 4u][lr % 4u] = qv.w;
            // K: key lr (0..63), dims 4ld..4ld+3 of the chunk
            let t = k0 + lr;
            var kv = vec4<f32>(0.0);
            if (t < last) { kv = kq(kb + t, d0 / 4u + ld); }
            sa[256u + (ld * 4u) * 16u + lr / 4u][lr % 4u] = kv.x;
            sa[256u + (ld * 4u + 1u) * 16u + lr / 4u][lr % 4u] = kv.y;
            sa[256u + (ld * 4u + 2u) * 16u + lr / 4u][lr % 4u] = kv.z;
            sa[256u + (ld * 4u + 3u) * 16u + lr / 4u][lr % 4u] = kv.w;
            workgroupBarrier();
            for (var d = 0u; d < DC; d++) {
                let qq = sa[d * 16u + ty];
                let kk = sa[256u + d * 16u + tx];
                s0 += qq.x * kk;
                s1 += qq.y * kk;
                s2 += qq.z * kk;
                s3 += qq.w * kk;
            }
        }
        // scale and mask into pt[key][row group]
        for (var c = 0u; c < 4u; c++) {
            let t = k0 + tx * 4u + c;
            var sc = 0.0;
            if (t < last) { sc = p.scale * ks[kb + t]; }
            var r = vec4<f32>(s0[c], s1[c], s2[c], s3[c]) * sc;
            for (var i = 0u; i < 4u; i++) {
                let row = ty * 4u + i;
                let pos = p.pos0 + j0 + row / G;
                if (row >= ROWS || t >= last || t > pos || (p.window > 0u && t + p.window <= pos)) { r[i] = -1e30; }
/*QSA*/         else {
/*QSA*/             let jj = j0 + row / G;
/*QSA*/             let nj = nv[jj];
/*QSA*/             if (nj != 0xffffffffu && t < nj * p.kp) {
/*QSA*/                 let b = jj * p.stride + t / p.kp;
/*QSA*/                 if (((sel[b / 4u] >> ((b % 4u) * 8u)) & 0xffu) == 0u) { r[i] = -1e30; }
/*QSA*/             }
/*QSA*/         }
            }
            pt[(tx * 4u + c) * 16u + ty] = r;
        }
        workgroupBarrier();
        if (li < 64u) {
            let rg = li / 4u;
            let rc = li % 4u;
            let old = rm[li];
            var mx = old;
            for (var c = 0u; c < TK; c++) { mx = max(mx, pt[c * 16u + rg][rc]); }
            let a = exp(old - mx);
            var l = rl[li] * a;
            for (var c = 0u; c < TK; c++) {
                let t = k0 + c;
                // a key masked for this row adds nothing, also while every key so far was masked (a split can
                // start past a row's position)
                let x = pt[c * 16u + rg][rc];
                var e = 0.0;
                if (x > -1e29) { e = exp(x - mx); }
                l += e;
                var sc = 0.0;
                if (t < last) { sc = vs[kb + t]; }
                pt[c * 16u + rg][rc] = e * sc;
            }
            rm[li] = mx;
            rl[li] = l;
            ra[li] = a;
        }
        workgroupBarrier();
        for (var i = 0u; i < 4u; i++) {
            let r = ra[ty * 4u + i];
            for (var c = 0u; c < NC; c++) { acc[i * NC + c] *= r; }
        }
        for (var kc = 0u; kc < TK; kc += VC) {
            workgroupBarrier();
            for (var e = 0u; e < 2u; e++) {
                let i = li + 256u * e;
                if (i < VC * HV) {
                    let kk = i / HV;
                    let w = i % HV;
                    let t = k0 + kc + kk;
                    var u = vec4<f32>(0.0);
                    if (t < last) { u = vq(kb + t, w); }
                    sa[i] = u;
                }
            }
            workgroupBarrier();
            for (var kk = 0u; kk < VC; kk++) {
                let pv = pt[(kc + kk) * 16u + ty];
                for (var c = 0u; c < NC; c++) {
                    let vv = sa[kk * HV + tx + 16u * c];
                    acc[c] += pv.x * vv;
                    acc[NC + c] += pv.y * vv;
                    acc[2u * NC + c] += pv.z * vv;
                    acc[3u * NC + c] += pv.w * vv;
                }
            }
        }
    }
    workgroupBarrier();
    // row 4ty + i: token j0 + row / G, query head g G + row % G
    for (var i = 0u; i < 4u; i++) {
        let r = ty * 4u + i;
        let j = j0 + r / G;
        if (r < ROWS && j < p.n) {
            let h = g * G + r % G;
            let row = j * p.nh + h;
            if (p.ns == 1u) {
                var l = rl[r];
/*SINK*/        l += exp(sk[h] - rm[r]);
                let inv = 1.0 / l;
                for (var c = 0u; c < NC; c++) { o[row * HV + tx + 16u * c] = acc[i * NC + c] * inv; }
            } else {
                let rows = p.n * p.nh;
                for (var c = 0u; c < NC; c++) { o[(wg.z * rows + row) * HV + tx + 16u * c] = acc[i * NC + c]; }
                if (tx == 0u) { ml[wg.z * rows + row] = vec2<f32>(rm[r], rl[r]); }
            }
        }
    }
}

// The split parts merged: one thread per 4 output columns, row = i / HV. Each part's outputs and sum are rescaled
// from its own running max to the largest, then summed and divided (with the head's sink in the sum, gpt-oss).
@compute @workgroup_size(256)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = p.n * p.nh;
    let i = gid.x;
    if (i >= rows * HV) { return; }
    let row = i / HV;
    var mx = -1e30;
    for (var s = 0u; s < p.ns; s++) { mx = max(mx, ml[s * rows + row].x); }
    var acc = vec4<f32>(0.0);
    var l = 0.0;
    for (var s = 0u; s < p.ns; s++) {
        let w = ml[s * rows + row];
        if (w.y > 0.0) {
            let f = exp(w.x - mx);
            acc += o[s * rows * HV + i] * f;
            l += w.y * f;
        }
    }
/*SINK*/    l += exp(sk[row % p.nh] - mx);
    fo[i] = acc / l;
}
