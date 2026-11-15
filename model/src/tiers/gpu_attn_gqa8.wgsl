// gpu.rs's attention over a prompt block: query heads in groups of 8 per KV head, head dim 256, keys and values int8
// per row with a scale per (KV head, position), as KvCache holds them. A workgroup takes 8 tokens of the block for one
// KV head (64 query rows sharing the same keys) and walks the keys 64 at a time with an online softmax. Thread
// (ty, tx) holds rows 4ty..4ty+3: their scores for keys 4tx..4tx+3, and their outputs for columns 4tx + 64c .. +3
// (c = 0..3), all in vec4 registers. Causal: token j of the block sees positions 0 ..= pos0 + j.
//
// Split over keys (ns > 1): workgroup z = s takes key tiles s*tps .. (s+1)*tps only and writes its unnormalised
// outputs to o's part s, with each row's running max and sum to ml; `combine` then merges the parts into fo. A small
// block has few workgroups, so splitting its keys keeps more of the GPU busy.

struct Params { n: u32, pos0: u32, cap: u32, scale: f32, nh: u32, ns: u32, tps: u32, _p2: u32 }

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;       // [n][nh][64]
@group(0) @binding(2) var<storage, read> k: array<u32>;             // [n_head_kv][cap][64], 4 int8 a word
@group(0) @binding(3) var<storage, read> v: array<u32>;             // [n_head_kv][cap][64]
@group(0) @binding(4) var<storage, read> ks: array<f32>;            // [n_head_kv][cap]
@group(0) @binding(5) var<storage, read> vs: array<f32>;            // [n_head_kv][cap]
@group(0) @binding(6) var<storage, read_write> o: array<vec4<f32>>; // [ns][n][nh][64]
@group(0) @binding(7) var<storage, read_write> ml: array<vec2<f32>>; // [ns][n][nh]: max, sum (ns > 1)
@group(0) @binding(8) var<storage, read_write> fo: array<vec4<f32>>; // [n][nh][64], combine's output

const G: u32 = 8u;
const TQ: u32 = 8u;
const TK: u32 = 64u;
const DC: u32 = 16u;   // dims per score chunk
const VC: u32 = 8u;    // keys per value chunk

// the score chunk: Q transposed [16 dims][16 row groups] then K transposed [16 dims][16 key groups]; or the value
// chunk [8 keys][64 column groups]
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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let g = wg.y;
    let j0 = wg.x * TQ;
    let ty = li / 16u;
    let tx = li % 16u;
    var a00 = vec4<f32>(0.0); var a01 = vec4<f32>(0.0); var a02 = vec4<f32>(0.0); var a03 = vec4<f32>(0.0);
    var a10 = vec4<f32>(0.0); var a11 = vec4<f32>(0.0); var a12 = vec4<f32>(0.0); var a13 = vec4<f32>(0.0);
    var a20 = vec4<f32>(0.0); var a21 = vec4<f32>(0.0); var a22 = vec4<f32>(0.0); var a23 = vec4<f32>(0.0);
    var a30 = vec4<f32>(0.0); var a31 = vec4<f32>(0.0); var a32 = vec4<f32>(0.0); var a33 = vec4<f32>(0.0);
    if (li < 64u) { rm[li] = -1e30; rl[li] = 0.0; }
    let last = p.pos0 + min(j0 + TQ, p.n);
    let ntile = (last + TK - 1u) / TK;
    let t_lo = min(wg.z * p.tps, ntile);
    let t_hi = min(t_lo + p.tps, ntile);
    let kb = g * p.cap;
    // this thread's slot in the Q chunk load: row lr, dims 4ld..4ld+3
    let lr = li / 4u;
    let ld = li % 4u;
    let lj = j0 + lr / G;
    let qrow = (lj * p.nh + g * G + lr % G) * 64u;
    for (var kt = t_lo; kt < t_hi; kt++) {
        let k0 = kt * TK;
        var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < 256u; d0 += DC) {
            workgroupBarrier();
            var qv = vec4<f32>(0.0);
            if (lj < p.n) { qv = q[qrow + d0 / 4u + ld]; }
            sa[(ld * 4u) * 16u + lr / 4u][lr % 4u] = qv.x;
            sa[(ld * 4u + 1u) * 16u + lr / 4u][lr % 4u] = qv.y;
            sa[(ld * 4u + 2u) * 16u + lr / 4u][lr % 4u] = qv.z;
            sa[(ld * 4u + 3u) * 16u + lr / 4u][lr % 4u] = qv.w;
            // K: key lr (0..63), dims 4ld..4ld+3 of the chunk
            let t = k0 + lr;
            var kv = vec4<f32>(0.0);
            if (t < last) { kv = i8x4(k[(kb + t) * 64u + d0 / 4u + ld]); }
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
                if (t >= last || t > p.pos0 + j0 + (ty * 4u + i) / G) { r[i] = -1e30; }
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
        let r0 = ra[ty * 4u]; let r1 = ra[ty * 4u + 1u]; let r2 = ra[ty * 4u + 2u]; let r3 = ra[ty * 4u + 3u];
        a00 *= r0; a01 *= r0; a02 *= r0; a03 *= r0;
        a10 *= r1; a11 *= r1; a12 *= r1; a13 *= r1;
        a20 *= r2; a21 *= r2; a22 *= r2; a23 *= r2;
        a30 *= r3; a31 *= r3; a32 *= r3; a33 *= r3;
        for (var kc = 0u; kc < TK; kc += VC) {
            workgroupBarrier();
            for (var e = 0u; e < 2u; e++) {
                let i = li + 256u * e;
                let kk = i / 64u;
                let w = i % 64u;
                let t = k0 + kc + kk;
                var u = vec4<f32>(0.0);
                if (t < last) { u = i8x4(v[(kb + t) * 64u + w]); }
                sa[i] = u;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < VC; kk++) {
                let pv = pt[(kc + kk) * 16u + ty];
                let v0 = sa[kk * 64u + tx];
                let v1 = sa[kk * 64u + tx + 16u];
                let v2 = sa[kk * 64u + tx + 32u];
                let v3 = sa[kk * 64u + tx + 48u];
                a00 += pv.x * v0; a01 += pv.x * v1; a02 += pv.x * v2; a03 += pv.x * v3;
                a10 += pv.y * v0; a11 += pv.y * v1; a12 += pv.y * v2; a13 += pv.y * v3;
                a20 += pv.z * v0; a21 += pv.z * v1; a22 += pv.z * v2; a23 += pv.z * v3;
                a30 += pv.w * v0; a31 += pv.w * v1; a32 += pv.w * v2; a33 += pv.w * v3;
            }
        }
    }
    workgroupBarrier();
    // rows 4ty + i: token j0 + (4ty + i) / 8 = j0 + ty / 2, head g 8 + (4ty + i) % 8
    let j = j0 + ty / 2u;
    if (j < p.n) {
        let row = j * p.nh + g * G + (ty % 2u) * 4u;
        if (p.ns == 1u) {
            let hb = row * 64u + tx;
            let i0 = 1.0 / rl[ty * 4u]; let i1 = 1.0 / rl[ty * 4u + 1u];
            let i2 = 1.0 / rl[ty * 4u + 2u]; let i3 = 1.0 / rl[ty * 4u + 3u];
            o[hb] = a00 * i0; o[hb + 16u] = a01 * i0; o[hb + 32u] = a02 * i0; o[hb + 48u] = a03 * i0;
            o[hb + 64u] = a10 * i1; o[hb + 80u] = a11 * i1; o[hb + 96u] = a12 * i1; o[hb + 112u] = a13 * i1;
            o[hb + 128u] = a20 * i2; o[hb + 144u] = a21 * i2; o[hb + 160u] = a22 * i2; o[hb + 176u] = a23 * i2;
            o[hb + 192u] = a30 * i3; o[hb + 208u] = a31 * i3; o[hb + 224u] = a32 * i3; o[hb + 240u] = a33 * i3;
        } else {
            let rows = p.n * p.nh;
            let hb = (wg.z * rows + row) * 64u + tx;
            o[hb] = a00; o[hb + 16u] = a01; o[hb + 32u] = a02; o[hb + 48u] = a03;
            o[hb + 64u] = a10; o[hb + 80u] = a11; o[hb + 96u] = a12; o[hb + 112u] = a13;
            o[hb + 128u] = a20; o[hb + 144u] = a21; o[hb + 160u] = a22; o[hb + 176u] = a23;
            o[hb + 192u] = a30; o[hb + 208u] = a31; o[hb + 224u] = a32; o[hb + 240u] = a33;
            if (tx < 4u) { ml[wg.z * rows + row + tx] = vec2<f32>(rm[ty * 4u + tx], rl[ty * 4u + tx]); }
        }
    }
}

// The split parts merged: one thread per 4 output columns, row = i / 64. Each part's outputs and sum are rescaled
// from its own running max to the largest, then summed and divided.
@compute @workgroup_size(256)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = p.n * p.nh;
    let i = gid.x;
    if (i >= rows * 64u) { return; }
    let row = i / 64u;
    var mx = -1e30;
    for (var s = 0u; s < p.ns; s++) { mx = max(mx, ml[s * rows + row].x); }
    var acc = vec4<f32>(0.0);
    var l = 0.0;
    for (var s = 0u; s < p.ns; s++) {
        let w = ml[s * rows + row];
        if (w.y > 0.0) {
            let f = exp(w.x - mx);
            acc += o[s * rows * 64u + i] * f;
            l += w.y * f;
        }
    }
    fo[i] = acc / l;
}
