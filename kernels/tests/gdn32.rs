//! gdn32.rs - the f32 DeltaNet step's native kernel (NEON / AVX2) against the portable definition, bit for bit: in
//! place and from a base state, a dimension with and without an 8-number tail.

fn rnd(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
fn native_matches_reference() {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for &(dim, rows) in &[(128usize, 32usize), (136, 8), (256, 12), (8, 4)] {
        for from_base in [false, true] {
            let state: Vec<f32> = (0..rows * dim).map(|_| rnd(&mut s)).collect();
            let q: Vec<f32> = (0..dim).map(|_| rnd(&mut s) * 0.1).collect();
            let k: Vec<f32> = (0..dim).map(|_| rnd(&mut s) * 0.1).collect();
            let v: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
            let (alpha, beta, q_scale) = (0.93, 0.4, 0.088);
            let (mut m1, mut m2) = (state.clone(), state.clone());
            let base = from_base.then(|| state.iter().map(|x| x * 1.5).collect::<Vec<f32>>());
            let (mut o1, mut o2) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            kernels::delta_rule_f32(base.as_deref(), &mut m1, &q, &k, &v, alpha, beta, q_scale, &mut o1, dim);
            kernels::delta_rule_f32_scalar(base.as_deref(), &mut m2, &q, &k, &v, alpha, beta, q_scale, &mut o2, dim);
            let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same(&o1, &o2), "dim {dim} rows {rows} base {from_base}: outputs differ");
            assert!(same(&m1, &m2), "dim {dim} rows {rows} base {from_base}: states differ");
        }
    }
}

/// The fp16 state's native step against its portable definition (where a native kernel takes the shape).
#[test]
fn fp16_native_matches_reference() {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    for &(dim, rows) in &[(128usize, 32usize), (256, 8), (64, 4)] {
        let state: Vec<u16> = (0..rows * dim).map(|_| kernels::q8k::f32_to_f16(rnd(&mut s))).collect();
        let q: Vec<f32> = (0..dim).map(|_| rnd(&mut s) * 0.1).collect();
        let k: Vec<f32> = (0..dim).map(|_| rnd(&mut s) * 0.1).collect();
        let v: Vec<f32> = (0..rows).map(|_| rnd(&mut s)).collect();
        let (mut m1, mut m2) = (state.clone(), state.clone());
        let (mut o1, mut o2) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        if !kernels::delta_rule_h(&mut m1, &q, &k, &v, 0.93, 0.4, 0.088, &mut o1, dim) {
            continue;
        }
        kernels::delta_rule_h_scalar(&mut m2, &q, &k, &v, 0.93, 0.4, 0.088, &mut o2, dim);
        assert!(o1.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()), "dim {dim}: outputs differ");
        assert_eq!(m1, m2, "dim {dim}: states differ");
    }
}
