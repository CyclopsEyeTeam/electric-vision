//! prod_gate_native_simd.rs
//!
//! PROD-GATE-2: Memory-safe Rust (#![forbid(unsafe_code)]) implementation of the
//! 10 closed-form measurement operators using strictly integer and Q16.16
//! fixed-point arithmetic (zero IEEE-754 floating-point non-determinism).

#![forbid(unsafe_code)]

#[inline]
fn abs_i32(v: i32) -> i32 {
    if v < 0 { -v } else { v }
}

pub fn op_mean_rgba_u8(rgba: &[u8], width: u32, height: u32) -> Option<[u64; 8]> {
    let total_px = (width as usize).checked_mul(height as usize)?;
    if total_px == 0 || rgba.len() < total_px * 4 {
        return None;
    }
    let (mut sr, mut sg, mut sb, mut sa) = (0u64, 0u64, 0u64, 0u64);
    for px in rgba[..total_px * 4].chunks_exact(4) {
        sr += px[0] as u64;
        sg += px[1] as u64;
        sb += px[2] as u64;
        sa += px[3] as u64;
    }
    let t = total_px as u64;
    Some([
        sr,
        sg,
        sb,
        sa,
        (sr << 16) / t,
        (sg << 16) / t,
        (sb << 16) / t,
        (sa << 16) / t,
    ])
}

pub fn op_luma_metrics_u8(rgba: &[u8], width: u32, height: u32) -> Option<([u32; 3], [u32; 16])> {
    let total_px = (width as usize).checked_mul(height as usize)?;
    if total_px == 0 || rgba.len() < total_px * 4 {
        return None;
    }
    let mut min_y = 255u32;
    let mut max_y = 0u32;
    let mut hist = [0u32; 16];
    for px in rgba[..total_px * 4].chunks_exact(4) {
        let y = (77u32 * (px[0] as u32) + 150u32 * (px[1] as u32) + 29u32 * (px[2] as u32)) >> 8;
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
        hist[((y >> 4) & 0xF) as usize] += 1;
    }
    let wcag_q16 = ((max_y + 13) << 16) / (min_y + 13);
    Some(([min_y, max_y, wcag_q16], hist))
}

pub fn op_bbox_and_count(
    rgba: &[u8],
    width: u32,
    height: u32,
    mode: i32,
    target: [u8; 4],
    tolerance: i32,
) -> Option<[i64; 6]> {
    let total_px = (width as usize).checked_mul(height as usize)?;
    if total_px == 0 || rgba.len() < total_px * 4 {
        return None;
    }
    let mut matched = 0i64;
    let mut non_transparent = 0i64;
    let mut x_min = width as i64;
    let mut y_min = height as i64;
    let mut x_max = -1i64;
    let mut y_max = -1i64;

    for y in 0..height {
        let row_off = (y as usize) * (width as usize) * 4;
        for x in 0..width {
            let off = row_off + (x as usize) * 4;
            let (r, g, b, a) = (rgba[off], rgba[off + 1], rgba[off + 2], rgba[off + 3]);
            if a > 0 {
                non_transparent += 1;
            }
            let hit = if mode == 0 {
                a > 0
            } else {
                abs_i32(r as i32 - target[0] as i32) <= tolerance
                    && abs_i32(g as i32 - target[1] as i32) <= tolerance
                    && abs_i32(b as i32 - target[2] as i32) <= tolerance
                    && abs_i32(a as i32 - target[3] as i32) <= tolerance
            };
            if hit {
                matched += 1;
                let (xi, yi) = (x as i64, y as i64);
                if xi < x_min { x_min = xi; }
                if xi > x_max { x_max = xi; }
                if yi < y_min { y_min = yi; }
                if yi > y_max { y_max = yi; }
            }
        }
    }
    let alpha_cov_q16 = (non_transparent << 16) / (total_px as i64);
    if matched == 0 {
        Some([0, -1, -1, -1, -1, alpha_cov_q16])
    } else {
        Some([matched, x_min, y_min, x_max, y_max, alpha_cov_q16])
    }
}
