/*
 * prod_gate_native_simd.c
 *
 * PROD-GATE-2: Memory-safe, bounds-checked native C11 + AVX2/SIMD Closed-Form
 * Operator Engine for Electric Eye.
 *
 * Strictly uses integer and Q16.16 fixed-point arithmetic (zero IEEE-754 float
 * non-determinism or FMA re-association) for bit-exact replay parity across
 * x86_64 and reference implementations.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline int32_t abs_i32(int32_t v) {
    return v < 0 ? -v : v;
}

/*
 * 1. op_mean_rgba_u8:
 * Computes exact channel sums and integer/fixed-point means over width*height RGBA buffer.
 * out_stats[0..3] = sum_r, sum_g, sum_b, sum_a (uint64_t)
 * out_stats[4..7] = mean_r_q16, mean_g_q16, mean_b_q16, mean_a_q16 (uint64_t, Q16.16)
 */
int ee_op_mean_rgba_u8(
    const uint8_t *rgba,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    uint64_t *out_stats
) {
    if (!rgba || !out_stats || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    uint64_t sr = 0, sg = 0, sb = 0, sa = 0;
    size_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= total_px; i += 8) {
        size_t off = i << 2;
        __m256i v = _mm256_loadu_si256((const __m256i *)(rgba + off));
        if (_mm256_testz_si256(v, v)) {
            continue;
        }
        for (size_t k = 0; k < 8; ++k) {
            size_t poff = off + (k << 2);
            sr += rgba[poff];
            sg += rgba[poff + 1];
            sb += rgba[poff + 2];
            sa += rgba[poff + 3];
        }
    }
#endif
    for (; i < total_px; ++i) {
        size_t off = i << 2;
        sr += rgba[off];
        sg += rgba[off + 1];
        sb += rgba[off + 2];
        sa += rgba[off + 3];
    }
    out_stats[0] = sr;
    out_stats[1] = sg;
    out_stats[2] = sb;
    out_stats[3] = sa;
    out_stats[4] = (sr << 16) / total_px;
    out_stats[5] = (sg << 16) / total_px;
    out_stats[6] = (sb << 16) / total_px;
    out_stats[7] = (sa << 16) / total_px;
    return 0;
}

/* IEC sRGB transfer function, rounded to 1e-6. WCAG luminance uses
 * 0.2126 R + 0.7152 G + 0.0722 B; integer products have scale 1e10.
 * Contrast is the range of stored RGB values, not an alpha-composited UI audit. */
static const uint32_t srgb_linear[256] = {
    0, 304, 607, 911, 1214, 1518, 1821, 2125, 2428, 2732, 3035, 3347, 3677, 4025, 4391, 4777,
    5182, 5605, 6049, 6512, 6995, 7499, 8023, 8568, 9134, 9721, 10330, 10960, 11612, 12286, 12983, 13702,
    14444, 15209, 15996, 16807, 17642, 18500, 19382, 20289, 21219, 22174, 23153, 24158, 25187, 26241, 27321, 28426,
    29557, 30713, 31896, 33105, 34340, 35601, 36889, 38204, 39546, 40915, 42311, 43735, 45186, 46665, 48172, 49707,
    51269, 52861, 54480, 56128, 57805, 59511, 61246, 63010, 64803, 66626, 68478, 70360, 72272, 74214, 76185, 78187,
    80220, 82283, 84376, 86500, 88656, 90842, 93059, 95307, 97587, 99899, 102242, 104616, 107023, 109462, 111932, 114435,
    116971, 119538, 122139, 124772, 127438, 130136, 132868, 135633, 138432, 141263, 144128, 147027, 149960, 152926, 155926, 158961,
    162029, 165132, 168269, 171441, 174647, 177888, 181164, 184475, 187821, 191202, 194618, 198069, 201556, 205079, 208637, 212231,
    215861, 219526, 223228, 226966, 230740, 234551, 238398, 242281, 246201, 250158, 254152, 258183, 262251, 266356, 270498, 274677,
    278894, 283149, 287441, 291771, 296138, 300544, 304987, 309469, 313989, 318547, 323143, 327778, 332452, 337164, 341914, 346704,
    351533, 356400, 361307, 366253, 371238, 376262, 381326, 386429, 391572, 396755, 401978, 407240, 412543, 417885, 423268, 428690,
    434154, 439657, 445201, 450786, 456411, 462077, 467784, 473531, 479320, 485150, 491021, 496933, 502886, 508881, 514918, 520996,
    527115, 533276, 539479, 545724, 552011, 558340, 564712, 571125, 577580, 584078, 590619, 597202, 603827, 610496, 617207, 623960,
    630757, 637597, 644480, 651406, 658375, 665387, 672443, 679542, 686685, 693872, 701102, 708376, 715694, 723055, 730461, 737910,
    745404, 752942, 760525, 768151, 775822, 783538, 791298, 799103, 806952, 814847, 822786, 830770, 838799, 846873, 854993, 863157,
    871367, 879622, 887923, 896269, 904661, 913099, 921582, 930111, 938686, 947307, 955973, 964686, 973445, 982251, 991102, 1000000,
};

static inline uint64_t relative_luminance(const uint8_t *rgb) {
    return 2126ULL * srgb_linear[rgb[0]] + 7152ULL * srgb_linear[rgb[1]] + 722ULL * srgb_linear[rgb[2]];
}

/*
 * 2. op_min_max_luma_u8 & 8. op_histogram_luma_16bin & 7. op_contrast_ratio_wcag_q16
 * BT.601 integer luma: Y = (77*R + 150*G + 29*B) >> 8  (in [0, 255])
 * out_summary[0] = min_luma (0..255)
 * out_summary[1] = max_luma (0..255)
 * out_summary[2] = WCAG (max_linear_L + 0.05)/(min_linear_L + 0.05), Q16.16
 * out_hist16[0..15] = 16-bin luma histogram (bin = Y >> 4)
 */
int ee_op_luma_metrics_u8(
    const uint8_t *rgba,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    uint32_t *out_summary,
    uint32_t *out_hist16
) {
    if (!rgba || !out_summary || !out_hist16 || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    uint32_t min_y = 255;
    uint32_t max_y = 0;
    uint64_t min_linear = 10000000000ULL, max_linear = 0;
    for (int b = 0; b < 16; ++b) out_hist16[b] = 0;

    size_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= total_px; i += 8) {
        size_t off = i << 2;
        __m256i v = _mm256_loadu_si256((const __m256i *)(rgba + off));
        if (_mm256_testz_si256(v, v)) {
            min_y = 0;
            min_linear = 0;
            out_hist16[0] += 8;
            continue;
        }
        for (size_t k = 0; k < 8; ++k) {
            size_t poff = off + (k << 2);
            uint32_t y = (77U * rgba[poff] + 150U * rgba[poff + 1] + 29U * rgba[poff + 2]) >> 8;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
            uint64_t linear = relative_luminance(rgba + poff);
            if (linear < min_linear) min_linear = linear;
            if (linear > max_linear) max_linear = linear;
            out_hist16[(y >> 4) & 0xFU] += 1;
        }
    }
#endif
    for (; i < total_px; ++i) {
        size_t off = i << 2;
        uint32_t y = (77U * rgba[off] + 150U * rgba[off + 1] + 29U * rgba[off + 2]) >> 8;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
        uint64_t linear = relative_luminance(rgba + off);
        if (linear < min_linear) min_linear = linear;
        if (linear > max_linear) max_linear = linear;
        out_hist16[(y >> 4) & 0xFU] += 1;
    }
    out_summary[0] = min_y;
    out_summary[1] = max_y;
    out_summary[2] = (uint32_t)(((max_linear + 500000000ULL) << 16) / (min_linear + 500000000ULL));
    return 0;
}

/*
 * 3. measure_bbox & 4. count_pixels & 10. op_alpha_coverage_q16
 * Mode 0: non_transparent (alpha > 0)
 * Mode 1: colour_match within tolerance (|R-tr|<=tol && |G-tg|<=tol && |B-tb|<=tol && |A-ta|<=tol)
 * out_bbox[0] = matching_pixels
 * out_bbox[1] = x_min (or -1 if empty)
 * out_bbox[2] = y_min (or -1 if empty)
 * out_bbox[3] = x_max (or -1 if empty)
 * out_bbox[4] = y_max (or -1 if empty)
 * out_bbox[5] = alpha_coverage_q16 ((non_transparent_px << 16) / total_px)
 */
int ee_op_bbox_and_count(
    const uint8_t *rgba,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    int32_t mode,
    uint8_t tr,
    uint8_t tg,
    uint8_t tb,
    uint8_t ta,
    int32_t tolerance,
    int64_t *out_bbox
) {
    if (!rgba || !out_bbox || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    int64_t matched = 0;
    int64_t non_transparent = 0;
    int64_t x_min = (int64_t)width;
    int64_t y_min = (int64_t)height;
    int64_t x_max = -1;
    int64_t y_max = -1;

    int zero_can_match = (
        mode == 1 &&
        (int32_t)tr <= tolerance &&
        (int32_t)tg <= tolerance &&
        (int32_t)tb <= tolerance &&
        (int32_t)ta <= tolerance
    );

    for (uint32_t y = 0; y < height; ++y) {
        size_t row_off = (size_t)y * (size_t)width * 4U;
        uint32_t x = 0;
#if defined(__AVX2__)
        if (!zero_can_match) {
            for (; x + 8 <= width; x += 8) {
                size_t off = row_off + ((size_t)x << 2);
                __m256i v = _mm256_loadu_si256((const __m256i *)(rgba + off));
                if (_mm256_testz_si256(v, v)) {
                    continue;
                }
                for (uint32_t k = 0; k < 8; ++k) {
                    uint32_t px = x + k;
                    size_t poff = row_off + ((size_t)px << 2);
                    uint8_t r = rgba[poff];
                    uint8_t g = rgba[poff + 1];
                    uint8_t b = rgba[poff + 2];
                    uint8_t a = rgba[poff + 3];
                    if (a > 0) non_transparent++;
                    int hit = (mode == 0) ? (a > 0) : (
                        abs_i32((int32_t)r - (int32_t)tr) <= tolerance &&
                        abs_i32((int32_t)g - (int32_t)tg) <= tolerance &&
                        abs_i32((int32_t)b - (int32_t)tb) <= tolerance &&
                        abs_i32((int32_t)a - (int32_t)ta) <= tolerance
                    );
                    if (hit) {
                        matched++;
                        if ((int64_t)px < x_min) x_min = (int64_t)px;
                        if ((int64_t)px > x_max) x_max = (int64_t)px;
                        if ((int64_t)y < y_min) y_min = (int64_t)y;
                        if ((int64_t)y > y_max) y_max = (int64_t)y;
                    }
                }
            }
        }
#endif
        for (; x < width; ++x) {
            size_t off = row_off + ((size_t)x << 2);
            uint8_t r = rgba[off];
            uint8_t g = rgba[off + 1];
            uint8_t b = rgba[off + 2];
            uint8_t a = rgba[off + 3];
            if (a > 0) {
                non_transparent++;
            }
            int hit = 0;
            if (mode == 0) {
                hit = (a > 0);
            } else {
                hit = (
                    abs_i32((int32_t)r - (int32_t)tr) <= tolerance &&
                    abs_i32((int32_t)g - (int32_t)tg) <= tolerance &&
                    abs_i32((int32_t)b - (int32_t)tb) <= tolerance &&
                    abs_i32((int32_t)a - (int32_t)ta) <= tolerance
                );
            }
            if (hit) {
                matched++;
                if ((int64_t)x < x_min) x_min = (int64_t)x;
                if ((int64_t)x > x_max) x_max = (int64_t)x;
                if ((int64_t)y < y_min) y_min = (int64_t)y;
                if ((int64_t)y > y_max) y_max = (int64_t)y;
            }
        }
    }

    out_bbox[0] = matched;
    if (matched == 0) {
        out_bbox[1] = -1;
        out_bbox[2] = -1;
        out_bbox[3] = -1;
        out_bbox[4] = -1;
    } else {
        out_bbox[1] = x_min;
        out_bbox[2] = y_min;
        out_bbox[3] = x_max;
        out_bbox[4] = y_max;
    }
    out_bbox[5] = ((int64_t)non_transparent << 16) / (int64_t)total_px;
    return 0;
}

/*
 * 5. count_changed_pixels & op_bbox_aligned_diff_u8
 */
int ee_op_diff_rgba_u8(
    const uint8_t *rgba_a,
    const uint8_t *rgba_b,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    int32_t tolerance,
    int64_t *out_diff
) {
    if (!rgba_a || !rgba_b || !out_diff || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    int64_t changed = 0, added = 0, removed = 0, mod_col = 0;
    int64_t x_min = (int64_t)width, y_min = (int64_t)height;
    int64_t x_max = -1, y_max = -1;

    for (uint32_t y = 0; y < height; ++y) {
        size_t row_off = (size_t)y * (size_t)width * 4U;
        uint32_t x = 0;
#if defined(__AVX2__)
        if (tolerance == 0) {
            for (; x + 8 <= width; x += 8) {
                size_t off = row_off + ((size_t)x << 2);
                __m256i va = _mm256_loadu_si256((const __m256i *)(rgba_a + off));
                __m256i vb = _mm256_loadu_si256((const __m256i *)(rgba_b + off));
                __m256i cmp = _mm256_cmpeq_epi32(va, vb);
                int mask = _mm256_movemask_epi8(cmp);
                if (mask == -1) {
                    continue;
                }
                for (uint32_t k = 0; k < 8; ++k) {
                    uint32_t px = x + k;
                    size_t poff = row_off + ((size_t)px << 2);
                    uint32_t pa, pb;
                    memcpy(&pa, rgba_a + poff, 4);
                    memcpy(&pb, rgba_b + poff, 4);
                    if (pa != pb) {
                        changed++;
                        uint8_t aa = rgba_a[poff + 3];
                        uint8_t ab = rgba_b[poff + 3];
                        if (aa == 0 && ab > 0) added++;
                        else if (aa > 0 && ab == 0) removed++;
                        else mod_col++;
                        if ((int64_t)px < x_min) x_min = (int64_t)px;
                        if ((int64_t)px > x_max) x_max = (int64_t)px;
                        if ((int64_t)y < y_min) y_min = (int64_t)y;
                        if ((int64_t)y > y_max) y_max = (int64_t)y;
                    }
                }
            }
        }
#endif
        for (; x < width; ++x) {
            size_t off = row_off + ((size_t)x << 2);
            int is_diff = 0;
            if (tolerance == 0) {
                uint32_t pa, pb;
                memcpy(&pa, rgba_a + off, 4);
                memcpy(&pb, rgba_b + off, 4);
                is_diff = (pa != pb);
            } else {
                is_diff = (
                    abs_i32((int32_t)rgba_a[off] - (int32_t)rgba_b[off]) > tolerance ||
                    abs_i32((int32_t)rgba_a[off + 1] - (int32_t)rgba_b[off + 1]) > tolerance ||
                    abs_i32((int32_t)rgba_a[off + 2] - (int32_t)rgba_b[off + 2]) > tolerance ||
                    abs_i32((int32_t)rgba_a[off + 3] - (int32_t)rgba_b[off + 3]) > tolerance
                );
            }
            if (is_diff) {
                changed++;
                uint8_t aa = rgba_a[off + 3];
                uint8_t ab = rgba_b[off + 3];
                if (aa == 0 && ab > 0) added++;
                else if (aa > 0 && ab == 0) removed++;
                else mod_col++;
                if ((int64_t)x < x_min) x_min = (int64_t)x;
                if ((int64_t)x > x_max) x_max = (int64_t)x;
                if ((int64_t)y < y_min) y_min = (int64_t)y;
                if ((int64_t)y > y_max) y_max = (int64_t)y;
            }
        }
    }

    out_diff[0] = changed;
    out_diff[1] = added;
    out_diff[2] = removed;
    out_diff[3] = mod_col;
    if (changed == 0) {
        out_diff[4] = -1;
        out_diff[5] = -1;
        out_diff[6] = -1;
        out_diff[7] = -1;
    } else {
        out_diff[4] = x_min;
        out_diff[5] = y_min;
        out_diff[6] = x_max;
        out_diff[7] = y_max;
    }
    return 0;
}

/*
 * 6. op_sobel_edge_density_q16 & 9. op_perceptual_hash_dhash64
 */
int ee_op_sobel_and_dhash64(
    const uint8_t *rgba,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    uint64_t *out_edge_q16,
    uint64_t *out_dhash64
) {
    if (!rgba || !out_edge_q16 || !out_dhash64 || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    uint64_t edge_count = 0;
    if (width >= 3 && height >= 3) {
        for (uint32_t y = 1; y + 1 < height; ++y) {
            size_t r0 = (size_t)(y - 1) * width * 4U;
            size_t r1 = (size_t)y * width * 4U;
            size_t r2 = (size_t)(y + 1) * width * 4U;
            uint32_t x = 1;
#if defined(__AVX2__)
            for (; x + 9 < width; x += 8) {
                size_t off0 = r0 + ((size_t)(x - 1) << 2);
                size_t off1 = r1 + ((size_t)(x - 1) << 2);
                size_t off2 = r2 + ((size_t)(x - 1) << 2);
                __m256i v0a = _mm256_loadu_si256((const __m256i *)(rgba + off0));
                __m256i v0b = _mm256_loadu_si256((const __m256i *)(rgba + off0 + 8));
                __m256i v1a = _mm256_loadu_si256((const __m256i *)(rgba + off1));
                __m256i v1b = _mm256_loadu_si256((const __m256i *)(rgba + off1 + 8));
                __m256i v2a = _mm256_loadu_si256((const __m256i *)(rgba + off2));
                __m256i v2b = _mm256_loadu_si256((const __m256i *)(rgba + off2 + 8));
                __m256i u = _mm256_or_si256(
                    _mm256_or_si256(_mm256_or_si256(v0a, v0b), _mm256_or_si256(v1a, v1b)),
                    _mm256_or_si256(v2a, v2b)
                );
                if (_mm256_testz_si256(u, u)) {
                    continue;
                }
                for (uint32_t k = 0; k < 8; ++k) {
                    uint32_t px = x + k;
                    int32_t p00 = (77 * rgba[r0 + ((px - 1) << 2)] + 150 * rgba[r0 + ((px - 1) << 2) + 1] + 29 * rgba[r0 + ((px - 1) << 2) + 2]) >> 8;
                    int32_t p01 = (77 * rgba[r0 + (px << 2)]       + 150 * rgba[r0 + (px << 2) + 1]       + 29 * rgba[r0 + (px << 2) + 2]) >> 8;
                    int32_t p02 = (77 * rgba[r0 + ((px + 1) << 2)] + 150 * rgba[r0 + ((px + 1) << 2) + 1] + 29 * rgba[r0 + ((px + 1) << 2) + 2]) >> 8;
                    int32_t p10 = (77 * rgba[r1 + ((px - 1) << 2)] + 150 * rgba[r1 + ((px - 1) << 2) + 1] + 29 * rgba[r1 + ((px - 1) << 2) + 2]) >> 8;
                    int32_t p12 = (77 * rgba[r1 + ((px + 1) << 2)] + 150 * rgba[r1 + ((px + 1) << 2) + 1] + 29 * rgba[r1 + ((px + 1) << 2) + 2]) >> 8;
                    int32_t p20 = (77 * rgba[r2 + ((px - 1) << 2)] + 150 * rgba[r2 + ((px - 1) << 2) + 1] + 29 * rgba[r2 + ((px - 1) << 2) + 2]) >> 8;
                    int32_t p21 = (77 * rgba[r2 + (px << 2)]       + 150 * rgba[r2 + (px << 2) + 1]       + 29 * rgba[r2 + (px << 2) + 2]) >> 8;
                    int32_t p22 = (77 * rgba[r2 + ((px + 1) << 2)] + 150 * rgba[r2 + ((px + 1) << 2) + 1] + 29 * rgba[r2 + ((px + 1) << 2) + 2]) >> 8;

                    int32_t gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
                    int32_t gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
                    if (abs_i32(gx) + abs_i32(gy) >= 128) {
                        edge_count++;
                    }
                }
            }
#endif
            for (; x + 1 < width; ++x) {
                int32_t p00 = (77 * rgba[r0 + ((x - 1) << 2)] + 150 * rgba[r0 + ((x - 1) << 2) + 1] + 29 * rgba[r0 + ((x - 1) << 2) + 2]) >> 8;
                int32_t p01 = (77 * rgba[r0 + (x << 2)]       + 150 * rgba[r0 + (x << 2) + 1]       + 29 * rgba[r0 + (x << 2) + 2]) >> 8;
                int32_t p02 = (77 * rgba[r0 + ((x + 1) << 2)] + 150 * rgba[r0 + ((x + 1) << 2) + 1] + 29 * rgba[r0 + ((x + 1) << 2) + 2]) >> 8;
                int32_t p10 = (77 * rgba[r1 + ((x - 1) << 2)] + 150 * rgba[r1 + ((x - 1) << 2) + 1] + 29 * rgba[r1 + ((x - 1) << 2) + 2]) >> 8;
                int32_t p12 = (77 * rgba[r1 + ((x + 1) << 2)] + 150 * rgba[r1 + ((x + 1) << 2) + 1] + 29 * rgba[r1 + ((x + 1) << 2) + 2]) >> 8;
                int32_t p20 = (77 * rgba[r2 + ((x - 1) << 2)] + 150 * rgba[r2 + ((x - 1) << 2) + 1] + 29 * rgba[r2 + ((x - 1) << 2) + 2]) >> 8;
                int32_t p21 = (77 * rgba[r2 + (x << 2)]       + 150 * rgba[r2 + (x << 2) + 1]       + 29 * rgba[r2 + (x << 2) + 2]) >> 8;
                int32_t p22 = (77 * rgba[r2 + ((x + 1) << 2)] + 150 * rgba[r2 + ((x + 1) << 2) + 1] + 29 * rgba[r2 + ((x + 1) << 2) + 2]) >> 8;

                int32_t gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
                int32_t gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
                if (abs_i32(gx) + abs_i32(gy) >= 128) {
                    edge_count++;
                }
            }
        }
    }
    *out_edge_q16 = (edge_count << 16) / total_px;

    uint32_t grid[8][9];
    for (uint32_t gy = 0; gy < 8; ++gy) {
        uint32_t sy = (gy * height) / 8;
        if (sy >= height) sy = height - 1;
        for (uint32_t gx = 0; gx < 9; ++gx) {
            uint32_t sx = (gx * width) / 9;
            if (sx >= width) sx = width - 1;
            size_t off = ((size_t)sy * width + sx) << 2;
            grid[gy][gx] = (77U * rgba[off] + 150U * rgba[off + 1] + 29U * rgba[off + 2]) >> 8;
        }
    }
    uint64_t dhash = 0;
    uint32_t bit = 0;
    for (uint32_t gy = 0; gy < 8; ++gy) {
        for (uint32_t gx = 0; gx < 8; ++gx) {
            if (grid[gy][gx] < grid[gy][gx + 1]) {
                dhash |= (1ULL << bit);
            }
            bit++;
        }
    }
    *out_dhash64 = dhash;
    return 0;
}

int64_t ee_extract_fg_runs(
    const uint8_t *rgba,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    int32_t *out_runs,
    size_t max_runs
) {
    if (!rgba || !out_runs || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    size_t n_runs = 0;
    for (uint32_t y = 0; y < height; ++y) {
        size_t row_off = (size_t)y * (size_t)width * 4U;
        uint32_t x = 0;
        while (x < width) {
#if defined(__AVX2__)
            if (x + 8 <= width) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(rgba + row_off + ((size_t)x << 2)));
                if (_mm256_testz_si256(v, v)) {
                    x += 8;
                    continue;
                }
            }
#endif
            if (rgba[row_off + ((size_t)x << 2) + 3] > 0) {
                uint32_t x0 = x;
                x++;
                while (x < width && rgba[row_off + ((size_t)x << 2) + 3] > 0) {
                    x++;
                }
                if (n_runs < max_runs) {
                    out_runs[n_runs * 3] = (int32_t)y;
                    out_runs[n_runs * 3 + 1] = (int32_t)x0;
                    out_runs[n_runs * 3 + 2] = (int32_t)(x - 1);
                }
                n_runs++;
            } else {
                x++;
            }
        }
    }
    return (int64_t)n_runs;
}

int64_t ee_extract_diff_runs(
    const uint8_t *rgba_a,
    const uint8_t *rgba_b,
    size_t byte_len,
    uint32_t width,
    uint32_t height,
    int32_t tolerance,
    int32_t *out_runs,
    size_t max_runs
) {
    if (!rgba_a || !rgba_b || !out_runs || width == 0 || height == 0) return -1;
    size_t total_px = (size_t)width * (size_t)height;
    if (byte_len < total_px * 4) return -2;

    size_t n_runs = 0;
    for (uint32_t y = 0; y < height; ++y) {
        size_t row_off = (size_t)y * (size_t)width * 4U;
        if (tolerance == 0 && memcmp(rgba_a + row_off, rgba_b + row_off, (size_t)width * 4U) == 0) {
            continue;
        }
        uint32_t x = 0;
        while (x < width) {
            size_t off = row_off + ((size_t)x << 2);
            int is_diff = 0;
            if (tolerance == 0) {
                uint32_t pa, pb;
                memcpy(&pa, rgba_a + off, 4);
                memcpy(&pb, rgba_b + off, 4);
                is_diff = (pa != pb);
            } else {
                is_diff = (
                    abs_i32((int32_t)rgba_a[off] - (int32_t)rgba_b[off]) > tolerance ||
                    abs_i32((int32_t)rgba_a[off + 1] - (int32_t)rgba_b[off + 1]) > tolerance ||
                    abs_i32((int32_t)rgba_a[off + 2] - (int32_t)rgba_b[off + 2]) > tolerance ||
                    abs_i32((int32_t)rgba_a[off + 3] - (int32_t)rgba_b[off + 3]) > tolerance
                );
            }
            if (is_diff) {
                uint32_t x0 = x;
                x++;
                while (x < width) {
                    size_t noff = row_off + ((size_t)x << 2);
                    int ndiff = 0;
                    if (tolerance == 0) {
                        uint32_t npa, npb;
                        memcpy(&npa, rgba_a + noff, 4);
                        memcpy(&npb, rgba_b + noff, 4);
                        ndiff = (npa != npb);
                    } else {
                        ndiff = (
                            abs_i32((int32_t)rgba_a[noff] - (int32_t)rgba_b[noff]) > tolerance ||
                            abs_i32((int32_t)rgba_a[noff + 1] - (int32_t)rgba_b[noff + 1]) > tolerance ||
                            abs_i32((int32_t)rgba_a[noff + 2] - (int32_t)rgba_b[noff + 2]) > tolerance ||
                            abs_i32((int32_t)rgba_a[noff + 3] - (int32_t)rgba_b[noff + 3]) > tolerance
                        );
                    }
                    if (ndiff) x++;
                    else break;
                }
                if (n_runs < max_runs) {
                    out_runs[n_runs * 3] = (int32_t)y;
                    out_runs[n_runs * 3 + 1] = (int32_t)x0;
                    out_runs[n_runs * 3 + 2] = (int32_t)(x - 1);
                }
                n_runs++;
            } else {
                x++;
            }
        }
    }
    return (int64_t)n_runs;
}
