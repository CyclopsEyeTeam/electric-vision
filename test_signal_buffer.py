"""Unit & parity tests for ElectricSignalTempBuffer, fast SIMD signal ingest, and lazy proxy materialization."""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
import unittest

from PIL import Image, ImageDraw

import visual_sources
from desktop_sight import DesktopSight, ORIGIN, SIZE, _coarse, _movement, _where
from electric_vision_agent_bridge import ElectricVisionAgentSession
from prod_gate_enclave import explicit_zeroize_bytearray
from signal_buffer import ElectricSignalTempBuffer, extract_visual_signal


def _make_test_window_image(variant: int) -> Image.Image:
    img = Image.new("RGB", (960, 640), (24, 34, 58))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 959, 51], fill=(16, 18, 22))
    if variant == 1:
        draw.rectangle([80, 120, 380, 420], fill=(220, 75, 45))
        draw.ellipse([520, 160, 780, 420], fill=(45, 195, 130))
        draw.rectangle([400, 460, 880, 580], fill=(240, 210, 70))
    elif variant == 2:
        draw.rectangle([180, 140, 480, 440], fill=(220, 75, 45))
        draw.ellipse([600, 180, 900, 520], fill=(60, 140, 245))
    elif variant == 3:
        draw.rectangle([80, 120, 380, 420], fill=(220, 75, 45))
        draw.ellipse([520, 160, 780, 420], fill=(45, 195, 130))
        draw.rectangle([420, 470, 900, 590], fill=(240, 210, 70))
    else:
        off = 40 * (variant % 12)
        draw.rectangle([off, 90, off + 280, 520], fill=(180, 60 + (variant * 13) % 180, 210))
    return img


def _reference_live_analyze(img: Image.Image, stream, tick: int, title: str):
    """Reference baseline live-translation pass (full ingest_frame + pixel-by-pixel chroma)."""
    header_crop = min(52, img.height // 8) if img.height > 160 else 0
    content = img.crop((0, header_crop, img.width, img.height)) if header_crop else img
    coarse = _coarse(content)
    mean_rgb = content.resize((1, 1), Image.BOX).getpixel((0, 0))
    mean_hex = "#%02x%02x%02x" % mean_rgb
    warmth_score = (mean_rgb[0] - mean_rgb[2]) + 0.35 * (mean_rgb[1] - mean_rgb[2])
    mean_warmth = (
        "neutral"
        if max(mean_rgb) - min(mean_rgb) < 14
        else ("warm" if warmth_score > 18 else "cool" if warmth_score < -15 else "balanced")
    )
    chrom = [n for n in visual_sources._NAMED if n[0] not in ("black", "white", "grey")]
    g48 = content.resize((48, 36), Image.BILINEAR)
    sat, sat_rgb = {}, {}
    for gy in range(1, 35):
        for gx in range(1, 47):
            px = g48.getpixel((gx, gy))
            peak = max(px)
            span = peak - min(px)
            if peak >= 38 and span >= 22:
                norm = tuple(min(255, round(c * (225.0 / peak))) for c in px) if peak < 190 else px
                cn = min(chrom, key=lambda n: sum((a - b) ** 2 for a, b in zip(norm, n[1])))[0]
                weight = 2 if span >= 55 and peak >= 85 else 1
                sat[cn] = sat.get(cn, 0) + weight
                if cn not in sat_rgb:
                    sat_rgb[cn] = [px[0] * weight, px[1] * weight, px[2] * weight, weight]
                else:
                    sat_rgb[cn][0] += px[0] * weight
                    sat_rgb[cn][1] += px[1] * weight
                    sat_rgb[cn][2] += px[2] * weight
                    sat_rgb[cn][3] += weight
    top_chrom = [k for k, v in sorted(sat.items(), key=lambda kv: -kv[1]) if v >= 10][:3]
    chrom_values = []
    for cn in top_chrom:
        sr, sg, sb, sc = sat_rgb[cn]
        crgb = (round(sr / sc), round(sg / sc), round(sb / sc))
        chrom_values.append({
            "name": cn,
            "rgb": list(crgb),
            "hex": "#%02x%02x%02x" % crgb,
            "luminance": round(0.299 * crgb[0] + 0.587 * crgb[1] + 0.114 * crgb[2], 1),
        })
    colour_desc = f"mostly {coarse['colour']} ({mean_hex}, rgb {mean_rgb[0]},{mean_rgb[1]},{mean_rgb[2]}, {mean_warmth})"
    if top_chrom:
        extra = [f"{c['name']} ({c['hex']})" for c in chrom_values if c["name"] != coarse["colour"]]
        if extra:
            colour_desc = (
                f"{', '.join(extra)} on {coarse['colour']} ({mean_hex})"
                if coarse["colour"] in ("grey", "white", "black") and sum(sat.values()) >= 45
                else f"mostly {coarse['colour']} ({mean_hex}) with {', '.join(extra)}"
            )
    rgba, _ = visual_sources.fit_to_canvas(content.convert("RGBA"), *SIZE)
    if content is not img:
        content.close()
    img.close()
    r = stream.ingest_frame(f"v{tick}", [], rgba=rgba, origin={"kind": ORIGIN}, origin_attestation=ORIGIN)
    tr = r.get("transition_from_previous") or {}
    win = r.get("temporal_window_summary") or {}
    ratio = tr.get("change_ratio")
    regime = (win.get("settling_and_oscillation_detector") or {}).get("regime")
    where = _where(tr.get("diff_bbox"), *SIZE) if ratio else None
    move = _movement(ratio)
    scene_tag = title.split("·", 1)[1].strip() if "·" in str(title) else ""
    words = (f"scene '{scene_tag}' — " if scene_tag else "") + f"{coarse['light']}, {colour_desc}, brightest at the {coarse['brightest']}"
    if ratio is not None:
        words += "; still" if move == "still" else f"; {'a lot' if move == 'lots of movement' else 'a little'} moving, {where if where == 'all over' else 'mostly at the ' + str(where)}"
    if regime in ("settled", "oscillating_cycle"):
        words += "; it has settled" if regime == "settled" else "; it keeps going back and forth"
    return {
        "ok": True,
        "title": title,
        "words": words,
        "move": move,
        "measures": {
            "change_ratio": ratio,
            "where_changed": where,
            "regime": regime,
            "mean_luminance": coarse["mean_luminance"],
            "colour_values": {
                "name": coarse["colour"],
                "rgb": list(mean_rgb),
                "hex": mean_hex,
                "warmth": mean_warmth,
                "swatches": chrom_values,
            },
        },
        "receipt": {
            "pixel_sha256": r.get("native_pixel_sha256"),
            "dhash64": r.get("dhash64"),
            "tick": r.get("tick"),
        },
    }


class SignalBufferTests(unittest.TestCase):
    def test_signal_buffer_parity_across_repeats_settling_cycles_and_eviction(self):
        seq = [1, 1, 1, 2, 1, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11]
        ref_eye = DesktopSight(lightweight=False)
        upg_eye = DesktopSight(lightweight=True)
        buf = ElectricSignalTempBuffer(max_entries=8)
        try:
            ref_stream = ref_eye._session.create_continuous_stream("src_ref", width=SIZE[0], height=SIZE[1], max_window_frames=8)
            upg_stream = upg_eye._session.create_continuous_stream("src_upg", width=SIZE[0], height=SIZE[1], max_window_frames=8)

            for tick, k in enumerate(seq, 1):
                ref_obs = _reference_live_analyze(_make_test_window_image(k), ref_stream, tick, "App · Meadow")

                cand_img = _make_test_window_image(k)
                raw_bytes = bytearray(cand_img.tobytes())
                raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                explicit_zeroize_bytearray(raw_bytes)
                sz = cand_img.size
                active = buf.get("win_a", "0x100")
                if active is not None and active.raw_sha256 == raw_sha and active.size == sz:
                    cand_img.close()
                    img_arg = None
                else:
                    img_arg = cand_img

                upg_obs = buf.observe_frame(
                    target_id="win_a",
                    xid="0x100",
                    title="App · Meadow",
                    stream=upg_stream,
                    tick=tick,
                    img=img_arg,
                    raw_sha256=raw_sha,
                    size=sz,
                )
                upg_obs.pop("at", None)
                upg_obs.pop("buffered", None)
                self.assertEqual(upg_obs, ref_obs, f"Mismatch at tick {tick}")

            st = buf.stats()
            self.assertEqual(st["raw_hits"], 5)
            self.assertLessEqual(len(upg_eye._session._source_dimensions), 8)
        finally:
            ref_eye.close()
            upg_eye.close()

    def test_lazy_enclave_boots_and_materializes_proxy_on_demand(self):
        run_dir = tempfile.mkdtemp(prefix="cee-lazy-test-", dir="/tmp")
        os.chmod(run_dir, 0o700)
        try:
            with ElectricVisionAgentSession(run_dir=run_dir, default_resolution=SIZE, lazy_enclave=True) as sess:
                self.assertIsNone(sess._harness)
                stream = sess.create_continuous_stream("src_lazy", width=SIZE[0], height=SIZE[1], max_window_frames=4)
                rgba1, _ = visual_sources.fit_to_canvas(_make_test_window_image(1).convert("RGBA"), *SIZE)
                rgba2, _ = visual_sources.fit_to_canvas(_make_test_window_image(2).convert("RGBA"), *SIZE)
                r1 = stream.ingest_signal_frame("v1", rgba=rgba1, origin={"kind": ORIGIN}, origin_attestation=ORIGIN)
                r2 = stream.ingest_signal_frame("v2", rgba=rgba2, origin={"kind": ORIGIN}, origin_attestation=ORIGIN)
                self.assertIsNone(sess._harness)
                self.assertEqual(r1["tick"], 1)
                self.assertGreater(r2["transition_from_previous"]["change_ratio"], 0.0)

                insp = sess.inspect_canvas("src_lazy", "v1")
                cmp_res = sess.compare_canvases(("src_lazy", "v1"), ("src_lazy", "v2"))
                self.assertIsNotNone(sess._harness)
                self.assertEqual(insp["status"], "ok")
                self.assertEqual(
                    cmp_res["native_diff_metrics"]["change_ratio"],
                    r2["transition_from_previous"]["change_ratio"],
                )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
