"""Active Electric Signal Temp Buffer (`ElectricSignalTempBuffer`).

Holds active per-source electric & visual signals in a bounded volatile memory buffer
instead of re-running live pixel-to-mark translation, image decoding, and chromatic
swatch loops on every look.

Guarantees:
  * Memory-only: raw pixels are never stored in `ElectricSignalTempBuffer` (only derived
    colour/luminance signals and cryptographic digests `raw_sha256`, `pixel_sha256`, `dhash64`);
    active 512x512 RGBA ring frames remain confined inside `_multi_res_vault` with
    `explicit_bzero` eviction.
  * Bit-exact output parity with live `analyze_frame` + `ContinuousElectricEyeStream`:
    same `words`, `move`, `measures`, `settling_and_oscillation_detector` regimes, and
    `receipt` (`pixel_sha256`, `dhash64`, `tick`).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import visual_sources
from desktop_sight import ORIGIN, REGIONS, SIZE, _light, _movement, _where


_CHROM_PALETTE: List[Tuple[str, int, int, int]] = [
    (name, int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for name, rgb in visual_sources._NAMED
    if name not in ("black", "white", "grey")
]
_CHROM_CACHE: Dict[Tuple[int, int, int], str] = {}
_CHROM_CACHE_MAX = 16384


def _nearest_chrom_name(r: int, g: int, b: int, peak: int) -> str:
    key = (r, g, b)
    hit = _CHROM_CACHE.get(key)
    if hit is not None:
        return hit
    if peak < 190:
        scale = 225.0 / peak
        nr = min(255, round(r * scale))
        ng = min(255, round(g * scale))
        nb = min(255, round(b * scale))
    else:
        nr, ng, nb = r, g, b
    best_name = _CHROM_PALETTE[0][0]
    best_dist = 1 << 30
    for cname, cr, cg, cb in _CHROM_PALETTE:
        dr = nr - cr
        dg = ng - cg
        db = nb - cb
        d = dr * dr + dg * dg + db * db
        if d < best_dist:
            best_dist = d
            best_name = cname
    if len(_CHROM_CACHE) >= _CHROM_CACHE_MAX:
        _CHROM_CACHE.clear()
    _CHROM_CACHE[key] = best_name
    return best_name


def extract_visual_signal(content: Image.Image) -> Dict[str, Any]:
    """Extracts coarse 3x3 light/region and 48x36 chromatic swatches with memoized RGB lookup."""
    grid3 = content.resize((3, 3), Image.BOX)
    raw3 = grid3.tobytes()
    mean_rgb = content.resize((1, 1), Image.BOX).getpixel((0, 0))
    mean_lum_raw = 0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]

    best_lum = -1.0
    bx, by = 0, 0
    for y in range(3):
        for x in range(3):
            off = (y * 3 + x) * 3
            clum = 0.299 * raw3[off] + 0.587 * raw3[off + 1] + 0.114 * raw3[off + 2]
            if clum > best_lum:
                best_lum = clum
                bx, by = x, y

    coarse = {
        "light": _light(mean_lum_raw),
        "colour": visual_sources.colour_name(tuple(mean_rgb)),
        "brightest": REGIONS[by][bx],
        "mean_luminance": round(mean_lum_raw, 1),
    }
    mean_hex = "#%02x%02x%02x" % mean_rgb
    warmth_score = (mean_rgb[0] - mean_rgb[2]) + 0.35 * (mean_rgb[1] - mean_rgb[2])
    mean_warmth = (
        "neutral"
        if max(mean_rgb) - min(mean_rgb) < 14
        else ("warm" if warmth_score > 18 else "cool" if warmth_score < -15 else "balanced")
    )

    g48 = content.resize((48, 36), Image.BILINEAR)
    raw48 = g48.tobytes()
    sat: Dict[str, int] = {}
    sat_rgb: Dict[str, List[int]] = {}
    for gy in range(1, 35):
        row_base = gy * 48 * 3
        for gx in range(1, 47):
            off = row_base + gx * 3
            r = raw48[off]
            g = raw48[off + 1]
            b = raw48[off + 2]
            peak = r if r >= g and r >= b else (g if g >= b else b)
            low = r if r <= g and r <= b else (g if g <= b else b)
            span = peak - low
            if peak >= 38 and span >= 22:
                cn = _nearest_chrom_name(r, g, b, peak)
                weight = 2 if span >= 55 and peak >= 85 else 1
                sat[cn] = sat.get(cn, 0) + weight
                acc = sat_rgb.get(cn)
                if acc is None:
                    sat_rgb[cn] = [r * weight, g * weight, b * weight, weight]
                else:
                    acc[0] += r * weight
                    acc[1] += g * weight
                    acc[2] += b * weight
                    acc[3] += weight

    top_chrom = [k for k, v in sorted(sat.items(), key=lambda kv: -kv[1]) if v >= 10][:3]
    chrom_values: List[Dict[str, Any]] = []
    for cn in top_chrom:
        sr, sg, sb, sc = sat_rgb[cn]
        crgb = (round(sr / sc), round(sg / sc), round(sb / sc))
        chrom_values.append(
            {
                "name": cn,
                "rgb": list(crgb),
                "hex": "#%02x%02x%02x" % crgb,
                "luminance": round(0.299 * crgb[0] + 0.587 * crgb[1] + 0.114 * crgb[2], 1),
            }
        )

    colour_desc = (
        f"mostly {coarse['colour']} ({mean_hex}, rgb {mean_rgb[0]},{mean_rgb[1]},{mean_rgb[2]}, {mean_warmth})"
    )
    if top_chrom:
        extra = [f"{c['name']} ({c['hex']})" for c in chrom_values if c["name"] != coarse["colour"]]
        if extra:
            colour_desc = (
                f"{', '.join(extra)} on {coarse['colour']} ({mean_hex})"
                if coarse["colour"] in ("grey", "white", "black") and sum(sat.values()) >= 45
                else f"mostly {coarse['colour']} ({mean_hex}) with {', '.join(extra)}"
            )

    return {
        "coarse": coarse,
        "mean_rgb": list(mean_rgb),
        "mean_hex": mean_hex,
        "mean_warmth": mean_warmth,
        "chrom_values": chrom_values,
        "colour_desc": colour_desc,
    }


@dataclass
class ActiveSignalEntry:
    target_id: str
    xid: str
    raw_sha256: str
    size: Tuple[int, int]
    visual: Dict[str, Any]
    pixel_sha256: str
    dhash64: str
    last_tick: int
    updated_at_ms: int
    hits: int = 0


class ElectricSignalTempBuffer:
    """
    Bounded in-memory buffer holding active electric signals per observed window/source.
    Prevents re-decoding, re-quantizing, and re-translating unchanged or live frames into
    128x128 synthetic proxy marks on every sweep.
    """

    def __init__(self, *, max_entries: int = 32) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, ActiveSignalEntry]" = OrderedDict()
        self.raw_hits = 0
        self.canvas_hits = 0
        self.updates = 0
        self.evictions = 0

    def get(self, target_id: str, xid: Optional[str] = None) -> Optional[ActiveSignalEntry]:
        entry = self._entries.get(target_id)
        if entry is None:
            return None
        if xid is not None and entry.xid != xid:
            self._entries.pop(target_id, None)
            self.evictions += 1
            return None
        return entry

    def drop(self, target_id: str) -> bool:
        if self._entries.pop(target_id, None) is not None:
            self.evictions += 1
            return True
        return False

    def clear(self) -> None:
        self.evictions += len(self._entries)
        self._entries.clear()

    def stats(self) -> Dict[str, int]:
        return {
            "active_signals": len(self._entries),
            "raw_hits": self.raw_hits,
            "canvas_hits": self.canvas_hits,
            "updates": self.updates,
            "evictions": self.evictions,
        }

    def observe_frame(
        self,
        *,
        target_id: str,
        xid: str,
        title: str,
        stream: Any,
        tick: int,
        img: Optional[Image.Image],
        raw_sha256: str,
        size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Produces an Electric Eye observation using the Active Signal Temp Buffer:
          - Fast path 1 (`img is None` — raw capture digest matched active buffered signal):
            skips all Pillow crops/resizes, chroma scans, `fit_to_canvas`, and SIMD/proxy work,
            advancing the stream's temporal state via `stream.ingest_buffered_repeat`.
          - Fast path 2 (cropped content `512x512` RGBA digest matched active buffered signal, e.g.
            only window titlebar changed): reuses active visual/electric signal via `ingest_buffered_repeat`.
          - Active update path (genuinely new visual signal): computes visual signal + native AVX2
            `stream.ingest_signal_frame` (no live 128x128 proxy mark translation) and updates buffer.
        """
        now_ms = int(time.time() * 1000)
        entry = self.get(target_id, xid)

        if img is None and entry is not None and entry.raw_sha256 == raw_sha256 and entry.size == size:
            self.raw_hits += 1
            entry.hits += 1
            entry.last_tick = tick
            entry.updated_at_ms = now_ms
            self._entries.move_to_end(target_id)
            r = stream.ingest_buffered_repeat(f"v{tick}")
            vis = entry.visual
        else:
            assert img is not None, "Initial or changed frame must supply an RGB Image"
            header_crop = min(52, img.height // 8) if img.height > 160 else 0
            content = img.crop((0, header_crop, img.width, img.height)) if header_crop else img
            try:
                rgba, _ = visual_sources.fit_to_canvas(content.convert("RGBA"), *SIZE)
                rgba_sha = f"sha256:{hashlib.sha256(rgba).hexdigest()}"
                if entry is not None and entry.pixel_sha256 == rgba_sha:
                    # Cropped content is pixel-identical (e.g. only window decoration above header_crop changed)
                    del rgba
                    self.canvas_hits += 1
                    entry.raw_sha256 = raw_sha256
                    entry.size = size
                    entry.hits += 1
                    entry.last_tick = tick
                    entry.updated_at_ms = now_ms
                    self._entries.move_to_end(target_id)
                    r = stream.ingest_buffered_repeat(f"v{tick}")
                    vis = entry.visual
                else:
                    vis = extract_visual_signal(content)
                    r = stream.ingest_signal_frame(
                        f"v{tick}",
                        rgba=rgba,
                        origin={"kind": ORIGIN},
                        origin_attestation=ORIGIN,
                    )
                    del rgba
                    self.updates += 1
                    new_entry = ActiveSignalEntry(
                        target_id=target_id,
                        xid=xid,
                        raw_sha256=raw_sha256,
                        size=size,
                        visual=vis,
                        pixel_sha256=r["native_pixel_sha256"],
                        dhash64=r["dhash64"],
                        last_tick=tick,
                        updated_at_ms=now_ms,
                    )
                    self._entries[target_id] = new_entry
                    self._entries.move_to_end(target_id)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                        self.evictions += 1
            finally:
                if content is not img:
                    content.close()
                img.close()

        coarse = vis["coarse"]
        tr = r.get("transition_from_previous") or {}
        win = r.get("temporal_window_summary") or {}
        ratio = tr.get("change_ratio")
        regime = (win.get("settling_and_oscillation_detector") or {}).get("regime")
        where = _where(tr.get("diff_bbox"), *SIZE) if ratio else None
        move = _movement(ratio)
        scene_tag = title.split("·", 1)[1].strip() if "·" in str(title) else ""
        words = (
            (f"scene '{scene_tag}' — " if scene_tag else "")
            + f"{coarse['light']}, {vis['colour_desc']}, brightest at the {coarse['brightest']}"
        )
        if ratio is not None:
            words += (
                "; still"
                if move == "still"
                else f"; {'a lot' if move == 'lots of movement' else 'a little'} moving, {where if where == 'all over' else 'mostly at the ' + str(where)}"
            )
        if regime in ("settled", "oscillating_cycle"):
            words += "; it has settled" if regime == "settled" else "; it keeps going back and forth"

        return {
            "ok": True,
            "at": now_ms,
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
                    "rgb": vis["mean_rgb"],
                    "hex": vis["mean_hex"],
                    "warmth": vis["mean_warmth"],
                    "swatches": vis["chrom_values"],
                },
            },
            "receipt": {
                "pixel_sha256": r.get("native_pixel_sha256"),
                "dhash64": r.get("dhash64"),
                "tick": r.get("tick"),
            },
            "buffered": bool(r.get("signal_buffer_hit", False)),
        }
