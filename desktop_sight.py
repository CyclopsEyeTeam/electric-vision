"""Continuous desktop sight for the Companion: Electric Eye on the companion's own glances.

On the owner's word (2026-11-04: "Yes, on my word"; "electric eye is there to be upgraded as a continous sight"),
the Companion may glance at the live desktop. This is the only live source Electric Eye accepts, and it is
said honestly as `origin_attestation="companion_desktop_glance"`. All other live capture stays refused.

Privacy contract:
  * a glance is captured into memory through a pipe (`import -window root png:-`) and never written to disk;
  * Electric Eye keeps a small ring of frames in its closure vault and zeroizes evicted frames (explicit_bzero);
  * the caller receives only measurements and plain words (no pixels, no text read from the screen);
  * Electric Eye's semantic boundary still holds: it measures (light, colour, change, motion, settling), and never
    says what anything on the screen means.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from PIL import Image

import visual_sources
from electric_vision_agent_bridge import ElectricVisionAgentSession

ORIGIN = "companion_desktop_glance"
SIZE = (512, 512)
REGIONS = [["top left", "top", "top right"], ["left", "middle", "right"], ["bottom left", "bottom", "bottom right"]]


def _capture() -> Image.Image:
    """One desktop frame, in memory only."""
    out = subprocess.run(["import", "-window", "root", "png:-"], capture_output=True, timeout=10, check=True)
    png = bytearray(out.stdout)
    try:
        img = Image.open(io.BytesIO(png)).convert("RGB")
        img.load()
        return img
    finally:
        png[:] = b"\x00" * len(png)  # do not leave the encoded frame lying in memory


def _light(v: float) -> str:
    return "dark" if v < 70 else "medium light" if v < 160 else "bright"


def _coarse(img: Image.Image) -> Dict[str, Any]:
    """Whole-screen light and colour, plus where it is brightest: from a 3x3 average (9 pixels)."""
    grid = img.resize((3, 3), Image.BOX)
    cells = [[grid.getpixel((x, y)) for x in range(3)] for y in range(3)]
    mean = img.resize((1, 1), Image.BOX).getpixel((0, 0))
    lum = lambda p: 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
    bx, by = max(((x, y) for y in range(3) for x in range(3)), key=lambda c: lum(cells[c[1]][c[0]]))
    return {"light": _light(lum(mean)), "colour": visual_sources.colour_name(tuple(mean)), "brightest": REGIONS[by][bx],
            "mean_luminance": round(lum(mean), 1)}


def _where(bbox: Optional[List[int]], w: int, h: int) -> Optional[str]:
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    if (x1 - x0) > 0.66 * w and (y1 - y0) > 0.66 * h:
        return "all over"
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    return REGIONS[min(2, int(cy * 3))][min(2, int(cx * 3))]


def _movement(ratio: Optional[float]) -> str:
    if ratio is None:
        return "first look"
    return "still" if ratio < 0.005 else "a little movement" if ratio < 0.08 else "lots of movement"


class DesktopSight:
    """A long-lived companion eye. glance() is cheap (well under a second)."""

    def __init__(self, window: int = 8, *, lightweight: bool = False) -> None:
        self._run = tempfile.mkdtemp(prefix="cee-", dir="/tmp")
        os.chmod(self._run, 0o700)
        self._lightweight = bool(lightweight)
        self._session = ElectricVisionAgentSession(
            run_dir=self._run,
            default_resolution=SIZE,
            lazy_enclave=self._lightweight,
        )
        self._session.__enter__()
        self._stream = self._session.create_continuous_stream("src_companion_glance", width=SIZE[0], height=SIZE[1], max_window_frames=window)
        self._tick = 0

    def glance(self) -> Dict[str, Any]:
        t0 = time.time()
        img = _capture()
        coarse = _coarse(img)
        rgba, _ = visual_sources.fit_to_canvas(img.convert("RGBA"), *SIZE)
        img.close()
        self._tick += 1
        r = self._stream.ingest_frame(f"v{self._tick}", [], rgba=rgba, origin={"kind": ORIGIN}, origin_attestation=ORIGIN)
        del rgba
        tr = r.get("transition_from_previous") or {}
        win = r.get("temporal_window_summary") or {}
        ratio = tr.get("change_ratio")
        where = _where(tr.get("diff_bbox"), *SIZE) if ratio else None
        regime = (win.get("settling_and_oscillation_detector") or {}).get("regime")
        moved = max(((abs(t["displacement"]["dx"]) + abs(t["displacement"]["dy"]), t["displacement"])
                     for t in ((tr.get("motion_summary") or {}).get("component_trajectories") or [])), default=(0, None), key=lambda x: x[0])[1]
        move = _movement(ratio)
        words = f"{coarse['light']}, mostly {coarse['colour']}, brightest at the {coarse['brightest']}"
        if ratio is not None:
            words += "; nothing changed since my last look" if move == "still" else f"; {'a lot' if move == 'lots of movement' else 'a little'} changed since my last look, {where if where == 'all over' else 'mostly at the ' + where}"
        if regime in ("settled", "oscillating_cycle"):
            words += "; it has settled" if regime == "settled" else "; it keeps going back and forth"
        return {
            "at": int(t0 * 1000),
            "origin": ORIGIN,
            # Same coarse shape as the Cyclops Eye line, so the codec and diary read it unchanged.
            "eyeLine": f"Eye: fresh look, {coarse['light']}, mostly {coarse['colour'].split()[-1]}, {move if move != 'first look' else 'still'}, some detail",
            "words": words,
            "measures": {"change_ratio": ratio, "where_changed": where, "dhash_step": tr.get("dhash64_hamming_distance"),
                         "dirty_tile_ratio": (tr.get("dirty_tiles_16x16") or {}).get("dirty_tile_ratio"), "regime": regime,
                         "largest_shift": moved, "mean_luminance": coarse["mean_luminance"], "brightest": coarse["brightest"]},
            # Layer A receipt: Electric Eye's own digests for this glance (no pixels). Interpretations link to it.
            "receipt": {"pixel_sha256": r.get("native_pixel_sha256"), "dhash64": r.get("dhash64"), "diff_replay_digest": tr.get("diff_replay_digest"), "tick": r.get("tick")},
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    def close(self) -> None:
        try:
            self._session.__exit__(None, None, None)
        finally:
            try:
                os.rmdir(self._run)
            except OSError:
                pass
