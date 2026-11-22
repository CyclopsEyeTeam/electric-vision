#!/usr/bin/env python3
"""
electric_vision_mcp.py

Scoped JSON-RPC 2.0 / MCP server for Cyclops CLI Electric Vision.

One server per chat. It draws agent-supplied marks onto private in-memory canvases
(128x128 to 1920x1080), measures them with the native operator library
(`libprod_gate_simd.so`) and a 128x128 measurement harness, compares and previews
canvases, keeps a bounded multi-frame stream, and can return PNGs to MCP hosts.
Nothing here captures the screen or a camera: every canvas is drawn from marks.
Helper processes (signing, pixel vault, audit log, input filter) are children of
this server and die with it.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import base64
import copy
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageStat

from electric_eye_harness import (
    CLOSED_FORM_OPERATORS,
    PROHIBITED_SEMANTIC_QUERY_TYPES,
    SOURCE_ID_REGEX,
    SOURCE_VERSION_REGEX,
    UNCERTAINTY_STYLE_SPEC,
    assert_no_raw_buffer_leakage,
)
from electric_vision_agent_bridge import (
    SUPPORTED_RESOLUTIONS,
    ContinuousElectricEyeStream,
    ElectricVisionAgentSession,
    _rasterize_normalized_specs,
    normalize_marks_for_canvas,
)
from prod_gate_enclave import explicit_zeroize_bytearray
import visual_sources

MAX_IMAGE_BYTES = 512 * 1024  # Keep base64 PNG comfortably inside 1 MiB MCP response limit
EMBLEM_WORLD_PATH = Path(__file__).resolve().parent / "emblem" / "world.json"
STREAM_SOURCE_ID = "src_continuous_stream"
# Native canvases kept for later diff/ghost/inspect calls; oldest ones retire first.
MAX_RETAINED_SOURCES = 48
NATIVE_BYTES_BUDGET = 192 * 1024 * 1024
# Stream frame PNGs and their metadata kept for packets.
MAX_RETAINED_FRAMES = 64
# A push stops taking new frames after this long so the reply beats the 10 s client timeout.
PUSH_TIME_BUDGET_S = 4.5
MAX_SEQUENCE = 999_999
SEMANTIC_QUERY_HINTS = (
    "semantic", "meaning", "aesthetic", "beaut", "intent", "ocr", "text", "read", "describe",
    "caption", "emotion", "judg", "interpret", "recogn", "classif", "identify", "label", "style",
)

_SOURCE_ID = {"type": "string", "pattern": "^src_[a-z0-9_]{1,48}$"}
_SOURCE_VERSION = {"type": "string", "pattern": "^v[0-9]{1,6}$"}
_MARKS = {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "object"}}
_WIDTH = {"enum": [128, 256, 512, 1920]}
# Colours to find on the canvas (Agent World III: all five reviewers asked, as round 2 had): each #rrggbb, matched within
# `tolerance` per channel among drawn pixels.
_COLOURS = {"type": "array", "minItems": 1, "maxItems": 12, "items": {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"}}
_TOLERANCE = {"type": "integer", "minimum": 0, "maximum": 255}
COLOUR_TOLERANCE = 40
_HEIGHT = {"enum": [128, 256, 512, 1080]}
_MIRROR = {"enum": ["none", "x", "y", "xy", "h", "v", "hv"]}

TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "eye.electric.status",
        "description": (
            "Read this chat’s Electric Vision server: helper-process health, registered canvases and "
            "the frame stream. Read-only; never captures the screen."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "eye.electric.inspect",
        "description": (
            "Draw a canvas from your marks (Prism k-marks in 0..1 units, cyclopsdraw shapes in pixels, "
            "CYCLOPS_DRAW world operations, drawing.json vectors) or preset:\"emblem\", then measure it: "
            "mean colour, luma, WCAG contrast, edge density, dHash64, bounding box and connected components "
            "(readable decimals are under native_simd_metrics.readable). Sizes 128x128, 256x256, 512x512, "
            "1920x1080. Prism example: {\"k\":\"ring\",\"x\":0.5,\"y\":0.5,\"r\":0.2,\"w\":0.02,\"c\":[255,0,0]} (k: arc, curve, dot, fill, glow, line, ring, spray); cyclopsdraw example: {\"shape\":\"rect\",\"x\":10,\"y\":10,\"width\":40,\"height\":20,\"fill\":{\"color\":\"#00ff00\"}}. mirror: x = left-right, y = top-bottom, xy = both. The canvas is drawn by you, not "
            "captured; origin_attestation is your own unchecked label. issue_receipt:true adds an HMAC "
            "receipt from this server session (it cannot be checked after the chat’s server stops). "
            "image:true returns the PNG on MCP hosts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": _SOURCE_ID,
                "source_version": _SOURCE_VERSION,
                "preset": {"enum": ["emblem"]},
                "marks": _MARKS,
                "width": _WIDTH,
                "height": _HEIGHT,
                "mirror": _MIRROR,
                "origin_attestation": {"enum": ["synthetic_fixture", "owner_supplied_picture"]},
                "issue_receipt": {"type": "boolean"},
                "image": {"type": "boolean"},
                "colours": _COLOURS,
                "tolerance": _TOLERANCE,
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "eye.electric.diff",
        "description": (
            "Compare two canvases (before_marks/after_marks, or registered source versions). "
            "native_diff_metrics are at the canvas resolution; derived_representations (change mask, "
            "difference map, motion, colour delta, spatial relationship) are on a 128x128 downsample, "
            "with scale factors in derived_representations_resolution. Also reports 16x16 dirty tiles."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id_a": _SOURCE_ID,
                "source_version_a": _SOURCE_VERSION,
                "source_id_b": _SOURCE_ID,
                "source_version_b": _SOURCE_VERSION,
                "before_marks": _MARKS,
                "after_marks": _MARKS,
                "width": _WIDTH,
                "height": _HEIGHT,
                "tolerance": {"type": "integer", "minimum": 0, "maximum": 255},
                "image": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "eye.electric.ghost",
        "description": (
            "Preview proposed ghost_marks over a base canvas without changing it, and measure the "
            "hypothetical result (observed=false, uncertainty_state=\"predicted\")."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "base_source_id": _SOURCE_ID,
                "base_source_version": _SOURCE_VERSION,
                "base_marks": _MARKS,
                "ghost_marks": _MARKS,
                "width": _WIDTH,
                "height": _HEIGHT,
                "mirror": _MIRROR,
                "image": {"type": "boolean"},
            },
            "required": ["ghost_marks"],
            "additionalProperties": False,
        },
    },
    {
        "name": "eye.electric.stream",
        "description": (
            "Multi-frame stream. push: add frames (each a list of marks) to a rolling window and get "
            "per-frame change, dHash drift, settling/oscillation and centroid motion; stage_ghost_marks "
            "scores a prediction against the next frame. packet: frames after after_sequence as a contact "
            "sheet (image:true on MCP hosts). status: window summary. reset: drop all frames; sequence "
            "numbers keep counting. Size and window are fixed once a stream starts; reset to change them. "
            "A long push may stop early and report not_ingested_count; push the rest again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"enum": ["push", "packet", "status", "reset"]},
                "frames": {"type": "array", "minItems": 1, "maxItems": 32, "items": _MARKS},
                "marks": _MARKS,
                "stage_ghost_marks": _MARKS,
                "width": _WIDTH,
                "height": _HEIGHT,
                "window_size": {"type": "integer", "minimum": 2, "maximum": 32},
                "settle_ticks": {"type": "integer", "minimum": 1, "maximum": 16},
                "issue_receipts": {"type": "boolean"},
                "after_sequence": {"type": "integer", "minimum": 0, "maximum": MAX_SEQUENCE},
                "max_frames": {"type": "integer", "minimum": 1, "maximum": 60},
                "image": {"type": "boolean"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "eye.electric.look",
        "description": (
            'Look at a real picture and measure it like inspect: an image file (png, jpg, gif, webp, bmp, tiff), an SVG (file, or svg text), a video (one frame at `at` seconds, or frames {start, every, count} added to the stream), an HTML page, or a play-engine painting or live game via play {painting} or play {game} (games are captured headless after `at` seconds, or as frames). The picture is fitted onto a canvas (chosen from its shape unless width/height are given); its source_id/source_version then work with diff, ghost and refuse. Shapes are measured on non-background pixels: background auto (default) makes the colour filling the border transparent; keep or #rrggbb change that. Files must be inside this chat’s folders. image:true returns the PNG on MCP hosts.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "minLength": 1, "maxLength": 4096},
                "svg": {"type": "string", "minLength": 1, "maxLength": 49152},
                "play": {"type": "object"},
                "at": {"type": "number", "minimum": 0, "maximum": 86400},
                "frames": {"type": "object"},
                "width": _WIDTH,
                "height": _HEIGHT,
                "window_size": {"type": "integer", "minimum": 2, "maximum": 32},
                "background": {"type": "string", "pattern": "^(auto|keep|#[0-9a-fA-F]{6})$"},
                "colours": _COLOURS,
                "tolerance": _TOLERANCE,
                # Set by Cyclops after it resolves play {game}/{painting} or an HTML file; never by the agent.
                "page_root": {"type": "string", "maxLength": 4096},
                "page_query": {"type": "string", "maxLength": 256},
                "source_id": _SOURCE_ID,
                "source_version": _SOURCE_VERSION,
                "image": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "eye.electric.refuse",
        "description": (
            "Run one closed-form measurement on a canvas, measured on its 128x128 downsample. "
            "query_type is one of " + ", ".join(sorted(CLOSED_FORM_OPERATORS)) + ". Semantic, aesthetic, "
            "OCR or intent questions are refused with a refusal_code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_type": {"type": "string", "minLength": 1, "maxLength": 128},
                "source_id": _SOURCE_ID,
                "source_version": _SOURCE_VERSION,
                "parameters": {"type": "object"},
            },
            "required": ["query_type"],
            "additionalProperties": False,
        },
    },
]


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise RuntimeError(f"Electric Vision startup refused: unsafe directory {path}")
    if st.st_uid != os.getuid():
        raise RuntimeError("Electric Vision startup refused: directory not owned by current user")
    os.chmod(path, 0o700)
    return path


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".ev-status-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode_png_bounded(image: Image.Image, max_dim: int = 512) -> Tuple[bytes, str]:
    work = image
    if image.width > max_dim or image.height > max_dim:
        work = image.copy()
        work.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    work.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Rendered PNG exceeds bounded MCP image payload limit")
    digest = hashlib.sha256(raw).hexdigest()
    return raw, digest


def _render_specs_to_png(
    specs: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    *,
    ghost_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[bytes, str]:
    img = Image.frombytes("RGBA", (width, height), _rasterize_normalized_specs(specs, width, height))
    if ghost_specs:
        # Ghost pixels replace the canvas wherever the ghost layer has any alpha.
        ghost = Image.frombytes("RGBA", (width, height), _rasterize_normalized_specs(ghost_specs, width, height))
        img.paste(ghost, (0, 0), ghost.getchannel("A").point(lambda v: 255 if v > 0 else 0))
    # Composite onto dark background (#0c1018) so transparent canvases are clearly visible
    bg = Image.new("RGBA", (width, height), (12, 16, 24, 255))
    bg.alpha_composite(img)
    return _encode_png_bounded(bg.convert("RGB"))


def _render_diff_panel_png(
    specs_a: Sequence[Dict[str, Any]],
    specs_b: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Tuple[bytes, str]:
    img_a = Image.frombytes("RGBA", (width, height), _rasterize_normalized_specs(specs_a, width, height))
    img_b = Image.frombytes("RGBA", (width, height), _rasterize_normalized_specs(specs_b, width, height))
    return _render_diff_panel_from_images(img_a, img_b)


def _render_diff_panel_from_images(img_a: Image.Image, img_b: Image.Image) -> Tuple[bytes, str]:
    width, height = img_b.size
    if img_a.size != img_b.size:
        img_a = img_a.resize(img_b.size)
    # Changed pixels (any channel differs) in magenta, the rest as the dimmed after-image.
    delta = ImageChops.difference(img_a, img_b)
    changed = delta.getchannel(0)
    for band in range(1, 4):
        changed = ImageChops.lighter(changed, delta.getchannel(band))
    changed = changed.point(lambda v: 255 if v > 0 else 0)
    dim = img_b.convert("RGB").point(lambda v: v // 3).convert("RGBA")
    dim.paste((255, 64, 129, 255), (0, 0), changed)
    cell_w, cell_h = 192, 192
    panel = Image.new("RGB", (cell_w * 3, cell_h + 24), "#141923")
    draw = ImageDraw.Draw(panel)
    for idx, (label, layer) in enumerate([("BEFORE", img_a), ("AFTER", img_b), ("DIFF MASK", dim)]):
        bg = Image.new("RGBA", (width, height), (12, 16, 24, 255))
        bg.alpha_composite(layer)
        thumb = bg.convert("RGB")
        thumb.thumbnail((cell_w - 8, cell_h - 8))
        x0 = idx * cell_w + (cell_w - thumb.width) // 2
        y0 = 4 + (cell_h - thumb.height) // 2
        panel.paste(thumb, (x0, y0))
        draw.text((idx * cell_w + 8, cell_h + 4), label, fill="white")
    return _encode_png_bounded(panel, max_dim=640)


def _render_temporal_contact_sheet_png(frames_meta: List[Dict[str, Any]]) -> Tuple[bytes, str]:
    cols = min(4, max(1, len(frames_meta)))
    rows = math.ceil(len(frames_meta) / float(cols))
    cell_w, cell_h = 160, 140
    panel = Image.new("RGB", (cols * cell_w, rows * cell_h), "#141923")
    draw = ImageDraw.Draw(panel)
    for index, frame in enumerate(frames_meta):
        raw = Path(frame["image_path"]).read_bytes()
        with Image.open(io.BytesIO(raw)) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((cell_w - 8, cell_h - 24))
            cx = (index % cols) * cell_w
            cy = (index // cols) * cell_h
            panel.paste(thumb, (cx + (cell_w - thumb.width) // 2, cy + 4 + (cell_h - 24 - thumb.height) // 2))
            label = f"#{frame['sequence']} {frame['source_version']}"
            draw.text((cx + 6, cy + cell_h - 18), label, fill="white")
    return _encode_png_bounded(panel, max_dim=640)


def _load_emblem_marks() -> List[Dict[str, Any]]:
    data = json.loads(EMBLEM_WORLD_PATH.read_text(encoding="utf-8"))
    for key in ("operations", "marks", "objects"):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError("Emblem world.json missing operations array")


def _q16(value: Any) -> Optional[float]:
    return round(int(value) / 65536.0, 4) if isinstance(value, int) else None


# A connected component this share of the canvas or larger counts as a shape; smaller ones are specks and glyph pieces.
SIGNIFICANT_SHAPE_SHARE = 0.001


def _compact_simd_metrics(metrics: Dict[str, Any], max_components: int = 8, pixels: Optional[int] = None) -> Dict[str, Any]:
    out = copy.deepcopy(metrics)
    # The *_q16 fields are fixed-point (value * 65536); spell them out for readers.
    mean = (out.get("op_mean_rgba_u8") or {}).get("mean_rgba_q16")
    out["readable"] = {
        "mean_rgba_0_255": [_q16(v) for v in mean] if isinstance(mean, list) else None,
        "contrast_ratio_wcag": _q16(out.get("op_contrast_ratio_wcag_q16")),
        "edge_density_0_1": _q16(out.get("op_sobel_edge_density_q16")),
        "alpha_coverage_0_1": _q16(out.get("op_alpha_coverage_q16")),
        "note": "*_q16 fields are fixed-point: divide by 65536.",
    }
    cc = out.get("op_connected_components_4c")
    if isinstance(cc, dict) and isinstance(cc.get("components"), list):
        comps = cc["components"]
        cc["components_total"] = int(cc.get("component_count", len(comps)))
        if pixels:
            cc["significant_shapes"] = sum(1 for c in comps if int(c.get("area_px", 0)) >= SIGNIFICANT_SHAPE_SHARE * pixels)
            cc["significant_shape_min_area_px"] = math.ceil(SIGNIFICANT_SHAPE_SHARE * pixels)
        if len(comps) > max_components:
            cc["components"] = sorted(
                comps, key=lambda c: (-int(c.get("area_px", 0)), int(c.get("component_id", 0)))
            )[:max_components]
            cc["components_truncated"] = True
    return out



def _where(x: float, y: float) -> str:
    col = "left" if x < 0.4 else "right" if x > 0.6 else ""
    row = "upper" if y < 0.4 else "lower" if y > 0.6 else ""
    return " ".join(p for p in (row, col) if p) or "centre"


def colour_presence(img: Image.Image, colours: Sequence[str], tolerance: int = COLOUR_TOLERANCE) -> List[Dict[str, Any]]:
    """Each colour's drawn pixels (every channel within `tolerance`, alpha > 0): count, share of the canvas, centroid."""
    img = img.convert("RGBA")
    w, h = img.size
    rgb, drawn = img.convert("RGB"), img.getchannel("A").point(lambda v: 255 if v else 0)
    out = []
    for hexcode in colours:
        colour = tuple(int(hexcode[i:i + 2], 16) for i in (1, 3, 5))
        delta = ImageChops.difference(rgb, Image.new("RGB", (w, h), colour))
        near = ImageChops.lighter(ImageChops.lighter(delta.getchannel(0), delta.getchannel(1)), delta.getchannel(2))
        mask = ImageChops.multiply(near.point(lambda v, t=tolerance: 255 if v <= t else 0), drawn)
        pixels = int(round(ImageStat.Stat(mask).sum[0] / 255))
        row = {"hex": hexcode.lower(), "pixels": pixels, "share": round(pixels / (w * h), 5), "present": pixels > 0}
        if pixels:
            cols = list(mask.resize((w, 1), Image.BOX).getdata())
            rows = list(mask.resize((1, h), Image.BOX).getdata())
            cx = sum((i + 0.5) * v for i, v in enumerate(cols)) / max(1e-9, sum(cols)) / w
            cy = sum((i + 0.5) * v for i, v in enumerate(rows)) / max(1e-9, sum(rows)) / h
            bbox = mask.getbbox()
            row.update({"centroid": {"x": round(cx, 3), "y": round(cy, 3), "where": _where(cx, cy)}, "bbox": list(bbox) if bbox else None})
        out.append(row)
    return out


class _PixelFrame:
    """One real frame for the stream: fitted RGBA, its PNG, and where it came from."""

    def __init__(self, rgba: bytes, png: bytes, origin: Dict[str, Any]) -> None:
        self.rgba, self.png, self.origin = rgba, png, origin


def _fitted(img: Image.Image, width: int, height: int, background: str = "auto", *, want_image: bool = True) -> Tuple[bytes, bytes, Dict[str, Any]]:
    keyed, keying = visual_sources.key_background(img, background)
    rgba, placement = visual_sources.fit_to_canvas(keyed, width, height)
    placement.update(keying)
    # The measurements are of the fitted canvas; the picture's own size is reported beside them (Agent World II: agents
    # read the canvas size as the file's).
    placement["file_pixels"] = {"width": img.width, "height": img.height}
    canvas = Image.frombytes("RGBA", (width, height), rgba)
    # The drawn pixels' own mean colour: a keyed-out background keeps its RGB under alpha 0, so the canvas mean over
    # alpha is not it (that division reported a mean luma of 655).
    drawn = canvas.getchannel("A").point(lambda v: 255 if v else 0)
    if drawn.getbbox():
        placement["drawn_mean_rgb_0_255"] = [round(v, 1) for v in ImageStat.Stat(canvas.convert("RGB"), drawn).mean]
    png = b""
    if want_image:
        bg = Image.new("RGBA", (width, height), (12, 16, 24, 255))
        bg.alpha_composite(canvas)
        png, _ = _encode_png_bounded(bg.convert("RGB"))
    return rgba, png, placement


# Source ids the harness accepts (file kinds such as "svg" or "png" are forbidden id words).
LOOK_SOURCE_IDS = {"image": "src_look_picture", "svg": "src_look_vector", "video": "src_look_footage",
                   "page": "src_look_game", "painting": "src_look_painting"}
MAX_LOOK_VIDEO_FRAMES = 24
MAX_LOOK_PAGE_FRAMES = 4
LOOK_TIME_BUDGET_S = 5.0


class ElectricVisionServer:
    def __init__(self, electric_root: Path, agent_slot: str) -> None:
        self.electric_root = _ensure_private_dir(electric_root.resolve())
        # One server per chat root: a second one would unlink this one's sockets and logs.
        self._lock_handle = open(self.electric_root / "server.lock", "a+")
        os.chmod(self.electric_root / "server.lock", 0o600)
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_handle.close()
            raise RuntimeError("another Electric Vision server is already running for this chat")
        self.frames_dir = _ensure_private_dir(self.electric_root / "frames")
        self._pending_frame_retirements: set[Path] = set()
        try:
            self._cleanup_frame_files()
        except BaseException:
            self._lock_handle.close()
            raise
        self.status_path = self.electric_root / "status.json"
        self.agent_slot = agent_slot
        scoped_run = self.electric_root / "run"
        if len(str(scoped_run / "q.sock").encode("utf-8")) <= 100:
            self.run_dir = scoped_run
            self._run_dir_is_temp = False
        else:
            self.run_dir = Path(tempfile.mkdtemp(prefix="cee-", dir="/tmp"))
            os.chmod(self.run_dir, 0o700)
            self._run_dir_is_temp = True
        self.session: Optional[ElectricVisionAgentSession] = None
        self.stream: Optional[ContinuousElectricEyeStream] = None
        self.stream_frames_meta: List[Dict[str, Any]] = []
        self.stream_sequence = 0
        self.compiled_specs_cache: Dict[Tuple[str, str], Tuple[List[Dict[str, Any]], int, int]] = {}
        # PNGs of registered real pictures (images, SVG, video frames, pages), for image:true replies.
        self.picture_png_cache: Dict[Tuple[str, str], bytes] = {}
        self.picture_rgba_cache: Dict[Tuple[str, str], Tuple[bytes, int, int]] = {}
        self._max_version: Dict[str, int] = {}
        self._source_order: List[Tuple[str, str]] = []
        self._pending_source_retirements: set[Tuple[str, str]] = set()
        self.retired_source_count = 0
        self.last_summary: Dict[str, Any] = {
            "phase": "idle",
            "agent_slot": self.agent_slot,
            "registered_sources": [],
            "stream_frames": 0,
            "stream_regime": "uninitialized",
            "updated_at": int(time.time() * 1000),
        }

    # ----- source versions -------------------------------------------------------------------

    def _taken(self, sess: ElectricVisionAgentSession, key: Tuple[str, str]) -> bool:
        harness_sources = getattr(sess._harness, "_sources", {})
        return key in sess._source_dimensions or key in harness_sources

    def _next_version(self, sess: ElectricVisionAgentSession, sid: str) -> str:
        # Versions are never reused, even after a canvas retires, because the harness keeps them immutable.
        idx = self._max_version.get(sid, 0) + 1
        while self._taken(sess, (sid, f"v{idx}")):
            idx += 1
        return f"v{idx}"

    def _latest_version(self, sess: ElectricVisionAgentSession, sid: str, default: str = "v1") -> str:
        numbers = [int(v[1:]) for (s_id, v) in sess._source_dimensions if s_id == sid]
        return f"v{max(numbers)}" if numbers else default

    def _register(
        self,
        sess: ElectricVisionAgentSession,
        sid: str,
        sver: str,
        marks: Sequence[Any],
        *,
        width: int,
        height: int,
        mirror: str = "none",
        protect: Sequence[Tuple[str, str]] = (),
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if sid == STREAM_SOURCE_ID:
            raise ValueError(f"{STREAM_SOURCE_ID} belongs to eye.electric.stream; use another source_id")
        reg = sess.register_synthetic_canvas(sid, sver, marks, width=width, height=height, mirror=mirror, **kwargs)
        specs, _ = normalize_marks_for_canvas(marks, width, height, mirror=mirror)
        self.compiled_specs_cache[(sid, sver)] = (specs, width, height)
        self._track(sess, sid, sver, protect)
        return reg

    def _track(self, sess: ElectricVisionAgentSession, sid: str, sver: str, protect: Sequence[Tuple[str, str]] = ()) -> None:
        self._max_version[sid] = max(self._max_version.get(sid, 0), int(sver[1:]))
        self._source_order.append((sid, sver))
        self._enforce_source_budget(sess, protect=[*protect, (sid, sver)])

    def _handle_look(self, args: Dict[str, Any]) -> Dict[str, Any]:
        given = [k for k in ("file", "svg") if k in args]
        if len(given) != 1:
            return self._error("eye.electric.look needs exactly one of 'file' (a path in this chat's folders) or 'svg' (SVG text)", "invalid_arguments")
        play = args.get("play") if isinstance(args.get("play"), dict) else None
        if "svg" in args:
            kind, path = "svg", None
            origin: Dict[str, Any] = {"kind": "svg_text", "preview": visual_sources.describe_svg_text(str(args["svg"]))}
        else:
            path = str(args["file"])
            kind = visual_sources.classify(path)
            origin = {"kind": kind, "name": os.path.basename(path)}
        if play:
            origin["play"] = {k: str(v)[:80] for k, v in play.items() if k in ("painting", "game")}
        id_kind = "painting" if play and "painting" in play else kind
        want_image = bool(args.get("image", False))
        width, height = args.get("width"), args.get("height")
        if (width is None) != (height is None):
            return self._error("give both width and height, or neither", "invalid_arguments")
        frames_spec = args.get("frames")
        background = str(args.get("background", "auto"))

        if frames_spec is not None:
            if kind not in ("video", "page"):
                return self._error("'frames' is for video files and HTML/game pages; use a single look for pictures", "invalid_arguments")
            start = float(frames_spec.get("start", 0.0))
            every = float(frames_spec.get("every", 1.0))
            cap = MAX_LOOK_VIDEO_FRAMES if kind == "video" else MAX_LOOK_PAGE_FRAMES
            count = int(frames_spec.get("count", 4))
            if count < 1 or count > cap or every <= 0 or start < 0:
                return self._error(f"frames needs start >= 0, every > 0 and a count of 1..{cap} for a {kind}", "invalid_arguments")
            times = [round(start + i * every, 3) for i in range(count)]
            cw: Optional[int]
            ch: Optional[int]
            if self.stream is not None:
                cw, ch = self.stream.width, self.stream.height
                if width is not None and (int(width), int(height)) != (cw, ch):
                    return self._error(f"The stream is {cw}x{ch}; call eye.electric.stream action:'reset' to look at frames at another size.", "invalid_arguments")
            elif width is not None:
                cw, ch = int(width), int(height)
            else:
                cw = ch = None  # decided from the first frame
            started = time.monotonic()
            pixel_frames: List[_PixelFrame] = []
            with ExitStack() as inputs:
                if kind == "video":
                    snapshot, info = inputs.enter_context(visual_sources.checked_video(path))
                    origin["video"] = info
                    times = [t for t in times if t < max(info["duration_s"], 0.001)] or [0.0]
                for t in times:
                    if pixel_frames and time.monotonic() - started > LOOK_TIME_BUDGET_S:
                        break
                    if kind == "video":
                        img = visual_sources._video_frames_snapshot(snapshot, [t])[0]
                    else:
                        vw, vh = (cw, ch) if cw else visual_sources.MAX_PAGE_SIZE
                        img = visual_sources.capture_page(path, width=vw, height=vh, at_ms=[int(t * 1000) or 1],  # type: ignore[arg-type]
                                                          root=args.get("page_root"), query=str(args.get("page_query", "")))[0]
                    if cw is None:
                        cw, ch = visual_sources.default_canvas(img.width, img.height, SUPPORTED_RESOLUTIONS)
                    rgba, png, placement = _fitted(img, cw, ch, background, want_image=want_image)  # type: ignore[arg-type]
                    pixel_frames.append(_PixelFrame(rgba, png, {**origin, "at_s": t, **placement}))
            reply = self._handle_stream({
                "action": "push", "frames": pixel_frames, "width": cw, "height": ch, "image": want_image,
                **({"window_size": int(args["window_size"])} if "window_size" in args else {}),
            })
            body = reply["structuredContent"]
            if not reply["isError"]:
                body["looked_at"] = dict(origin)
                body["frame_times_s"] = times[: len(pixel_frames)]
                skipped = len(times) - len(pixel_frames)
                if skipped:
                    body["frames_not_captured"] = skipped
                    body["note"] = f"Stopped after {LOOK_TIME_BUDGET_S:g}s; look again from {times[len(pixel_frames)]} s for the rest."
                reply["content"][-1]["text"] = json.dumps(body, sort_keys=True)
            return reply

        sess = self.ensure_session()
        at = float(args.get("at", 0.0))
        if kind == "image":
            img = visual_sources.load_image(path)  # type: ignore[arg-type]
        elif kind == "svg":
            img = visual_sources.render_svg(str(args["svg"])) if path is None else visual_sources.load_svg(path)
        elif kind == "video":
            with visual_sources.checked_video(path) as (snapshot, info):
                origin["video"] = info
                img = visual_sources._video_frames_snapshot(snapshot, [at])[0]
            origin["at_s"] = at
        else:
            vw, vh = (int(width), int(height)) if width is not None else visual_sources.MAX_PAGE_SIZE
            img = visual_sources.capture_page(path, width=vw, height=vh, at_ms=[int(at * 1000) or 1000],  # type: ignore[arg-type]
                                              root=args.get("page_root"), query=str(args.get("page_query", "")))[0]
            origin["at_s"] = at or 1.0
        if width is not None:
            cw, ch = int(width), int(height)
        else:
            cw, ch = visual_sources.default_canvas(img.width, img.height, SUPPORTED_RESOLUTIONS)
        rgba, png, placement = _fitted(img, cw, ch, background, want_image=want_image)
        origin.update(placement)
        sid = str(args.get("source_id") or LOOK_SOURCE_IDS[id_kind])
        if sid == STREAM_SOURCE_ID:
            return self._error(f"{STREAM_SOURCE_ID} belongs to eye.electric.stream; use another source_id", "invalid_arguments")
        sver = str(args.get("source_version") or self._next_version(sess, sid))
        reg = sess.register_pixel_canvas(sid, sver, rgba, cw, ch, origin=origin)
        if png:
            self.picture_png_cache[(sid, sver)] = png
        self.picture_rgba_cache[(sid, sver)] = (rgba, cw, ch)
        # Keep lazy image representations until the corresponding source retires.
        # _track enforces both the source count and native-byte budgets.
        self._track(sess, sid, sver)
        return self._inspect_reply(sess, sid, sver, reg, issue_receipt=False, want_image=want_image, extra={"looked_at": origin},
                                   wanted=args.get("colours"), tolerance=args.get("tolerance"))


    def _enforce_source_budget(self, sess: ElectricVisionAgentSession, protect: Sequence[Tuple[str, str]]) -> None:
        for key in sorted(self._pending_source_retirements):
            self._retire_source(sess, key)
        self._source_order = [k for k in self._source_order if k in sess._source_dimensions]

        def native_bytes() -> int:
            return sum(sess._source_dimensions[k][0] * sess._source_dimensions[k][1] * 4 for k in self._source_order)

        keep = set(protect)
        while len(self._source_order) > MAX_RETAINED_SOURCES or native_bytes() > NATIVE_BYTES_BUDGET:
            victim = next((k for k in self._source_order if k not in keep), None)
            if victim is None:
                break
            self._retire_source(sess, victim)

    def _retire_source(self, sess: ElectricVisionAgentSession, key: Tuple[str, str]) -> None:
        self._pending_source_retirements.add(key)
        sess.evict_synthetic_canvas(*key)
        self.compiled_specs_cache.pop(key, None)
        self.picture_png_cache.pop(key, None)
        self.picture_rgba_cache.pop(key, None)
        self._source_order.remove(key)
        self.retired_source_count += 1
        self._pending_source_retirements.discard(key)

    # ----- lifecycle ---------------------------------------------------------------------------

    def ensure_session(self) -> ElectricVisionAgentSession:
        if self.session is None:
            _ensure_private_dir(self.run_dir)
            sess = ElectricVisionAgentSession(run_dir=str(self.run_dir), default_resolution=(256, 256))
            try:
                sess.__enter__()
            except BaseException:
                # Stop any helper that did start before the failure.
                try:
                    sess.__exit__(None, None, None)
                except Exception:
                    pass
                raise
            self.session = sess
            self.last_summary["phase"] = "ready"
            self._save_status()
        return self.session

    def _save_status(self, *, summary: Optional[Dict[str, Any]] = None, strict: bool = False) -> None:
        value = self.last_summary if summary is None else summary
        value["updated_at"] = int(time.time() * 1000)
        try:
            _atomic_json(self.status_path, value)
        except Exception:
            if strict:
                raise

    def close(self) -> None:
        if self.session is not None:
            try:
                self.session.__exit__(None, None, None)
            except Exception:
                pass
            self.session = None
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir, ignore_errors=True)
        try:
            self._cleanup_frame_files()
        finally:
            self._lock_handle.close()

    def _cleanup_frame_files(self) -> None:
        failure = None
        if self.frames_dir.exists():
            with os.scandir(self.frames_dir) as entries:
                for entry in entries:
                    if not re.fullmatch(r"frame-[0-9]{6}\.(?:png|rgba)(?:\.(?:partial|tmp))?", entry.name):
                        continue
                    if not entry.is_file(follow_symlinks=False) and not entry.is_symlink():
                        continue
                    self._pending_frame_retirements.add(self.frames_dir / entry.name)
        for path in sorted(self._pending_frame_retirements):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                if failure is None:
                    failure = exc
            else:
                self._pending_frame_retirements.discard(path)
        if failure is not None:
            raise failure

    # ----- replies -----------------------------------------------------------------------------

    def _result(
        self,
        body: Dict[str, Any],
        images: Optional[List[bytes]] = None,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        body = {**body, "served_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        content: List[Dict[str, Any]] = []
        if images:
            for png_bytes in images:
                content.append({
                    "type": "image",
                    "mimeType": "image/png",
                    "data": base64.b64encode(png_bytes).decode("ascii"),
                })
        content.append({"type": "text", "text": json.dumps(body, sort_keys=True)})
        return {"isError": is_error, "structuredContent": body, "content": content}

    def _error(self, message: str, code: str = "electric_vision_refused") -> Dict[str, Any]:
        return self._result({"error": {"code": code, "message": message}}, is_error=True)

    # ----- stream helpers ----------------------------------------------------------------------

    def _reset_stream(self) -> int:
        released = self.stream.release() if self.stream is not None else 0
        self.stream = None
        self.stream_frames_meta.clear()
        self._cleanup_frame_files()
        return released

    def _retire_old_frames(self, frames: List[Dict[str, Any]]) -> None:
        while len(frames) > MAX_RETAINED_FRAMES:
            old = frames[0]
            for name in ("image_path", "rgba_path"):
                path = Path(old[name])
                self._pending_frame_retirements.add(path)
                path.unlink(missing_ok=True)
                self._pending_frame_retirements.discard(path)
            frames.pop(0)

    def _ensure_frame_png(self, frame: Dict[str, Any]) -> None:
        if frame.get("sha256"):
            return
        canvas = Image.frombytes("RGBA", (frame["width"], frame["height"]), Path(frame["rgba_path"]).read_bytes())
        bg = Image.new("RGBA", canvas.size, (12, 16, 24, 255))
        bg.alpha_composite(canvas)
        png, sha = _encode_png_bounded(bg.convert("RGB"))
        Path(frame["image_path"]).write_bytes(png)
        os.chmod(frame["image_path"], 0o600)
        frame["sha256"] = sha

    def _canvas_image(self, key: Tuple[str, str]) -> Optional[Image.Image]:
        """The canvas's own RGBA pixels (transparency kept): drawn from its marks, or the looked-at picture as fitted."""
        if key in self.compiled_specs_cache:
            specs, cw, ch = self.compiled_specs_cache[key]
            return Image.frombytes("RGBA", (cw, ch), _rasterize_normalized_specs(specs, cw, ch))
        held = self.picture_rgba_cache.get(key)
        return Image.frombytes("RGBA", (held[1], held[2]), held[0]) if held else None

    def _picture_png(self, key: Tuple[str, str]) -> Optional[bytes]:
        """The canvas as a PNG for MCP hosts: drawn from its marks, or the stored real picture."""
        if key in self.compiled_specs_cache:
            specs, cw, ch = self.compiled_specs_cache[key]
            return _render_specs_to_png(specs, cw, ch)[0]
        cached = self.picture_png_cache.get(key)
        if cached is not None:
            return cached
        canvas = self._canvas_image(key)
        if canvas is None:
            return None
        bg = Image.new("RGBA", canvas.size, (12, 16, 24, 255))
        bg.alpha_composite(canvas)
        png, _ = _encode_png_bounded(bg.convert("RGB"))
        self.picture_png_cache[key] = png
        return png

    def _inspect_reply(
        self,
        sess: ElectricVisionAgentSession,
        sid: str,
        sver: str,
        reg: Dict[str, Any],
        *,
        issue_receipt: bool,
        want_image: bool,
        extra: Optional[Dict[str, Any]] = None,
        wanted: Optional[Sequence[str]] = None,
        tolerance: Optional[int] = None,
    ) -> Dict[str, Any]:
        insp = sess.inspect_canvas(sid, sver, issue_observation_receipt=issue_receipt)
        images: List[bytes] = []
        image_meta: Optional[Dict[str, Any]] = None
        png_bytes = self._picture_png((sid, sver)) if want_image else None
        if png_bytes is not None:
            images.append(png_bytes)
            latest_png = self.electric_root / "latest-canvas.png"
            latest_png.write_bytes(png_bytes)
            os.chmod(latest_png, 0o600)
            image_meta = {"mimeType": "image/png", "sha256": hashlib.sha256(png_bytes).hexdigest(), "byte_length": len(png_bytes)}

        self.last_summary.update({
            "phase": "inspected",
            "last_source": f"{sid}@{sver}",
            "last_dhash64": insp["native_simd_metrics"]["op_perceptual_hash_dhash64"],
            "last_observed": bool(insp["render"]["observed"]),
            "last_uncertainty_state": insp["render"]["uncertainty_state"],
            "registered_sources": [f"{k[0]}@{k[1]}" for k in sorted(sess._source_dimensions.keys())][-64:],
        })
        self._save_status()

        compact_insp = {
            "status": insp["status"],
            "source_id": insp["source_id"],
            "source_version": insp["source_version"],
            "resolution": insp["resolution"],
            "native_simd_backend": insp["native_simd_backend"],
            "native_simd_metrics": _compact_simd_metrics(insp["native_simd_metrics"], pixels=int(insp["resolution"]["width"]) * int(insp["resolution"]["height"])),
            "native_replay_digest": insp["native_replay_digest"],
            "render": insp["render"],
            "observation_receipt": insp["observation_receipt"],
            "hsm_receipt_countersignature": insp["hsm_receipt_countersignature"],
            "accessibility": insp["accessibility"],
            "elapsed_ms": insp["elapsed_ms"],
        }
        # A picture's shapes with their own colours (the canvas mean mixes them all): from the picture's pixels.
        colours = None
        picture = self._canvas_image((sid, sver))
        comps = ((compact_insp["native_simd_metrics"] or {}).get("op_connected_components_4c") or {}).get("components")
        if picture is not None and isinstance(comps, list) and comps:
            try:
                colours = visual_sources.component_colours(picture, comps)
            except (OSError, ValueError):
                colours = None
        presence, tol = None, COLOUR_TOLERANCE if tolerance is None else int(tolerance)
        if wanted:
            canvas = picture
            presence = colour_presence(canvas, wanted, tol) if canvas is not None else [{"hex": c.lower(), "unchecked": "the canvas pixels are no longer held"} for c in wanted]
        body = {
            "status": "ok",
            "source_id": sid,
            "source_version": sver,
            **(extra or {}),
            **({"component_colours": colours} if colours else {}),
            **({"colour_presence": presence, "colour_tolerance": tol} if presence is not None else {}),
            "registration": reg,
            "inspection": compact_insp,
            "epistemic_style": UNCERTAINTY_STYLE_SPEC[insp["render"]["uncertainty_state"]],
            **({"rendered_image": image_meta} if image_meta else {}),
        }
        if issue_receipt:
            body["receipt_scope"] = (
                "observed/verified here means this server measured this canvas; it is not a live screen or "
                "camera capture, and the signing key ends with this chat's server."
            )
        assert_no_raw_buffer_leakage(body)
        return self._result(body, images=images)

    # ----- tools -------------------------------------------------------------------------------

    def handle_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args, dict):
            return self._error("tool arguments must be a JSON object", "invalid_arguments")
        try:
            if name == "eye.electric.status":
                sess = self.ensure_session()
                health = sess.get_enclave_health_and_audit_summary()
                stream_summary = (
                    self.stream.compute_temporal_window_summary() if self.stream is not None else None
                )
                body = {
                    "status": "ok",
                    "enclave_health": health,
                    "supported_resolutions": [list(r) for r in SUPPORTED_RESOLUTIONS],
                    "registered_sources": [
                        {"source_id": sid, "source_version": sver, "width": w, "height": h}
                        for (sid, sver), (w, h) in sorted(sess._source_dimensions.items())
                    ][:64],
                    "retention": {
                        "max_canvases": MAX_RETAINED_SOURCES,
                        "native_bytes_budget": NATIVE_BYTES_BUDGET,
                        "retired_canvases": self.retired_source_count,
                        "max_stream_frames_kept": MAX_RETAINED_FRAMES,
                    },
                    "continuous_stream": {
                        "initialized": self.stream is not None,
                        "captured_frames": len(self.stream_frames_meta),
                        "next_sequence": self.stream_sequence + 1,
                        "temporal_window_summary": stream_summary,
                    },
                    "summary": copy.deepcopy(self.last_summary),
                }
                body["summary"]["registered_sources"] = body["summary"].get("registered_sources", [])[-64:]
                assert_no_raw_buffer_leakage(body)
                return self._result(body)

            if name == "eye.electric.inspect":
                sess = self.ensure_session()
                preset = args.get("preset")
                marks = args.get("marks")
                if preset == "emblem":
                    if not marks:
                        marks = _load_emblem_marks()
                    default_w, default_h = 512, 512
                    default_sid = "src_cyclops_emblem"
                else:
                    default_w, default_h = 256, 256
                    default_sid = "src_agent_canvas"

                sid = args.get("source_id", default_sid)
                w = int(args.get("width", default_w))
                h = int(args.get("height", default_h))
                mirror = args.get("mirror", "none")
                origin = args.get("origin_attestation", "synthetic_fixture")
                issue_receipt = bool(args.get("issue_receipt", False))
                want_image = bool(args.get("image", False))

                if not marks:
                    sver = args.get("source_version") or self._latest_version(sess, sid, "v1")
                    if (sid, sver) not in sess._source_dimensions:
                        return self._error(
                            "eye.electric.inspect requires either 'marks', preset:'emblem', or a source_id/source_version that is still registered",
                            "invalid_arguments",
                        )
                    w, h = sess._source_dimensions[(sid, sver)]
                    reg = {"status": "already_registered", "source_id": sid, "source_version": sver, "resolution": [w, h]}
                else:
                    if preset == "emblem" and "marks" not in args and "source_version" not in args and (sid, "v1") in sess._source_dimensions:
                        sver = "v1"
                        w, h = sess._source_dimensions[(sid, sver)]
                        reg = {"status": "already_registered", "source_id": sid, "source_version": sver, "resolution": [w, h]}
                    else:
                        sver = args.get("source_version") or self._next_version(sess, sid)
                        source_kind = "captured_frame" if issue_receipt else ("image_bytes" if origin == "owner_supplied_picture" else "drawing_state")
                        reg = self._register(
                            sess, sid, sver, marks, width=w, height=h, mirror=mirror,
                            source_kind=source_kind, origin_attestation=origin,
                        )

                return self._inspect_reply(sess, sid, sver, reg, issue_receipt=issue_receipt, want_image=want_image,
                                           wanted=args.get("colours"), tolerance=args.get("tolerance"))

            if name == "eye.electric.diff":
                sess = self.ensure_session()
                sid_a = args.get("source_id_a", "src_agent_canvas")
                sid_b = args.get("source_id_b", sid_a)
                w = int(args.get("width", 256))
                h = int(args.get("height", 256))
                tolerance = int(args.get("tolerance", 0))
                want_image = bool(args.get("image", False))

                # Pick B's version first so registering A can never retire the canvas B names.
                sver_b_named = None if "after_marks" in args else args.get("source_version_b", "v2")
                if "before_marks" in args:
                    sver_a = args.get("source_version_a") or self._next_version(sess, sid_a)
                    self._register(
                        sess, sid_a, sver_a, args["before_marks"], width=w, height=h,
                        protect=[(sid_b, sver_b_named)] if sver_b_named else [],
                    )
                else:
                    sver_a = args.get("source_version_a", "v1")
                if "after_marks" in args:
                    sver_b = args.get("source_version_b") or self._next_version(sess, sid_b)
                    self._register(sess, sid_b, sver_b, args["after_marks"], width=w, height=h, protect=[(sid_a, sver_a)])
                else:
                    sver_b = args.get("source_version_b", "v2")

                diff = sess.compare_canvases((sid_a, sver_a), (sid_b, sver_b), tolerance=tolerance)
                dirty_tiles = sess._multi_res_vault["compute_dirty_tiles_16x16"]((sid_a, sver_a), (sid_b, sver_b))

                images = []
                image_meta = None
                if want_image and (sid_a, sver_a) in self.compiled_specs_cache and (sid_b, sver_b) in self.compiled_specs_cache:
                    sp_a, wa, ha = self.compiled_specs_cache[(sid_a, sver_a)]
                    sp_b, _, _ = self.compiled_specs_cache[(sid_b, sver_b)]
                    png_bytes, png_sha = _render_diff_panel_png(sp_a, sp_b, wa, ha)
                    images.append(png_bytes)
                    image_meta = {"mimeType": "image/png", "sha256": png_sha, "byte_length": len(png_bytes)}
                elif want_image:
                    png_a, png_b = self._picture_png((sid_a, sver_a)), self._picture_png((sid_b, sver_b))
                    if png_a is not None and png_b is not None:
                        with Image.open(io.BytesIO(png_a)) as img_a, Image.open(io.BytesIO(png_b)) as img_b:
                            png_bytes, png_sha = _render_diff_panel_from_images(img_a.convert("RGBA"), img_b.convert("RGBA"))
                        images.append(png_bytes)
                        image_meta = {"mimeType": "image/png", "sha256": png_sha, "byte_length": len(png_bytes)}

                body = {
                    **diff,
                    "dirty_tiles_16x16": dirty_tiles,
                    "epistemic_style": UNCERTAINTY_STYLE_SPEC["derived"],
                    **({"rendered_diff_panel": image_meta} if image_meta else {}),
                }
                assert_no_raw_buffer_leakage(body)
                return self._result(body, images=images)

            if name == "eye.electric.ghost":
                sess = self.ensure_session()
                base_sid = args.get("base_source_id", "src_agent_canvas")
                w = int(args.get("width", 256))
                h = int(args.get("height", 256))
                mirror = args.get("mirror", "none")
                want_image = bool(args.get("image", False))

                if "base_marks" in args:
                    base_sver = args.get("base_source_version") or self._next_version(sess, base_sid)
                    self._register(sess, base_sid, base_sver, args["base_marks"], width=w, height=h, mirror=mirror)
                else:
                    base_sver = args.get("base_source_version") or self._latest_version(sess, base_sid, "v1")
                    if (base_sid, base_sver) not in sess._source_dimensions:
                        if "base_source_version" in args:
                            return self._error(
                                f"{base_sid}@{base_sver} is not registered (it may have retired); pass base_marks",
                                "invalid_arguments",
                            )
                        base_sver = self._next_version(sess, base_sid)
                        default_base = [{"k": "dot", "x": 0.5, "y": 0.5, "r": 0.08, "c": [0, 229, 255], "a": 1.0}]
                        self._register(sess, base_sid, base_sver, default_base, width=w, height=h, mirror=mirror)

                ghost_marks = args["ghost_marks"]
                preview = sess.preview_ghost_marks(base_sid, base_sver, ghost_marks, mirror=mirror)

                images = []
                image_meta = None
                if want_image and (base_sid, base_sver) in self.compiled_specs_cache:
                    base_specs, bw, bh = self.compiled_specs_cache[(base_sid, base_sver)]
                    ghost_specs, _ = normalize_marks_for_canvas(ghost_marks, bw, bh, mirror=mirror)
                    png_bytes, png_sha = _render_specs_to_png(base_specs, bw, bh, ghost_specs=ghost_specs)
                    images.append(png_bytes)
                    image_meta = {"mimeType": "image/png", "sha256": png_sha, "byte_length": len(png_bytes)}

                body = {
                    **preview,
                    "hypothetical_metrics": _compact_simd_metrics(preview["hypothetical_metrics"]),
                    "epistemic_style": UNCERTAINTY_STYLE_SPEC["predicted"],
                    **({"rendered_ghost_preview": image_meta} if image_meta else {}),
                }
                assert_no_raw_buffer_leakage(body)
                return self._result(body, images=images)

            if name == "eye.electric.stream":
                return self._handle_stream(args)

            if name == "eye.electric.look":
                return self._handle_look(args)

            if name == "eye.electric.refuse":
                sess = self.ensure_session()
                query_type = str(args["query_type"])
                lowered = query_type.lower()
                if (
                    query_type not in CLOSED_FORM_OPERATORS
                    and query_type not in PROHIBITED_SEMANTIC_QUERY_TYPES
                    and not any(hint in lowered for hint in SEMANTIC_QUERY_HINTS)
                ):
                    # An unknown measurement name is a typo, not a request for interpretation.
                    wanted = lowered.replace("-", "_")
                    words = {w.rstrip("s") for w in wanted.split("_")}

                    def _closeness(op: str) -> Tuple[int, float]:
                        shared = len(words & {w.rstrip("s") for w in op.split("_")} - {"measure"})
                        return shared, difflib.SequenceMatcher(None, wanted, op).ratio()

                    ranked = sorted(CLOSED_FORM_OPERATORS, key=_closeness, reverse=True)
                    close = [op for op in ranked if _closeness(op)[0] > 0 or _closeness(op)[1] >= 0.6][:2]
                    hint = f" Did you mean {' or '.join(close)}?" if close else ""
                    return self._error(
                        f"Unknown measurement {query_type!r}.{hint} Valid query_type values: {', '.join(sorted(CLOSED_FORM_OPERATORS))}.",
                        "invalid_arguments",
                    )
                sid = args.get("source_id", "src_epistemic_probe")
                sver = args.get("source_version") or self._latest_version(sess, sid, "v1")
                params = args.get("parameters", {})
                if (sid, sver) not in sess._source_dimensions:
                    if "source_id" in args or "source_version" in args:
                        return self._error(f"{sid}@{sver} is not registered (it may have retired)", "invalid_arguments")
                    # A small probe canvas lets refusal behaviour be checked before anything is drawn.
                    sver = self._next_version(sess, sid)
                    self._register(
                        sess, sid, sver,
                        [{"k": "dot", "x": 0.5, "y": 0.5, "r": 0.1, "c": [0, 229, 255], "a": 1.0}],
                        width=256, height=256,
                    )
                res = sess.evaluate_agent_query(query_type, params, source_id=sid, source_version=sver)
                w, h = sess._source_dimensions[(sid, sver)]
                body = {
                    "query_type": query_type,
                    "source_id": sid,
                    "source_version": sver,
                    "evaluation": res,
                    "measured_on": {
                        "width": 128,
                        "height": 128,
                        "native_width": w,
                        "native_height": h,
                        "note": "Coordinates and pixel counts are on the 128x128 downsample of the native canvas.",
                    },
                    "epistemic_style": UNCERTAINTY_STYLE_SPEC["unknown" if res.get("status") == "refused" else "derived"],
                }
                assert_no_raw_buffer_leakage(body)
                return self._result(body)

            return self._error(f"Unknown Electric Vision tool: {name}", "unknown_tool")
        except PermissionError as exc:
            return self._error(str(exc), "epistemic_or_quarantine_refused")
        except (ValueError, KeyError, TypeError, OverflowError, IndexError) as exc:
            return self._error(str(exc), "invalid_arguments")
        except (OSError, RuntimeError) as exc:
            # Missing or unreadable files and renderer failures are the request's problem, not the server's.
            return self._error(str(exc), "visual_source_unavailable")
        except Exception as exc:
            return self._error(f"Electric Vision execution error: {exc}", "internal_error")

    def _handle_stream(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if args.get("action") != "push":
            return self._handle_stream_request(args)
        tick_before = self.stream._tick_counter if self.stream is not None else 0
        try:
            return self._handle_stream_request(args)
        except BaseException as exc:
            if self.stream is not None:
                self.stream.invalidate_failure(exc, tick_before=tick_before)
            self.stream_frames_meta.clear()
            self.last_summary.update(phase="failed", stream_frames=0, stream_regime="failed_closed")
            try:
                self._cleanup_frame_files()
            except BaseException as cleanup_exc:
                exc.add_note(f"Frame file retirement pending: {cleanup_exc}")
            try:
                self._save_status(strict=True)
            except BaseException as status_exc:
                exc.add_note(f"Failed-stream status publication failed: {status_exc}")
            raise

    def _handle_stream_request(self, args: Dict[str, Any]) -> Dict[str, Any]:
        action = args["action"]
        want_image = bool(args.get("image", False))

        if action == "reset":
            released = self._reset_stream()
            self.last_summary.update({"stream_frames": 0, "stream_regime": "reset"})
            self._save_status()
            return self._result({
                "status": "ok",
                "action": "reset",
                "stream_frames": 0,
                "released_frames": released,
                "next_sequence": self.stream_sequence + 1,
            })

        if action in ("status", "packet") and self.stream is None:
            return self._result({
                "status": "ok",
                "action": action,
                "initialized": False,
                "captured_frames": 0,
                "next_sequence": self.stream_sequence + 1,
                **({"packet": None, "total_stream_frames": 0} if action == "packet" else {}),
            })

        if action == "push":
            sess = self.ensure_session()
            wanted = {
                "width": int(args.get("width", self.stream.width if self.stream else 256)),
                "height": int(args.get("height", self.stream.height if self.stream else 256)),
                "window_size": int(args.get("window_size", self.stream.max_window_frames if self.stream else 6)),
                "settle_ticks": int(args.get("settle_ticks", self.stream.settle_ticks if self.stream else 2)),
            }
            if self.stream is None:
                self.stream = sess.create_continuous_stream(
                    source_id=STREAM_SOURCE_ID,
                    width=wanted["width"],
                    height=wanted["height"],
                    max_window_frames=wanted["window_size"],
                    settle_ticks=wanted["settle_ticks"],
                )
            else:
                current = {
                    "width": self.stream.width,
                    "height": self.stream.height,
                    "window_size": self.stream.max_window_frames,
                    "settle_ticks": self.stream.settle_ticks,
                }
                if wanted != current:
                    return self._error(
                        f"The stream is running with {current}; call action:'reset' before pushing with {wanted}.",
                        "invalid_arguments",
                    )

        stream = self.stream
        assert stream is not None
        w, h = stream.width, stream.height

        if action == "status":
            summary = stream.compute_temporal_window_summary()
            return self._result({
                "status": "ok",
                "action": "status",
                "initialized": True,
                "resolution": {"width": w, "height": h},
                "captured_frames": len(self.stream_frames_meta),
                "next_sequence": self.stream_sequence + 1,
                "temporal_window_summary": summary,
            })

        if action == "push":
            issue_receipts = bool(args.get("issue_receipts", False))
            raw_frames: List[List[Dict[str, Any]]] = []
            if "frames" in args and args["frames"]:
                raw_frames.extend(args["frames"])
            elif "marks" in args and args["marks"]:
                raw_frames.append(args["marks"])
            else:
                return self._error("eye.electric.stream action='push' requires 'frames' or 'marks'", "invalid_arguments")
            # Check every frame before taking any, so a bad frame never leaves half a push behind.
            for idx, frame_marks in enumerate(raw_frames):
                if isinstance(frame_marks, _PixelFrame):
                    continue
                try:
                    normalize_marks_for_canvas(frame_marks, w, h)
                except ValueError as exc:
                    return self._error(f"frame {idx}: {exc}; nothing was pushed", "invalid_arguments")
            if self.stream_sequence + len(raw_frames) > MAX_SEQUENCE:
                return self._error("the stream has used every sequence number; start a new chat", "invalid_arguments")

            staged_ghost_id: Optional[str] = None
            if "stage_ghost_marks" in args and args["stage_ghost_marks"] and stream._window_versions:
                base_ver = stream._window_versions[-1]
                staged = stream.stage_ghost_prediction(base_ver, args["stage_ghost_marks"])
                staged_ghost_id = staged["ghost_id"]

            started = time.monotonic()
            tick_reports: List[Dict[str, Any]] = []
            frames_meta = list(self.stream_frames_meta)
            sequence = self.stream_sequence
            for idx, frame_marks in enumerate(raw_frames):
                if idx > 0 and time.monotonic() - started > PUSH_TIME_BUDGET_S:
                    break
                seq = sequence + 1
                while self._taken(sess, (stream.source_id, f"v{seq}")):
                    seq += 1
                ver = f"v{seq}"
                verify_id = staged_ghost_id if (idx == 0 and staged_ghost_id is not None) else None
                pixel = frame_marks if isinstance(frame_marks, _PixelFrame) else None
                report = stream.ingest_frame(
                    ver,
                    [] if pixel else frame_marks,
                    source_kind="captured_frame" if issue_receipts else ("image_bytes" if pixel else "drawing_state"),
                    issue_observation_receipt=issue_receipts,
                    verify_staged_ghost_id=verify_id,
                    rgba=pixel.rgba if pixel else None,
                    origin=pixel.origin if pixel else None,
                )
                sequence = seq
                if pixel:
                    rgba_bytes = pixel.rgba
                    png_bytes = pixel.png if want_image else b""
                else:
                    specs, _ = normalize_marks_for_canvas(frame_marks, w, h)
                    rgba_bytes = _rasterize_normalized_specs(specs, w, h)
                    png_bytes = _render_specs_to_png(specs, w, h)[0] if want_image else b""
                png_sha = hashlib.sha256(png_bytes).hexdigest() if png_bytes else None
                frame_path = self.frames_dir / f"frame-{seq:06d}.png"
                rgba_path = frame_path.with_suffix(".rgba")
                self._pending_frame_retirements.update((frame_path, rgba_path))
                rgba_path.write_bytes(rgba_bytes)
                os.chmod(rgba_path, 0o600)
                if png_bytes:
                    frame_path.write_bytes(png_bytes)
                    os.chmod(frame_path, 0o600)
                frames_meta.append({
                    "sequence": seq,
                    "frame_id": uuid.uuid4().hex,
                    "source_version": ver,
                    "image_path": str(frame_path),
                    "rgba_path": str(rgba_path),
                    "width": w, "height": h,
                    "sha256": png_sha,
                    "dhash64": report["dhash64"],
                    "capture_time": round(time.monotonic(), 4),
                    **({"origin": pixel.origin} if pixel else {}),
                })
                if pixel and "drawn_mean_rgb_0_255" in pixel.origin:  # the frame's own drawn colour, for its blank check
                    report["drawn_mean_rgb_0_255"] = pixel.origin["drawn_mean_rgb_0_255"]
                tick_reports.append(report)
            self._retire_old_frames(frames_meta)

            latest_summary = tick_reports[-1]["temporal_window_summary"]
            next_summary = {
                **self.last_summary,
                "phase": "streaming",
                "stream_frames": len(frames_meta),
                "stream_regime": latest_summary["settling_and_oscillation_detector"]["regime"],
                "last_dhash64": tick_reports[-1]["dhash64"],
            }

            images = []
            if want_image and frames_meta:
                for frame in frames_meta[-min(8, len(frames_meta)):]:
                    self._ensure_frame_png(frame)
                panel_bytes, _ = _render_temporal_contact_sheet_png(frames_meta[-min(8, len(frames_meta)):])
                detail_bytes = Path(frames_meta[-1]["image_path"]).read_bytes()
                images.extend([panel_bytes, detail_bytes])

            ticks_compact = [
                {
                    "tick": r["tick"],
                    "source_version": r["source_version"],
                    "dhash64": r["dhash64"],
                    "native_pixel_sha256": r["native_pixel_sha256"],
                    "observed": r["observation_state"]["observed"],
                    "uncertainty_state": r["observation_state"]["uncertainty_state"],
                    "decayed_to_stale_versions": r["decayed_to_stale_versions"],
                    "changed_pixels": (
                        r["transition_from_previous"]["changed_pixels"]
                        if r["transition_from_previous"]
                        else 0
                    ),
                    "dirty_tile_count": (
                        r["transition_from_previous"]["dirty_tiles_16x16"]["dirty_tile_count"]
                        if r["transition_from_previous"]
                        else 0
                    ),
                    "ghost_realization_audit": r["ghost_realization_audit"],
                    **({"drawn_mean_rgb_0_255": r["drawn_mean_rgb_0_255"]} if "drawn_mean_rgb_0_255" in r else {}),
                }
                for r in tick_reports
            ]
            latest_tick_compact = {
                k: v for k, v in tick_reports[-1].items() if k != "temporal_window_summary"
            }
            ghost_audits = [r["ghost_realization_audit"] for r in tick_reports if r.get("ghost_realization_audit")]
            not_ingested = len(raw_frames) - len(tick_reports)
            body = {
                "status": "ok",
                "action": "push",
                "ingested_count": len(tick_reports),
                "total_stream_frames": len(frames_meta),
                "next_sequence": sequence + 1,
                "ticks": ticks_compact,
                "latest_tick": latest_tick_compact,
                "temporal_window_summary": latest_summary,
                **({"ghost_realization_audits": ghost_audits} if ghost_audits else {}),
                **({
                    "not_ingested_count": not_ingested,
                    "note": f"Stopped after {PUSH_TIME_BUDGET_S:g}s; push the last {not_ingested} frame(s) again.",
                } if not_ingested else {}),
            }
            assert_no_raw_buffer_leakage(body)
            reply = self._result(body, images=images)
            for frame in frames_meta:
                self._pending_frame_retirements.discard(Path(frame["image_path"]))
                self._pending_frame_retirements.discard(Path(frame["rgba_path"]))
            self._save_status(summary=next_summary, strict=True)
            self.stream_frames_meta = frames_meta
            self.stream_sequence = sequence
            self.last_summary = next_summary
            return reply

        if action == "packet":
            after_seq = int(args.get("after_sequence", 0))
            max_frames = int(args.get("max_frames", 12))
            selected = [f for f in self.stream_frames_meta if f["sequence"] > after_seq][:max_frames]
            oldest_kept = self.stream_frames_meta[0]["sequence"] if self.stream_frames_meta else None
            if not selected:
                return self._result({
                    "status": "ok",
                    "action": "packet",
                    "packet": None,
                    "total_stream_frames": len(self.stream_frames_meta),
                    "oldest_retained_sequence": oldest_kept,
                    "next_sequence": self.stream_sequence + 1,
                })
            detail = self.stream_frames_meta[-1]
            images, panel_sha = [], None
            if want_image:
                for frame in selected + [detail]:
                    self._ensure_frame_png(frame)
                panel_bytes, panel_sha = _render_temporal_contact_sheet_png(selected)
                detail_bytes = Path(detail["image_path"]).read_bytes()
                images = [panel_bytes, detail_bytes]
            public_frames = [
                {k: v for k, v in f.items() if k not in ("image_path", "rgba_path")} for f in selected
            ]
            packet = {
                "packet_id": uuid.uuid4().hex,
                "panel_kind": "temporal_overview",
                "frame_count": len(selected),
                "sequence_start": selected[0]["sequence"],
                "sequence_end": selected[-1]["sequence"],
                "next_sequence": selected[-1]["sequence"],
                "oldest_retained_sequence": oldest_kept,
                "frames": public_frames,
                "overview_png_sha256": panel_sha,
                "detail_sequence": detail["sequence"],
                "detail_source_version": detail["source_version"],
                "detail_png_sha256": detail["sha256"],
                "temporal_window_summary": stream.compute_temporal_window_summary(),
            }
            assert_no_raw_buffer_leakage(packet)
            return self._result({"status": "ok", "action": "packet", "packet": packet}, images=images)

        return self._error(f"Unknown stream action: {action!r}", "invalid_arguments")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="electric_vision_mcp")
    parser.add_argument("--electric-root", required=True)
    parser.add_argument("--agent-slot", default="agent-1")
    parser.add_argument("--read-snapshot", action="store_true")
    parsed = parser.parse_args(argv)

    root = Path(parsed.electric_root)
    if parsed.read_snapshot:
        default_idle = json.dumps({"phase": "idle", "registered_sources": [], "stream_frames": 0}) + "\n"
        try:
            st_root = os.lstat(root)
            if stat.S_ISLNK(st_root.st_mode) or not stat.S_ISDIR(st_root.st_mode) or st_root.st_uid != os.getuid():
                sys.stdout.write(default_idle)
                return 0
            status_file = root / "status.json"
            st_file = os.lstat(status_file)
            if stat.S_ISLNK(st_file.st_mode) or not stat.S_ISREG(st_file.st_mode) or st_file.st_uid != os.getuid():
                sys.stdout.write(default_idle)
                return 0
            data = json.loads(status_file.read_text(encoding="utf-8"))
            sys.stdout.write(json.dumps(data) + "\n")
            return 0
        except FileNotFoundError:
            sys.stdout.write(default_idle)
            return 0

    server = ElectricVisionServer(root, parsed.agent_slot)
    main_pid = os.getpid()

    def _handle_term(signum: int, _frame: Any) -> None:
        if os.getpid() == main_pid:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)
    initialized = False

    def _reply(value: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(value) + "\n")
        sys.stdout.flush()

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception:
                _reply({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            if not isinstance(req, dict):
                _reply({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: expected a JSON object"}})
                continue
            if "id" not in req:
                continue

            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params") if req.get("params") is not None else {}
            if not isinstance(params, dict):
                _reply({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: expected an object"}})
                continue

            try:
                if method == "initialize":
                    initialized = True
                    reply = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": params.get("protocolVersion", "2025-11-25"),
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "CyclopsElectricVision", "version": "0.2.0"},
                        },
                    }
                elif not initialized:
                    reply = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32002, "message": "Not initialized"}}
                elif method == "tools/list":
                    reply = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}
                elif method == "tools/call":
                    t_name = params.get("name", "")
                    t_args = params.get("arguments") if params.get("arguments") is not None else {}
                    res = server.handle_tool(str(t_name), t_args)
                    reply = {"jsonrpc": "2.0", "id": req_id, "result": res}
                else:
                    reply = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
            except Exception as exc:
                reply = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": f"Internal error: {exc}"}}
            _reply(reply)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        sys.stderr.write(f"Electric Vision startup refused: {err}\n")
        sys.exit(1)
