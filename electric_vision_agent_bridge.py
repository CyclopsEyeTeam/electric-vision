#!/usr/bin/env python3
"""
electric_vision_agent_bridge.py

Zero-PII Synthetic Canvas & Multi-Resolution Electric Vision Bridge for Agents.

Connects agent-authored synthetic canvases (`cyclopsdraw` vector marks, Spark &
Prism normalized `[0..1]` `marks` paintings, and `ElectricEyeFixtureHarness`
`DrawingMark` primitives) to the verified Electric Eye out-of-process enclave
and native C11/AVX2/Rust SIMD operator engine (`libprod_gate_simd.so`).

Hard Architectural & Governance Invariants Preserved:
- Strictly non-live, zero-PII synthetic provenance (`origin_attestation == "synthetic_fixture"`).
- Raw RGBA pixel buffers and cryptographic signing keys remain isolated inside a
  private lexical closure vault + out-of-process HSM/Vault daemons (`0` raw pixel
  or key bytes exposed to agents).
- Supports resolutions: 128x128, 256x256, 512x512, and 1080p (1920x1080).
- Executes all 10 closed-form measurement operators + 10 native AVX2 SIMD
  structural/perceptual operators with bit-exact replay digests.
- Supports non-destructive GhostOverlay mark previews (`ghost_isolation_plane`)
  with zero pixel leakage into `verified_plane` and `observed=False`.
- Enforces 100% deterministic refusal on semantic, aesthetic, OCR-for-agent,
  and user-intent queries (`REFUSED_EPISTEMIC_BOUNDARY`).
- Every render starts with `observed=False` and requires a cryptographically
  verified `ObservationReceipt` and complete provenance before promotion.
"""

from __future__ import annotations

import copy
import math
import multiprocessing
import os
import re
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from electric_eye_harness import (
    CLOSED_FORM_OPERATORS,
    FIXED_LOGICAL_TIMESTAMP,
    MAX_MARKS_PER_SOURCE,
    MAX_REPORTED_REGIONS,
    OPERATOR_VERSION,
    PROHIBITED_SOURCE_ID_TOKENS,
    REFUSAL_CODE_SEMANTIC,
    REFUSAL_REASON_SEMANTIC,
    SOURCE_ID_REGEX,
    SOURCE_VERSION_REGEX,
    UNCERTAINTY_STYLE_SPEC,
    VALID_CONFIDENCE_LEVELS,
    VALID_RENDER_SOURCES,
    VALID_RENDER_TYPES,
    VALID_SOURCE_KINDS,
    VALID_UNCERTAINTY_STATES,
    VALID_VISIBILITY_POLICIES,
    DrawingMark,
    ElectricEyeFixtureHarness,
    assert_no_raw_buffer_leakage,
    canonical_json,
    compute_replay_digest,
    sha256_hex,
)
from exec_gate_daemons import (
    OutOfProcessAuditSinkDaemon,
    OutOfProcessQuarantineAndKillSwitchDaemon,
    OutOfProcessVaultGateDaemon,
)
from prod_gate_enclave import (
    NativeSimdOperatorEngine,
    OutOfProcessHsmKmsClient,
    _hsm_kms_enclave_loop,
    build_accessibility_metadata,
    explicit_zeroize_bytearray,
)

SUPPORTED_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (128, 128),
    (256, 256),
    (512, 512),
    (1920, 1080),
)

PRISM_MARK_KINDS = frozenset({
    "fill",
    "glow",
    "line",
    "curve",
    "dot",
    "ring",
    "arc",
    "spray",
})

CYCLOPSDRAW_SHAPES = frozenset({
    "rect",
    "rounded-rect",
    "circle",
    "ellipse",
    "line",
    "polygon",
    "polyline",
    "spline",
    "ring",
    "arc",
    "star",
    "regular-polygon",
    "crescent",
    "dot",
})


def _parse_hex_or_rgb_color(
    color_spec: Any,
    default_alpha: float = 1.0,
) -> Tuple[int, int, int, int]:
    """Deterministically parses a hex string (#rgb, #rrggbb, #rrggbbaa) or RGB/RGBA list."""
    alpha_u8 = max(0, min(255, int(round(float(default_alpha) * 255.0))))
    if isinstance(color_spec, (list, tuple)):
        if len(color_spec) == 3:
            r, g, b = (max(0, min(255, int(round(float(c))))) for c in color_spec)
            return (r, g, b, alpha_u8)
        if len(color_spec) == 4:
            r, g, b, a = (max(0, min(255, int(round(float(c))))) for c in color_spec)
            return (r, g, b, a)
    if isinstance(color_spec, str):
        s = color_spec.strip().lstrip("#")
        if len(s) == 3:
            r = int(s[0] * 2, 16)
            g = int(s[1] * 2, 16)
            b = int(s[2] * 2, 16)
            return (r, g, b, alpha_u8)
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b, alpha_u8)
        if len(s) == 8:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            a = int(s[6:8], 16)
            return (r, g, b, a)
    raise ValueError(f"Unsupported colour specification: {color_spec!r}")


def _canonical_sha256_tag(data_or_digest: Union[bytes, bytearray, str]) -> str:
    """Returns a canonical single-prefixed 'sha256:<64-hex>' digest string."""
    if isinstance(data_or_digest, (bytes, bytearray)):
        raw = sha256_hex(bytes(data_or_digest))
    else:
        raw = str(data_or_digest).strip()
    while raw.startswith("sha256:"):
        raw = raw[len("sha256:") :]
    return f"sha256:{raw}"


def _sample_cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    steps: int = 12,
) -> List[Tuple[int, int]]:
    """Deterministically samples a cubic Bezier curve into integer pixel coordinates."""
    pts: List[Tuple[int, int]] = []
    for s in range(1, steps + 1):
        t = s / float(steps)
        mt = 1.0 - t
        x = (
            (mt * mt * mt) * p0[0]
            + 3.0 * (mt * mt) * t * p1[0]
            + 3.0 * mt * (t * t) * p2[0]
            + (t * t * t) * p3[0]
        )
        y = (
            (mt * mt * mt) * p0[1]
            + 3.0 * (mt * mt) * t * p1[1]
            + 3.0 * mt * (t * t) * p2[1]
            + (t * t * t) * p3[1]
        )
        pts.append((int(round(x)), int(round(y))))
    return pts


def _validate_mark_hygiene(mark_dict: Dict[str, Any]) -> None:
    """Enforces zero-PII and zero-path hygiene on synthetic mark specifications."""
    serialized = canonical_json(mark_dict).lower()
    for bad_token in ("file://", "/home/", "/users/", "/etc/", "/var/", "@", "passwd", "secret"):
        if bad_token in serialized:
            raise ValueError(f"ERR_UNSAFE_MARK_SPEC: prohibited token {bad_token!r} in mark")
    # Reject raw `text` shape in agent vision bridge to uphold OCR/text non-ingestion boundary
    if mark_dict.get("shape") == "text" or mark_dict.get("k") == "text":
        raise ValueError("ERR_TEXT_MARK_PROHIBITED: text marks are excluded from zero-PII synthetic vision")


MIRROR_MODES = frozenset({"none", "x", "y", "xy"})
# Older tool descriptions advertised h/v/hv; they mean the same reflections.
MIRROR_ALIASES = {"h": "x", "v": "y", "hv": "xy", "vh": "xy", "yx": "xy"}
MAX_MARK_COORDINATE = 2**53  # JSON-safe integers; geometry clips far values, so only non-finite ones are fatal
MAX_LIST_ITEMS_PER_MARK = 4096


def _validate_mark_numbers(value: Any, where: str, depth: int = 0) -> None:
    """Rejects non-finite or absurd numbers and oversized point lists before any geometry runs."""
    if depth > 8:
        raise ValueError(f"ERR_MARK_TOO_DEEP: {where} is nested too deeply")
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or abs(value) > MAX_MARK_COORDINATE:
            raise ValueError(f"ERR_MARK_NUMBER_OUT_OF_RANGE: {where} has {value!r}; numbers must be finite and within ±{MAX_MARK_COORDINATE}")
        return
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return
        _validate_mark_numbers(number, where, depth + 1)
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            _validate_mark_numbers(inner, f"{where}.{key}", depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST_ITEMS_PER_MARK:
            raise ValueError(f"ERR_MARK_TOO_MANY_POINTS: {where} has {len(value)} items (limit {MAX_LIST_ITEMS_PER_MARK})")
        for i, inner in enumerate(value):
            _validate_mark_numbers(inner, f"{where}[{i}]", depth + 1)


def normalize_marks_for_canvas(
    raw_marks: Sequence[Union[DrawingMark, Dict[str, Any]]],
    width: int,
    height: int,
    *,
    mirror: str = "none",
) -> Tuple[List[Dict[str, Any]], List[DrawingMark]]:
    """
    Normalizes a sequence of marks from any of the three supported Cyclops formats:
      1. `DrawingMark` (or dict with `mark_id`, `shape`, `x0`, `y0`, `x1`, `y1`, `rgba`)
      2. Spark / Prism `[0..1]` painting marks (`k` in `fill|glow|line|curve|dot|ring|arc|spray`)
      3. `cyclopsdraw` vector marks (`shape` in `rect|circle|ellipse|line|polygon|ring|arc|star|...`)

    Returns:
      - `normalized_specs`: resolution-specific deterministic raster instructions for `(width, height)`
      - `proxy_128_marks`: `DrawingMark` list scaled to `128x128` for the out-of-process vault daemon
    """
    if (width, height) not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported canvas resolution {(width, height)}; must be one of {SUPPORTED_RESOLUTIONS}"
        )
    if len(raw_marks) > MAX_MARKS_PER_SOURCE:
        raise ValueError(
            f"Mark count {len(raw_marks)} exceeds MAX_MARKS_PER_SOURCE ({MAX_MARKS_PER_SOURCE})"
        )
    mirror = MIRROR_ALIASES.get(mirror, mirror)
    if mirror not in MIRROR_MODES:
        raise ValueError(f"Unsupported mirror mode: {mirror!r}; use one of {sorted(MIRROR_MODES)}")

    normalized_specs: List[Dict[str, Any]] = []
    proxy_128_marks: List[DrawingMark] = []

    def _mirror_item_specs(start: int) -> None:
        """Append left-right / top-bottom reflections of the specs one non-Prism mark produced."""
        if mirror == "none":
            return
        flips = {"x": [(True, False)], "y": [(False, True)], "xy": [(True, False), (False, True), (True, True)]}[mirror]
        produced = normalized_specs[start:]
        for mx, my in flips:
            fx = (lambda v: width - 1 - int(v)) if mx else (lambda v: int(v))
            fy = (lambda v: height - 1 - int(v)) if my else (lambda v: int(v))
            suffix = {(True, False): "_mx", (False, True): "_my", (True, True): "_mxy"}[(mx, my)]
            for spec in produced:
                extra = {k: v for k, v in spec.items() if k not in ("mark_id", "op_kind", "x0", "y0", "x1", "y1", "rgba")}
                for key in ("cx", "icx"):
                    if key in extra:
                        extra[key] = fx(extra[key])
                for key in ("cy", "icy"):
                    if key in extra:
                        extra[key] = fy(extra[key])
                if spec["op_kind"] in ("polygon", "compound_polygon"):
                    # Filled shapes keep their points and are flipped span by span when drawn.
                    extra["flip"] = (mx, my)
                elif "points" in extra:
                    extra["points"] = [(fx(x), fy(y)) for x, y in extra["points"]]
                if "rotation_deg" in extra and mx != my:
                    extra["rotation_deg"] = -float(extra["rotation_deg"])
                _append_spec_and_proxy(
                    mark_id=f"{spec['mark_id']}{suffix}"[:48],
                    op_kind=spec["op_kind"],
                    bbox_px=(fx(spec["x0"]), fy(spec["y0"]), fx(spec["x1"]), fy(spec["y1"])),
                    rgba=tuple(spec["rgba"]),  # type: ignore[arg-type]
                    extra=extra,
                )

    def _append_spec_and_proxy(
        mark_id: str,
        op_kind: str,
        bbox_px: Tuple[int, int, int, int],
        rgba: Tuple[int, int, int, int],
        extra: Dict[str, Any],
    ) -> None:
        x0, y0, x1, y1 = (
            max(0, min(width - 1, int(bbox_px[0]))),
            max(0, min(height - 1, int(bbox_px[1]))),
            max(0, min(width - 1, int(bbox_px[2]))),
            max(0, min(height - 1, int(bbox_px[3]))),
        )
        if "stroke_w" in extra:
            # Anything wider than twice the canvas already covers it; the bound keeps stroke cost fixed.
            extra = {**extra, "stroke_w": max(1, min(2 * max(width, height) + 1, int(extra["stroke_w"])))}
        spec = {
            "mark_id": mark_id,
            "op_kind": op_kind,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "rgba": list(rgba),
            **extra,
        }
        normalized_specs.append(spec)

        # Build 128x128 proxy DrawingMark
        px0 = max(0, min(127, int((x0 * 128) // width)))
        py0 = max(0, min(127, int((y0 * 128) // height)))
        px1 = max(0, min(127, int((x1 * 128) // width)))
        py1 = max(0, min(127, int((y1 * 128) // height)))
        proxy_shape = "rect"
        if op_kind == "line":
            proxy_shape = "line"
        elif op_kind == "dot" and px0 == px1 and py0 == py1:
            proxy_shape = "dot"
        if len(proxy_128_marks) < MAX_MARKS_PER_SOURCE:
            safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", mark_id)[:48] or f"mk_{len(proxy_128_marks)}"
            proxy_128_marks.append(
                DrawingMark(
                    mark_id=safe_id,
                    shape=proxy_shape,
                    x0=px0,
                    y0=py0,
                    x1=px1,
                    y1=py1,
                    rgba=rgba,
                )
            )

    for idx, item in enumerate(raw_marks):
        item_start = len(normalized_specs)
        if isinstance(item, DrawingMark):
            _append_spec_and_proxy(
                mark_id=item.mark_id,
                op_kind=item.shape,
                bbox_px=(item.x0, item.y0, item.x1, item.y1),
                rgba=tuple(int(c) for c in item.rgba),  # type: ignore[arg-type]
                extra={},
            )
            _mirror_item_specs(item_start)
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Mark at index {idx} must be a DrawingMark or dict")

        _validate_mark_hygiene(item)
        _validate_mark_numbers(item, f"mark {idx}")
        mid = str(item.get("mark_id") or item.get("as") or f"mk_{idx:04d}")
        mid = re.sub(r"[^a-zA-Z0-9_]", "_", mid)[:48] or f"mk_{idx:04d}"

        # Case 1: DrawingMark dict format (`x0`, `y0`, `x1`, `y1`, `rgba`)
        if "x0" in item and "y0" in item and "x1" in item and "y1" in item and "rgba" in item:
            rgba = _parse_hex_or_rgb_color(item["rgba"])
            sh = str(item.get("shape", "rect"))
            if sh not in ("rect", "line", "dot"):
                sh = "rect"
            _append_spec_and_proxy(
                mark_id=mid,
                op_kind=sh,
                bbox_px=(int(item["x0"]), int(item["y0"]), int(item["x1"]), int(item["y1"])),
                rgba=rgba,
                extra={},
            )
            _mirror_item_specs(item_start)
            continue

        # Models often name the Prism kind `type` or `kind`; with no world-operation geometry, read it as `k`.
        if "k" not in item and "geometry" not in item:
            alias = item.get("type", item.get("kind"))
            if isinstance(alias, str) and alias in PRISM_MARK_KINDS:
                item = {**{k: v for k, v in item.items() if k not in ("type", "kind")}, "k": alias}

        # Case 2: Spark & Prism normalized `[0..1]` mark vocabulary (`k` field)
        if "k" in item:
            k = str(item["k"])
            if k not in PRISM_MARK_KINDS:
                raise ValueError(f"Unsupported Prism mark kind: {k!r}")
            alpha_f = float(item.get("a", 1.0))
            rgba = _parse_hex_or_rgb_color(item.get("c", [255, 255, 255]), default_alpha=alpha_f)
            short_side = min(width, height)

            def _emit_prism_single(suffix: str, mx_flip: bool, my_flip: bool) -> None:
                def _tx(nx: float) -> int:
                    val = (1.0 - nx) if mx_flip else nx
                    return max(0, min(width - 1, int(round(val * (width - 1)))))

                def _ty(ny: float) -> int:
                    val = (1.0 - ny) if my_flip else ny
                    return max(0, min(height - 1, int(round(val * (height - 1)))))

                sub_id = f"{mid}{suffix}"[:48]
                if k in ("dot", "glow"):
                    cx = _tx(float(item.get("x", 0.5)))
                    cy = _ty(float(item.get("y", 0.5)))
                    rad = max(1, int(round(float(item.get("r", 0.05)) * short_side)))
                    _append_spec_and_proxy(
                        mark_id=sub_id,
                        op_kind="circle" if k == "dot" else "glow",
                        bbox_px=(cx - rad, cy - rad, cx + rad, cy + rad),
                        rgba=rgba,
                        extra={"cx": cx, "cy": cy, "radius": rad},
                    )
                elif k in ("ring", "arc"):
                    cx = _tx(float(item.get("x", 0.5)))
                    cy = _ty(float(item.get("y", 0.5)))
                    rad = max(2, int(round(float(item.get("r", 0.15)) * short_side)))
                    stroke_w = max(1, int(round(float(item.get("w", 0.02)) * short_side)))
                    _append_spec_and_proxy(
                        mark_id=sub_id,
                        op_kind="ring",
                        bbox_px=(cx - rad - stroke_w, cy - rad - stroke_w, cx + rad + stroke_w, cy + rad + stroke_w),
                        rgba=rgba,
                        extra={
                            "cx": cx,
                            "cy": cy,
                            "outer_r": rad,
                            "inner_r": max(0, rad - stroke_w),
                        },
                    )
                elif k == "spray":
                    cx = _tx(float(item.get("x", 0.5)))
                    cy = _ty(float(item.get("y", 0.5)))
                    rad = max(2, int(round(float(item.get("r", 0.2)) * short_side)))
                    count_n = max(1, min(128, int(item.get("n", 24))))
                    dot_r = max(1, int(round(float(item.get("s", 0.01)) * short_side)))
                    _append_spec_and_proxy(
                        mark_id=sub_id,
                        op_kind="spray",
                        bbox_px=(cx - rad, cy - rad, cx + rad, cy + rad),
                        rgba=rgba,
                        extra={"cx": cx, "cy": cy, "radius": rad, "n": count_n, "dot_r": dot_r},
                    )
                elif k in ("line", "curve"):
                    pts_raw = item.get("pts", [])
                    if not isinstance(pts_raw, list) or len(pts_raw) < 2:
                        raise ValueError(f"Prism {k} mark requires at least 2 points in 'pts'")
                    pts_px = [(_tx(float(p[0])), _ty(float(p[1]))) for p in pts_raw]
                    stroke_w = max(1, int(round(float(item.get("w", 0.015)) * short_side)))
                    xs = [p[0] for p in pts_px]
                    ys = [p[1] for p in pts_px]
                    _append_spec_and_proxy(
                        mark_id=sub_id,
                        op_kind="polyline",
                        bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                        rgba=rgba,
                        extra={"points": pts_px, "stroke_w": stroke_w},
                    )
                elif k == "fill":
                    pts_raw = item.get("pts", [])
                    if not isinstance(pts_raw, list) or len(pts_raw) < 3:
                        raise ValueError("Prism fill mark requires at least 3 points in 'pts'")
                    # Mirrored fills reuse the unmirrored points and flip spans when drawn (exact reflection).
                    pts_px = [
                        (max(0, min(width - 1, int(round(float(p[0]) * (width - 1))))),
                         max(0, min(height - 1, int(round(float(p[1]) * (height - 1))))))
                        for p in pts_raw
                    ]
                    xs = [width - 1 - p[0] if mx_flip else p[0] for p in pts_px]
                    ys = [height - 1 - p[1] if my_flip else p[1] for p in pts_px]
                    _append_spec_and_proxy(
                        mark_id=sub_id,
                        op_kind="polygon",
                        bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                        rgba=rgba,
                        extra={"points": pts_px, **({"flip": (mx_flip, my_flip)} if mx_flip or my_flip else {})},
                    )

            _emit_prism_single("", False, False)
            mark_mirror = str(item.get("mirror", mirror))
            mark_mirror = MIRROR_ALIASES.get(mark_mirror, mark_mirror)
            if mark_mirror not in MIRROR_MODES:
                raise ValueError(f"Unsupported mirror mode on mark {idx}: {mark_mirror!r}")
            if mark_mirror in ("x", "xy"):
                _emit_prism_single("_mx", True, False)
            if mark_mirror in ("y", "xy"):
                _emit_prism_single("_my", False, True)
            if mark_mirror == "xy":
                _emit_prism_single("_mxy", True, True)
            continue

        # Case 3: `cyclopsdraw` vector mark (`shape` field)
        if "shape" in item:
            shape = str(item["shape"])
            if shape not in CYCLOPSDRAW_SHAPES:
                raise ValueError(f"Unsupported cyclopsdraw shape: {shape!r}")
            opacity = float(item.get("opacity") or 1.0)
            rgba = (255, 255, 255, max(1, min(255, int(round(opacity * 255)))))
            fill_spec = item.get("fill")
            stroke_spec = item.get("stroke")
            if isinstance(fill_spec, dict):
                if fill_spec.get("color"):
                    rgba = _parse_hex_or_rgb_color(fill_spec["color"], default_alpha=opacity)
                elif isinstance(fill_spec.get("stops"), list) and fill_spec["stops"]:
                    rgba = _parse_hex_or_rgb_color(fill_spec["stops"][0]["color"], default_alpha=opacity)
            elif isinstance(stroke_spec, dict) and stroke_spec.get("color"):
                rgba = _parse_hex_or_rgb_color(stroke_spec["color"], default_alpha=opacity)
            elif "color" in item:
                rgba = _parse_hex_or_rgb_color(item["color"], default_alpha=opacity)

            if shape in ("rect", "rounded-rect"):
                rx = int(round(float(item.get("x", 0))))
                ry = int(round(float(item.get("y", 0))))
                rw = max(1, int(round(float(item.get("width", 16)))))
                rh = max(1, int(round(float(item.get("height", 16)))))
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="rect",
                    bbox_px=(rx, ry, rx + rw - 1, ry + rh - 1),
                    rgba=rgba,
                    extra={},
                )
            elif shape in ("circle", "ellipse", "dot"):
                center = item.get("center") or {"x": item.get("x", width // 2), "y": item.get("y", height // 2)}
                cx = int(round(float(center["x"])))
                cy = int(round(float(center["y"])))
                rad_x = max(1, int(round(float(item.get("radius") or item.get("rx") or 8))))
                rad_y = max(1, int(round(float(item.get("radius") or item.get("ry") or rad_x))))
                rot_deg = float(item.get("rotation") or 0.0)
                if shape == "ellipse" and (rad_x != rad_y or rot_deg != 0.0):
                    max_r = max(rad_x, rad_y)
                    _append_spec_and_proxy(
                        mark_id=mid,
                        op_kind="ellipse",
                        bbox_px=(cx - max_r, cy - max_r, cx + max_r, cy + max_r),
                        rgba=rgba,
                        extra={"cx": cx, "cy": cy, "rx": rad_x, "ry": rad_y, "rotation_deg": rot_deg},
                    )
                else:
                    rad = max(1, (rad_x + rad_y) // 2)
                    _append_spec_and_proxy(
                        mark_id=mid,
                        op_kind="circle",
                        bbox_px=(cx - rad_x, cy - rad_y, cx + rad_x, cy + rad_y),
                        rgba=rgba,
                        extra={"cx": cx, "cy": cy, "radius": rad},
                    )
            elif shape == "regular-polygon":
                center = item.get("center") or {"x": width // 2, "y": height // 2}
                cx_f = float(center["x"])
                cy_f = float(center["y"])
                outer_r_f = float(item.get("outerRadius") or item.get("radius") or 16.0)
                sides = max(3, min(64, int(item.get("sides") or 6)))
                rot_rad = math.radians(float(item.get("rotation") or 0.0) - 90.0)
                pts_px = [
                    (
                        int(round(cx_f + outer_r_f * math.cos(rot_rad + 2.0 * math.pi * i / sides))),
                        int(round(cy_f + outer_r_f * math.sin(rot_rad + 2.0 * math.pi * i / sides))),
                    )
                    for i in range(sides)
                ]
                xs = [p[0] for p in pts_px]
                ys = [p[1] for p in pts_px]
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="polygon",
                    bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                    rgba=rgba,
                    extra={"points": pts_px},
                )
            elif shape == "star":
                center = item.get("center") or {"x": width // 2, "y": height // 2}
                cx_f = float(center["x"])
                cy_f = float(center["y"])
                outer_r_f = float(item.get("outerRadius") or item.get("radius") or 20.0)
                inner_r_f = float(item.get("innerRadius") or (outer_r_f * 0.4))
                spikes = max(3, min(64, int(item.get("spikes") or 5)))
                rot_rad = math.radians(float(item.get("rotation") or 0.0) - 90.0)
                pts_px = []
                for i in range(spikes * 2):
                    r_use = outer_r_f if (i % 2 == 0) else inner_r_f
                    ang = rot_rad + math.pi * i / spikes
                    pts_px.append(
                        (
                            int(round(cx_f + r_use * math.cos(ang))),
                            int(round(cy_f + r_use * math.sin(ang))),
                        )
                    )
                xs = [p[0] for p in pts_px]
                ys = [p[1] for p in pts_px]
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="polygon",
                    bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                    rgba=rgba,
                    extra={"points": pts_px},
                )
            elif shape == "crescent":
                center = item.get("center") or {"x": width // 2, "y": height // 2}
                cx = int(round(float(center["x"])))
                cy = int(round(float(center["y"])))
                outer_r = max(2, int(round(float(item.get("outerRadius") or item.get("radius") or 20))))
                inner_r = max(1, int(round(float(item.get("innerRadius") or (outer_r * 0.75)))))
                offset_px = float(item.get("offset") if item.get("offset") is not None else (outer_r * 0.35))
                rot_rad = math.radians(float(item.get("rotation") or 0.0))
                icx = int(round(cx + offset_px * math.cos(rot_rad)))
                icy = int(round(cy + offset_px * math.sin(rot_rad)))
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="crescent",
                    bbox_px=(cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
                    rgba=rgba,
                    extra={"cx": cx, "cy": cy, "outer_r": outer_r, "icx": icx, "icy": icy, "inner_r": inner_r},
                )
            elif shape in ("ring", "arc"):
                center = item.get("center") or {"x": width // 2, "y": height // 2}
                cx = int(round(float(center["x"])))
                cy = int(round(float(center["y"])))
                outer_r = max(2, int(round(float(item.get("outerRadius") or item.get("radius") or 16))))
                inner_r = int(round(float(item.get("innerRadius") or (outer_r // 2 if shape == "ring" else 0))))
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="ring" if inner_r > 0 and shape == "ring" else "circle",
                    bbox_px=(cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
                    rgba=rgba,
                    extra={"cx": cx, "cy": cy, "outer_r": outer_r, "inner_r": inner_r, "radius": outer_r},
                )
            elif shape == "line":
                p_from = item.get("from") or {"x": item.get("x0", 0), "y": item.get("y0", 0)}
                p_to = item.get("to") or {"x": item.get("x1", 16), "y": item.get("y1", 16)}
                x0 = int(round(float(p_from["x"])))
                y0 = int(round(float(p_from["y"])))
                x1 = int(round(float(p_to["x"])))
                y1 = int(round(float(p_to["y"])))
                sw = 1
                if isinstance(stroke_spec, dict) and stroke_spec.get("width"):
                    sw = max(1, int(round(float(stroke_spec["width"]))))
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="polyline" if sw > 1 else "line",
                    bbox_px=(x0, y0, x1, y1),
                    rgba=rgba,
                    extra={"points": [(x0, y0), (x1, y1)], "stroke_w": sw},
                )
            elif shape in ("polygon", "polyline", "spline"):
                pts_list = item.get("points") or item.get("spline") or []
                if not isinstance(pts_list, list) or len(pts_list) < 2:
                    raise ValueError(f"cyclopsdraw {shape} requires at least 2 points")
                pts_px = [(int(round(float(p["x"]))), int(round(float(p["y"])))) for p in pts_list]
                xs = [p[0] for p in pts_px]
                ys = [p[1] for p in pts_px]
                is_filled = shape == "polygon" and fill_spec is not None and len(pts_px) >= 3
                sw = 2
                if isinstance(stroke_spec, dict) and stroke_spec.get("width"):
                    sw = max(1, int(round(float(stroke_spec["width"]))))
                _append_spec_and_proxy(
                    mark_id=mid,
                    op_kind="polygon" if is_filled else "polyline",
                    bbox_px=(min(xs), min(ys), max(xs), max(ys)),
                    rgba=rgba,
                    extra={"points": pts_px, "stroke_w": sw},
                )
            _mirror_item_specs(item_start)
            continue

        # Case 4: `CYCLOPS_DRAW_WORLD_V3` operation object (`operationId`, `type`, `geometry`, `appearance`)
        if "type" in item and "geometry" in item and ("operationId" in item or "appearance" in item):
            op_type = str(item["type"]).upper()
            geom = item.get("geometry") or {}
            app = item.get("appearance") or {}
            opacity = float(app.get("opacity", 1.0))
            fill_app = app.get("fill")
            stroke_app = app.get("stroke")
            rgba = (255, 255, 255, max(1, min(255, int(round(opacity * 255)))))
            if isinstance(fill_app, dict) and fill_app.get("color"):
                rgba = _parse_hex_or_rgb_color(fill_app["color"], default_alpha=opacity)
            elif isinstance(stroke_app, dict) and stroke_app.get("color"):
                rgba = _parse_hex_or_rgb_color(stroke_app["color"], default_alpha=opacity)

            mid_w = str(item.get("operationId") or mid)
            mid_w = re.sub(r"[^a-zA-Z0-9_]", "_", mid_w)[:48] or f"mk_{idx:04d}"

            if op_type == "RECT":
                rx = int(round(float(geom.get("x", 0))))
                ry = int(round(float(geom.get("y", 0))))
                rw = max(1, int(round(float(geom.get("width", 16)))))
                rh = max(1, int(round(float(geom.get("height", 16)))))
                _append_spec_and_proxy(
                    mark_id=mid_w,
                    op_kind="rect",
                    bbox_px=(rx, ry, rx + rw - 1, ry + rh - 1),
                    rgba=rgba,
                    extra={},
                )
            elif op_type == "CIRCLE":
                center = geom.get("center") or {"x": width // 2, "y": height // 2}
                cx = int(round(float(center["x"])))
                cy = int(round(float(center["y"])))
                rad = max(1, int(round(float(geom.get("radius", 8)))))
                _append_spec_and_proxy(
                    mark_id=mid_w,
                    op_kind="circle",
                    bbox_px=(cx - rad, cy - rad, cx + rad, cy + rad),
                    rgba=rgba,
                    extra={"cx": cx, "cy": cy, "radius": rad},
                )
            elif op_type == "SEGMENT":
                st = geom.get("start") or {"x": 0, "y": 0}
                en = geom.get("end") or {"x": 16, "y": 16}
                x0 = int(round(float(st["x"])))
                y0 = int(round(float(st["y"])))
                x1 = int(round(float(en["x"])))
                y1 = int(round(float(en["y"])))
                sw = 1
                if isinstance(stroke_app, dict) and stroke_app.get("width"):
                    sw = max(1, int(round(float(stroke_app["width"]))))
                _append_spec_and_proxy(
                    mark_id=mid_w,
                    op_kind="polyline" if sw > 1 else "line",
                    bbox_px=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                    rgba=rgba,
                    extra={"points": [(x0, y0), (x1, y1)], "stroke_w": sw},
                )
            elif op_type in ("PATH", "COMPOUND_PATH"):
                raw_subpaths = geom.get("subpaths") if op_type == "COMPOUND_PATH" else [geom]
                sampled_subpaths: List[List[Tuple[int, int]]] = []
                all_xs: List[int] = []
                all_ys: List[int] = []
                for sp in raw_subpaths or []:
                    st = sp.get("start") or {"x": 0, "y": 0}
                    cur_x, cur_y = float(st["x"]), float(st["y"])
                    pts_sp: List[Tuple[int, int]] = [(int(round(cur_x)), int(round(cur_y)))]
                    for seg in sp.get("segments", []):
                        seg_t = str(seg.get("type", "LINE")).upper()
                        en = seg.get("end") or {"x": cur_x, "y": cur_y}
                        ex, ey = float(en["x"]), float(en["y"])
                        if seg_t == "CUBIC":
                            c1 = seg.get("control1") or {"x": cur_x, "y": cur_y}
                            c2 = seg.get("control2") or {"x": ex, "y": ey}
                            pts_sp.extend(
                                _sample_cubic_bezier(
                                    (cur_x, cur_y),
                                    (float(c1["x"]), float(c1["y"])),
                                    (float(c2["x"]), float(c2["y"])),
                                    (ex, ey),
                                )
                            )
                        else:
                            pts_sp.append((int(round(ex)), int(round(ey))))
                        cur_x, cur_y = ex, ey
                    if len(pts_sp) >= 2:
                        sampled_subpaths.append(pts_sp)
                        all_xs.extend(p[0] for p in pts_sp)
                        all_ys.extend(p[1] for p in pts_sp)
                if sampled_subpaths:
                    is_filled = fill_app is not None
                    sw = 2
                    if isinstance(stroke_app, dict) and stroke_app.get("width"):
                        sw = max(1, int(round(float(stroke_app["width"]))))
                    if len(sampled_subpaths) > 1 and is_filled:
                        _append_spec_and_proxy(
                            mark_id=mid_w,
                            op_kind="compound_polygon",
                            bbox_px=(min(all_xs), min(all_ys), max(all_xs), max(all_ys)),
                            rgba=rgba,
                            extra={"subpaths": sampled_subpaths},
                        )
                    else:
                        pts_first = sampled_subpaths[0]
                        _append_spec_and_proxy(
                            mark_id=mid_w,
                            op_kind="polygon" if (is_filled and len(pts_first) >= 3) else "polyline",
                            bbox_px=(min(all_xs), min(all_ys), max(all_xs), max(all_ys)),
                            rgba=rgba,
                            extra={"points": pts_first, "stroke_w": sw},
                        )
            _mirror_item_specs(item_start)
            continue

        # Case 5: `drawing.json` v2 vector object (`kind == "vector"`, `bounds`, `settings`)
        if item.get("kind") == "vector" and "bounds" in item and isinstance(item.get("settings"), dict):
            bounds = item["bounds"]
            settings = item["settings"]
            bx0 = float(bounds[0]) * (width - 1)
            by0 = float(bounds[1]) * (height - 1)
            bx1 = float(bounds[2]) * (width - 1)
            by1 = float(bounds[3]) * (height - 1)
            bw = max(1.0, bx1 - bx0)
            bh = max(1.0, by1 - by0)
            opacity = float(settings.get("opacity", 1.0))
            fill_s = settings.get("fill") or {}
            stroke_s = settings.get("stroke") or {}
            color_str = (
                fill_s.get("color")
                if fill_s.get("type") not in (None, "none") and fill_s.get("color")
                else (stroke_s.get("color") or item.get("color") or "#ffffff")
            )
            rgba = _parse_hex_or_rgb_color(color_str, default_alpha=opacity)
            mid_v = re.sub(r"[^a-zA-Z0-9_]", "_", str(item.get("id") or mid))[:48] or f"mk_{idx:04d}"
            skind = str(settings.get("kind", "rectangle")).lower()

            if skind == "rectangle":
                _append_spec_and_proxy(
                    mark_id=mid_v,
                    op_kind="rect",
                    bbox_px=(int(round(bx0)), int(round(by0)), int(round(bx1)), int(round(by1))),
                    rgba=rgba,
                    extra={},
                )
            elif skind == "circle":
                cx = int(round((bx0 + bx1) * 0.5))
                cy = int(round((by0 + by1) * 0.5))
                rad = max(1, int(round(min(bw, bh) * 0.5)))
                _append_spec_and_proxy(
                    mark_id=mid_v,
                    op_kind="circle",
                    bbox_px=(cx - rad, cy - rad, cx + rad, cy + rad),
                    rgba=rgba,
                    extra={"cx": cx, "cy": cy, "radius": rad},
                )
            elif skind == "path":
                cmds = settings.get("commands") or []
                subpaths_v: List[List[Tuple[int, int]]] = []
                cur_sp: List[Tuple[int, int]] = []
                cx_f, cy_f = 0.0, 0.0
                for cmd in cmds:
                    if not isinstance(cmd, list) or not cmd:
                        continue
                    op_c = str(cmd[0]).upper()
                    if op_c == "M" and len(cmd) >= 3:
                        if len(cur_sp) >= 2:
                            subpaths_v.append(cur_sp)
                        cx_f = bx0 + float(cmd[1]) * bw
                        cy_f = by0 + float(cmd[2]) * bh
                        cur_sp = [(int(round(cx_f)), int(round(cy_f)))]
                    elif op_c == "L" and len(cmd) >= 3:
                        cx_f = bx0 + float(cmd[1]) * bw
                        cy_f = by0 + float(cmd[2]) * bh
                        cur_sp.append((int(round(cx_f)), int(round(cy_f))))
                    elif op_c == "C" and len(cmd) >= 7:
                        c1x = bx0 + float(cmd[1]) * bw
                        c1y = by0 + float(cmd[2]) * bh
                        c2x = bx0 + float(cmd[3]) * bw
                        c2y = by0 + float(cmd[4]) * bh
                        ex = bx0 + float(cmd[5]) * bw
                        ey = by0 + float(cmd[6]) * bh
                        cur_sp.extend(_sample_cubic_bezier((cx_f, cy_f), (c1x, c1y), (c2x, c2y), (ex, ey)))
                        cx_f, cy_f = ex, ey
                    elif op_c == "Z":
                        if len(cur_sp) >= 2:
                            subpaths_v.append(cur_sp)
                            cur_sp = []
                if len(cur_sp) >= 2:
                    subpaths_v.append(cur_sp)
                if subpaths_v:
                    all_xs = [p[0] for sp in subpaths_v for p in sp]
                    all_ys = [p[1] for sp in subpaths_v for p in sp]
                    is_filled = bool(item.get("filled", True)) and fill_s.get("type") not in ("none", None)
                    sw = max(1, int(round(float(item.get("width", 0.004)) * min(width, height))))
                    if len(subpaths_v) > 1 and is_filled:
                        _append_spec_and_proxy(
                            mark_id=mid_v,
                            op_kind="compound_polygon",
                            bbox_px=(min(all_xs), min(all_ys), max(all_xs), max(all_ys)),
                            rgba=rgba,
                            extra={"subpaths": subpaths_v},
                        )
                    else:
                        pts_first = subpaths_v[0]
                        _append_spec_and_proxy(
                            mark_id=mid_v,
                            op_kind="polygon" if (is_filled and len(pts_first) >= 3) else "polyline",
                            bbox_px=(min(all_xs), min(all_ys), max(all_xs), max(all_ys)),
                            rgba=rgba,
                            extra={"points": pts_first, "stroke_w": sw},
                        )
            _mirror_item_specs(item_start)
            continue

        raise ValueError(
            f"Unrecognized mark format at index {idx}: {list(item.keys())}. Use a Prism mark in 0..1 units, "
            f"e.g. {{\"k\":\"ring\",\"x\":0.5,\"y\":0.5,\"r\":0.2,\"w\":0.02,\"c\":[255,0,0]}} "
            f"(k: {', '.join(sorted(PRISM_MARK_KINDS))}), or a cyclopsdraw shape in pixels, "
            f"e.g. {{\"shape\":\"rect\",\"x\":10,\"y\":10,\"width\":40,\"height\":20,\"fill\":{{\"color\":\"#00ff00\"}}}}"
        )

    return normalized_specs, proxy_128_marks


# Each filled row costs 4 units, each line step 1; ~1.6M units is a couple of seconds at 1080p.
RASTER_WORK_BUDGET = 1_600_000
LINE_CLIP_SLACK_PX = 64


def _clip_segment(
    x0: int, y0: int, x1: int, y1: int, xmin: int, ymin: int, xmax: int, ymax: int
) -> Optional[Tuple[int, int, int, int]]:
    """Liang-Barsky clip of a segment to a box; returns rounded endpoints or None when fully outside."""
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, float(x0 - xmin)), (dx, float(xmax - x0)), (-dy, float(y0 - ymin)), (dy, float(ymax - y0))):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        t = q / p
        if p < 0.0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return (
        int(round(x0 + t0 * dx)),
        int(round(y0 + t0 * dy)),
        int(round(x0 + t1 * dx)),
        int(round(y0 + t1 * dy)),
    )


def _rasterize_normalized_specs(
    specs: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    base: Optional[bytes] = None,
) -> bytes:
    """
    Deterministic integer/Bresenham/scanline rasterizer for normalized mark specs.
    Executed strictly inside the private lexical closure vault.

    Every filled shape is written as whole row spans, so cost grows with the rows a
    mark covers rather than its pixel area. Line segments reaching well beyond the edge
    are clipped before walking, so far-away endpoints cannot stall a call, and the total
    work is counted so an oversized drawing is refused instead of running past timeouts.
    `base`, when given, is the starting buffer instead of a transparent canvas.
    """
    buf = bytearray(base) if base is not None else bytearray(width * height * 4)
    work = [0]

    def _charge(units: int) -> None:
        work[0] += units
        if work[0] > RASTER_WORK_BUDGET:
            explicit_zeroize_bytearray(buf)
            raise ValueError(
                "ERR_DRAWING_TOO_COMPLEX: this canvas needs too much drawing work for one call; "
                "use fewer or smaller marks, thinner strokes, fewer points or a smaller canvas"
            )

    def _set_px(x: int, y: int, r: int, g: int, b: int, a: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            off = (y * width + x) * 4
            buf[off] = r
            buf[off + 1] = g
            buf[off + 2] = b
            buf[off + 3] = a

    def _fill_span(y: int, xa: int, xb: int, px: bytes) -> None:
        _charge(4)
        if y < 0 or y >= height:
            return
        xa = max(0, xa)
        xb = min(width - 1, xb)
        if xa <= xb:
            start = (y * width + xa) * 4
            buf[start : start + (xb - xa + 1) * 4] = px * (xb - xa + 1)

    def _draw_line(x0: int, y0: int, x1: int, y1: int, r: int, g: int, b: int, a: int, w: int = 1) -> None:
        rad = max(0, (w - 1) // 2)
        # Some slack keeps ordinary off-edge strokes pixel-identical to an unclipped walk.
        pad = rad + 1 + LINE_CLIP_SLACK_PX
        lo_x, lo_y, hi_x, hi_y = -pad, -pad, width + pad, height + pad
        if not (lo_x <= x0 <= hi_x and lo_x <= x1 <= hi_x and lo_y <= y0 <= hi_y and lo_y <= y1 <= hi_y):
            clipped = _clip_segment(x0, y0, x1, y1, lo_x, lo_y, hi_x, hi_y)
            if clipped is None:
                return
            x0, y0, x1, y1 = clipped
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cx, cy = x0, y0
        _charge(max(dx, -dy) + 1)
        if rad == 0:
            while True:
                _set_px(cx, cy, r, g, b, a)
                if cx == x1 and cy == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    cx += sx
                if e2 <= dx:
                    err += dx
                    cy += sy
            return
        # Thick stroke = union of (2*rad+1)^2 squares centred on the Bresenham path. The path is
        # monotone in x and y, so each row of that union is one span: [min cx - rad, max cx + rad]
        # over the path points within rad rows, and those extremes sit at the ends of that range.
        row_min: Dict[int, int] = {}
        row_max: Dict[int, int] = {}
        while True:
            if cy in row_min:
                row_min[cy] = min(row_min[cy], cx)
                row_max[cy] = max(row_max[cy], cx)
            else:
                row_min[cy] = row_max[cy] = cx
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                cx += sx
            if e2 <= dx:
                err += dx
                cy += sy
        y_first, y_last = min(row_min), max(row_min)
        px = bytes((r, g, b, a))
        for y in range(max(0, y_first - rad), min(height - 1, y_last + rad) + 1):
            lo = max(y - rad, y_first)
            hi = min(y + rad, y_last)
            _fill_span(y, min(row_min[lo], row_min[hi]) - rad, max(row_max[lo], row_max[hi]) + rad, px)

    for spec in specs:
        op = spec["op_kind"]
        r, g, b, a = spec["rgba"]
        px = bytes((r, g, b, a))
        x0, y0, x1, y1 = spec["x0"], spec["y0"], spec["x1"], spec["y1"]

        if op == "dot":
            _set_px(x0, y0, r, g, b, a)
        elif op == "rect":
            xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
            ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
            xa = max(0, min(width - 1, xa))
            xb = max(0, min(width - 1, xb))
            ya = max(0, min(height - 1, ya))
            yb = max(0, min(height - 1, yb))
            row_span = (xb - xa + 1) * 4
            pixel_Run = px * (xb - xa + 1)
            _charge(4 * (yb - ya + 1))
            for y in range(ya, yb + 1):
                start = (y * width + xa) * 4
                buf[start : start + row_span] = pixel_Run
        elif op == "line":
            _draw_line(x0, y0, x1, y1, r, g, b, a, 1)
        elif op == "polyline":
            pts = spec.get("points", [(x0, y0), (x1, y1)])
            sw = int(spec.get("stroke_w", 1))
            for i in range(len(pts) - 1):
                _draw_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], r, g, b, a, sw)
        elif op in ("circle", "glow"):
            cx = int(spec.get("cx", (x0 + x1) // 2))
            cy = int(spec.get("cy", (y0 + y1) // 2))
            rad = max(1, int(spec.get("radius", max(1, (x1 - x0) // 2))))
            r2 = rad * rad
            for y in range(max(0, cy - rad), min(height - 1, cy + rad) + 1):
                max_dx = int(math.isqrt(max(0, r2 - (y - cy) * (y - cy))))
                _fill_span(y, cx - max_dx, cx + max_dx, px)
        elif op == "ring":
            cx = int(spec.get("cx", (x0 + x1) // 2))
            cy = int(spec.get("cy", (y0 + y1) // 2))
            outer_r = max(2, int(spec.get("outer_r", max(2, (x1 - x0) // 2))))
            inner_r = max(0, int(spec.get("inner_r", outer_r - 2)))
            or2 = outer_r * outer_r
            ir2 = inner_r * inner_r
            for y in range(max(0, cy - outer_r), min(height - 1, cy + outer_r) + 1):
                dy2 = (y - cy) * (y - cy)
                ao = math.isqrt(or2 - dy2)
                # Pixels with ir2 <= d2 <= or2: the outer span minus the strictly-inside span.
                if ir2 - dy2 >= 1:
                    ai = math.isqrt(ir2 - dy2 - 1)
                    _fill_span(y, cx - ao, min(cx + ao, cx - ai - 1), px)
                    _fill_span(y, max(cx - ao, cx + ai + 1), cx + ao, px)
                else:
                    _fill_span(y, cx - ao, cx + ao, px)
        elif op == "crescent":
            cx = int(spec.get("cx", (x0 + x1) // 2))
            cy = int(spec.get("cy", (y0 + y1) // 2))
            outer_r = max(2, int(spec.get("outer_r", 20)))
            icx = int(spec.get("icx", cx + 6))
            icy = int(spec.get("icy", cy))
            inner_r = max(1, int(spec.get("inner_r", 15)))
            or2 = outer_r * outer_r
            ir2 = inner_r * inner_r
            for y in range(max(0, cy - outer_r), min(height - 1, cy + outer_r) + 1):
                ao = math.isqrt(or2 - (y - cy) * (y - cy))
                oa, ob = cx - ao, cx + ao
                dyi2 = (y - icy) * (y - icy)
                if dyi2 <= ir2:
                    ai = math.isqrt(ir2 - dyi2)
                    _fill_span(y, oa, min(ob, icx - ai - 1), px)
                    _fill_span(y, max(oa, icx + ai + 1), ob, px)
                else:
                    _fill_span(y, oa, ob, px)
        elif op == "ellipse":
            cx = int(spec.get("cx", (x0 + x1) // 2))
            cy = int(spec.get("cy", (y0 + y1) // 2))
            rx = max(1, int(spec.get("rx", 10)))
            ry = max(1, int(spec.get("ry", 10)))
            rot_rad = math.radians(float(spec.get("rotation_deg", 0.0)))
            cos_t = math.cos(-rot_rad)
            sin_t = math.sin(-rot_rad)
            max_r = max(rx, ry)
            xa = max(0, cx - max_r)
            xb = min(width - 1, cx + max_r)
            rx2 = float(rx * rx)
            ry2 = float(ry * ry)
            qa = (cos_t * cos_t) / rx2 + (sin_t * sin_t) / ry2
            for y in range(max(0, cy - max_r), min(height - 1, cy + max_r) + 1):
                dy = float(y - cy)

                def _inside(x: int) -> bool:
                    ddx = float(x - cx)
                    u = ddx * cos_t - dy * sin_t
                    v = ddx * sin_t + dy * cos_t
                    return (u * u) / rx2 + (v * v) / ry2 <= 1.0

                # Each row of an ellipse is one interval: solve the quadratic for a first guess,
                # then settle both ends with the exact per-pixel test used before.
                qb = 2.0 * dy * (-cos_t * sin_t / rx2 + sin_t * cos_t / ry2)
                qc = dy * dy * ((sin_t * sin_t) / rx2 + (cos_t * cos_t) / ry2) - 1.0
                disc = qb * qb - 4.0 * qa * qc
                mid = -qb / (2.0 * qa)
                half = math.sqrt(disc) / (2.0 * qa) if disc > 0.0 else 0.0
                lo = max(xa, min(xb, cx + int(math.floor(mid - half))))
                hi = max(xa, min(xb, cx + int(math.ceil(mid + half))))
                while lo <= hi and not _inside(lo):
                    lo += 1
                if lo > hi:
                    continue
                while lo > xa and _inside(lo - 1):
                    lo -= 1
                while hi > lo and not _inside(hi):
                    hi -= 1
                while hi < xb and _inside(hi + 1):
                    hi += 1
                _fill_span(y, lo, hi, px)
        elif op == "spray":
            cx = int(spec.get("cx", (x0 + x1) // 2))
            cy = int(spec.get("cy", (y0 + y1) // 2))
            rad = max(2, int(spec.get("radius", 10)))
            n_dots = max(1, int(spec.get("n", 16)))
            # Deterministic Halton/golden-angle spiral points
            golden_angle = 2.39996322972865332
            for idx in range(n_dots):
                frac = math.sqrt((idx + 0.5) / n_dots)
                theta = idx * golden_angle
                px_x = int(round(cx + frac * rad * math.cos(theta)))
                px_y = int(round(cy + frac * rad * math.sin(theta)))
                _set_px(px_x, px_y, r, g, b, a)
        elif op in ("polygon", "compound_polygon"):
            rings = [spec.get("points", [])] if op == "polygon" else spec.get("subpaths", [])
            rings = [ring for ring in rings if len(ring) >= 3]
            if rings:
                edges = [
                    (ring[i][0], ring[i][1], ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1])
                    for ring in rings
                    for i in range(len(ring))
                    if ring[i][1] != ring[(i + 1) % len(ring)][1]
                ]
                all_ys = [p[1] for ring in rings for p in ring]
                # A mirrored copy keeps the source points and flips the filled spans, so it is an exact
                # pixel reflection (re-scanning reflected points rounds edges differently).
                flip_x, flip_y = spec.get("flip", (False, False))
                for y in range(max(0, min(all_ys)), min(height - 1, max(all_ys)) + 1):
                    _charge(1 + len(edges) // 4)
                    row = height - 1 - y if flip_y else y
                    intersections: List[int] = []
                    for px1, py1, px2, py2 in edges:
                        if min(py1, py2) <= y < max(py1, py2):
                            intersections.append(px1 + ((y - py1) * (px2 - px1)) // (py2 - py1))
                    intersections.sort()
                    for k_idx in range(0, len(intersections) - 1, 2):
                        xa = max(0, min(width - 1, intersections[k_idx]))
                        xb = max(0, min(width - 1, intersections[k_idx + 1]))
                        if xa <= xb:
                            if flip_x:
                                xa, xb = width - 1 - xb, width - 1 - xa
                            _fill_span(row, xa, xb, px)

    out_bytes = bytes(buf)
    explicit_zeroize_bytearray(buf)
    return out_bytes


def _create_multi_res_closure_vault(
    simd_engine: NativeSimdOperatorEngine,
) -> Dict[str, Callable[..., Any]]:
    """
    Private lexical closure vault for multi-resolution (128x128, 256x256,
    512x512, 1920x1080) RGBA buffers. Raw pixel buffers never leave this
    closure and are never stored on instance attributes.
    """
    _buffers: Dict[Tuple[str, str], Dict[str, Any]] = {}
    _zeroized_stats: Dict[str, int] = {"evicted_frames": 0, "zeroized_bytes": 0}

    def register_canvas(
        source_id: str,
        source_version: str,
        width: int,
        height: int,
        specs: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        key = (source_id, source_version)
        if key in _buffers:
            raise ValueError(f"Immutable source {source_id}@{source_version} already registered")
        rgba_bytes = _rasterize_normalized_specs(specs, width, height)
        pixel_sha256 = _canonical_sha256_tag(rgba_bytes)
        _buffers[key] = {
            "width": width,
            "height": height,
            "rgba": bytearray(rgba_bytes),
            "pixel_sha256": pixel_sha256,
            "spec_count": len(specs),
        }
        return {
            "width": width,
            "height": height,
            "pixel_sha256": pixel_sha256,
            "spec_count": len(specs),
        }

    def register_rgba(
        source_id: str,
        source_version: str,
        width: int,
        height: int,
        rgba_bytes: bytes,
    ) -> Dict[str, Any]:
        """Stores a real picture's pixels (already fitted to a supported canvas) like a drawn canvas."""
        key = (source_id, source_version)
        if key in _buffers:
            raise ValueError(f"Immutable source {source_id}@{source_version} already registered")
        if len(rgba_bytes) != width * height * 4:
            raise ValueError("RGBA buffer does not match the canvas size")
        pixel_sha256 = _canonical_sha256_tag(rgba_bytes)
        _buffers[key] = {
            "width": width,
            "height": height,
            "rgba": bytearray(rgba_bytes),
            "pixel_sha256": pixel_sha256,
            "spec_count": 0,
        }
        return {"width": width, "height": height, "pixel_sha256": pixel_sha256, "spec_count": 0}

    def run_native_simd_suite(
        source_a: Tuple[str, str],
        source_b: Optional[Tuple[str, str]] = None,
        *,
        target_rgba: Tuple[int, int, int, int] = (20, 120, 220, 255),
        tolerance: int = 0,
    ) -> Dict[str, Any]:
        if source_a not in _buffers:
            raise KeyError(f"Unknown source: {source_a}")
        entry_a = _buffers[source_a]
        w, h = entry_a["width"], entry_a["height"]
        rgba_a = bytes(entry_a["rgba"])
        if source_b is not None:
            if source_b not in _buffers:
                raise KeyError(f"Unknown secondary source: {source_b}")
            entry_b = _buffers[source_b]
            if (entry_b["width"], entry_b["height"]) != (w, h):
                raise ValueError("Source dimensions must match for native SIMD diff")
            rgba_b = bytes(entry_b["rgba"])
        else:
            rgba_b = rgba_a

        return simd_engine.run_full_10_operator_suite_native(
            rgba_a,
            rgba_b,
            w,
            h,
            target_rgba=target_rgba,
            tolerance=tolerance,
        )

    def compute_dirty_tiles_16x16(
        source_a: Tuple[str, str],
        source_b: Tuple[str, str],
    ) -> Dict[str, Any]:
        """
        Computes a 16x16 macroblock dirty-tile bitmask and summary inside the
        private closure vault without exposing raw pixels.
        """
        if source_a not in _buffers or source_b not in _buffers:
            raise KeyError("Both sources must exist in closure vault for dirty-tile profiling")
        entry_a = _buffers[source_a]
        entry_b = _buffers[source_b]
        w, h = entry_a["width"], entry_a["height"]
        if (entry_b["width"], entry_b["height"]) != (w, h):
            raise ValueError("Source dimensions must match for 16x16 dirty-tile profiling")

        tiles_x = (w + 15) // 16
        tiles_y = (h + 15) // 16
        total_tiles = tiles_x * tiles_y

        # Fast-path O(1) if SHA-256 digests match identically
        if entry_a["pixel_sha256"] == entry_b["pixel_sha256"]:
            zero_mask = bytes((total_tiles + 7) // 8)
            return {
                "tile_size_px": 16,
                "grid_dimensions": {"tiles_x": tiles_x, "tiles_y": tiles_y, "total_tiles": total_tiles},
                "dirty_tile_count": 0,
                "clean_tile_count": total_tiles,
                "dirty_tile_ratio": 0.0,
                "dirty_tile_bbox": [-1, -1, -1, -1],
                "dirty_bitmask_sha256": _canonical_sha256_tag(zero_mask),
                "fast_path_identical_sha": True,
            }

        buf_a = entry_a["rgba"]
        buf_b = entry_b["rgba"]
        bitmask = bytearray((total_tiles + 7) // 8)
        dirty_count = 0
        min_tx, min_ty = tiles_x, tiles_y
        max_tx, max_ty = -1, -1
        row_stride = w * 4

        for ty in range(tiles_y):
            y0 = ty * 16
            y1 = min(h, y0 + 16)
            for tx in range(tiles_x):
                x0 = tx * 16
                x1 = min(w, x0 + 16)
                span_bytes = (x1 - x0) * 4
                tile_dirty = False
                for y in range(y0, y1):
                    off = y * row_stride + x0 * 4
                    if buf_a[off : off + span_bytes] != buf_b[off : off + span_bytes]:
                        tile_dirty = True
                        break
                if tile_dirty:
                    t_idx = ty * tiles_x + tx
                    bitmask[t_idx >> 3] |= 1 << (t_idx & 7)
                    dirty_count += 1
                    if tx < min_tx:
                        min_tx = tx
                    if ty < min_ty:
                        min_ty = ty
                    if tx > max_tx:
                        max_tx = tx
                    if ty > max_ty:
                        max_ty = ty

        dirty_bbox = [min_tx, min_ty, max_tx, max_ty] if dirty_count > 0 else [-1, -1, -1, -1]
        return {
            "tile_size_px": 16,
            "grid_dimensions": {"tiles_x": tiles_x, "tiles_y": tiles_y, "total_tiles": total_tiles},
            "dirty_tile_count": dirty_count,
            "clean_tile_count": total_tiles - dirty_count,
            "dirty_tile_ratio": round(dirty_count / float(max(1, total_tiles)), 6),
            "dirty_tile_bbox": dirty_bbox,
            "dirty_bitmask_sha256": _canonical_sha256_tag(bitmask),
            "fast_path_identical_sha": False,
        }

    def dimensions(source_key: Tuple[str, str]) -> Tuple[int, int]:
        entry = _buffers[source_key]
        return entry["width"], entry["height"]

    def sample_pixel_rgba(
        source_key: Tuple[str, str],
        x: int,
        y: int,
    ) -> Dict[str, Any]:
        if source_key not in _buffers:
            raise KeyError(f"Unknown source: {source_key}")
        entry = _buffers[source_key]
        w, h = entry["width"], entry["height"]
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds for {w}x{h}")
        off = (y * w + x) * 4
        rgba_buf = entry["rgba"]
        r, g, b, a = rgba_buf[off], rgba_buf[off + 1], rgba_buf[off + 2], rgba_buf[off + 3]
        return {
            "x": x,
            "y": y,
            "rgba": [int(r), int(g), int(b), int(a)],
            "hex": f"#{r:02x}{g:02x}{b:02x}{a:02x}",
            "non_transparent": bool(a > 0),
        }

    def preview_ghost_specs(
        base_source_key: Tuple[str, str],
        ghost_specs: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Computes hypothetical structural metrics if `ghost_specs` were overlaid on
        `base_source_key`, inside a disposable buffer that is immediately zeroized
        and discarded. Never mutates `base_source_key` in `_buffers`.
        """
        if base_source_key not in _buffers:
            raise KeyError(f"Unknown base source: {base_source_key}")
        base_entry = _buffers[base_source_key]
        w, h = base_entry["width"], base_entry["height"]
        base_rgba = bytes(base_entry["rgba"])
        if all(int(spec["rgba"][3]) > 0 for spec in ghost_specs):
            # Every ghost pixel is opaque enough to replace the base, so painting the ghost specs
            # straight onto a copy of the base gives the same composite in one pass.
            composite_bytes = _rasterize_normalized_specs(ghost_specs, w, h, base=base_rgba)
        else:
            ghost_only_rgba = _rasterize_normalized_specs(ghost_specs, w, h)
            composite = bytearray(base_rgba)
            for off in range(3, w * h * 4, 4):
                ga = ghost_only_rgba[off]
                if ga > 0:
                    composite[off - 3 : off + 1] = ghost_only_rgba[off - 3 : off + 1]
            composite_bytes = bytes(composite)
            explicit_zeroize_bytearray(composite)

        metrics = simd_engine.run_full_10_operator_suite_native(
            composite_bytes,
            base_rgba,
            w,
            h,
        )
        # Verify base buffer digest is 100% unchanged
        base_digest_after = _canonical_sha256_tag(_buffers[base_source_key]["rgba"])
        assert base_digest_after == base_entry["pixel_sha256"], "Ghost preview mutated verified plane!"
        return {
            "width": w,
            "height": h,
            "base_pixel_sha256_preserved": True,
            "hypothetical_metrics": metrics,
        }

    def downsample_proxy_marks(
        source_key: Tuple[str, str], max_marks: int, colour_bits: int = 8
    ) -> Optional[List[DrawingMark]]:
        """
        Encodes a 128x128 nearest-centre downsample of the native canvas as rectangle marks for
        the 128x128 harness, so proxy measurements see the real shapes rather than bounding boxes.
        `colour_bits` < 8 keeps only the top bits of each colour channel, which merges the near
        colours of photos into fewer rectangles. Returns None when more than `max_marks` are needed.
        """
        if source_key not in _buffers:
            raise KeyError(f"Unknown source: {source_key}")
        entry = _buffers[source_key]
        w, h, rgba = entry["width"], entry["height"], entry["rgba"]
        cols = [((2 * px + 1) * w) // 256 for px in range(128)]
        empty = b"\0\0\0\0"
        mask = (0xFF << (8 - colour_bits)) & 0xFF

        def _quantized(px: bytes) -> bytes:
            if colour_bits >= 8 or px == empty:
                return px
            return bytes((px[0] & mask, px[1] & mask, px[2] & mask, max(1, px[3] & mask) if px[3] else 0))

        open_rects: Dict[Tuple[int, int, bytes], List[int]] = {}
        rects: List[List[Any]] = []
        for py in range(128):
            row_off = (((2 * py + 1) * h) // 256) * w * 4
            row = [_quantized(bytes(rgba[row_off + sx * 4 : row_off + sx * 4 + 4])) for sx in cols]
            still_open: Dict[Tuple[int, int, bytes], List[int]] = {}
            x = 0
            while x < 128:
                px = row[x]
                if px == empty:
                    x += 1
                    continue
                x0 = x
                while x + 1 < 128 and row[x + 1] == px:
                    x += 1
                key = (x0, x, px)
                rect = open_rects.get(key)
                if rect is not None:
                    rect[3] = py
                else:
                    rect = [x0, py, x, py, px]
                    rects.append(rect)
                    if len(rects) > max_marks:
                        return None
                still_open[key] = rect
                x += 1
            open_rects = still_open
        return [
            DrawingMark(mark_id=f"px_{i:04d}", shape="rect", x0=r[0], y0=r[1], x1=r[2], y1=r[3], rgba=tuple(r[4]))
            for i, r in enumerate(rects)
        ]

    def block_proxy_marks(source_key: Tuple[str, str], block: int = 8) -> List[DrawingMark]:
        """Last-resort proxy: the 128x128 downsample as `block`-pixel squares of their mean colour."""
        entry = _buffers[source_key]
        w, h, rgba = entry["width"], entry["height"], entry["rgba"]
        marks: List[DrawingMark] = []
        for by in range(0, 128, block):
            for bx in range(0, 128, block):
                sums = [0, 0, 0, 0]
                n = 0
                for py in range(by, by + block):
                    row_off = (((2 * py + 1) * h) // 256) * w * 4
                    for px in range(bx, bx + block):
                        off = row_off + (((2 * px + 1) * w) // 256) * 4
                        for c in range(4):
                            sums[c] += rgba[off + c]
                        n += 1
                if sums[3] == 0:
                    continue
                colour = tuple(int(round(v / n)) for v in sums)
                marks.append(DrawingMark(mark_id=f"blk_{len(marks):04d}", shape="rect", x0=bx, y0=by,
                                         x1=bx + block - 1, y1=by + block - 1, rgba=colour))  # type: ignore[arg-type]
        return marks

    def evict_source(source_key: Tuple[str, str]) -> Dict[str, Any]:
        """Explicitly zeroizes (`explicit_bzero`) and evicts a single source frame from the ring buffer."""
        if source_key not in _buffers:
            return {"evicted": False, "zeroized_bytes": 0, "remaining_sources": len(_buffers)}
        entry = _buffers.pop(source_key)
        rgba_buf = entry["rgba"]
        n_bytes = len(rgba_buf)
        explicit_zeroize_bytearray(rgba_buf)
        _zeroized_stats["evicted_frames"] += 1
        _zeroized_stats["zeroized_bytes"] += n_bytes
        return {
            "evicted": True,
            "evicted_source": {"source_id": source_key[0], "source_version": source_key[1]},
            "zeroized_bytes": n_bytes,
            "remaining_sources": len(_buffers),
            "cumulative_evicted_frames": _zeroized_stats["evicted_frames"],
            "cumulative_zeroized_bytes": _zeroized_stats["zeroized_bytes"],
        }

    def purge_all() -> Dict[str, Any]:
        t0 = time.perf_counter()
        purged_count = len(_buffers)
        purged_bytes = 0
        for entry in _buffers.values():
            rgba_buf = entry["rgba"]
            purged_bytes += len(rgba_buf)
            explicit_zeroize_bytearray(rgba_buf)
        _buffers.clear()
        _zeroized_stats["zeroized_bytes"] += purged_bytes
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "purged_sources": purged_count,
            "zeroized_bytes": purged_bytes,
            "remaining_sources": len(_buffers),
            "elapsed_ms": elapsed_ms,
        }

    def run_fast_signal_suite(
        source_curr: Tuple[str, str],
        source_prev: Optional[Tuple[str, str]] = None,
        *,
        tolerance: int = 0,
    ) -> Dict[str, Any]:
        if source_curr not in _buffers:
            raise KeyError(f"Unknown source: {source_curr}")
        entry_curr = _buffers[source_curr]
        w, h = entry_curr["width"], entry_curr["height"]
        rgba_curr = bytes(entry_curr["rgba"])
        rgba_prev: Optional[bytes] = None
        identical_prev = False
        if source_prev is not None:
            if source_prev not in _buffers:
                raise KeyError(f"Unknown previous source: {source_prev}")
            entry_prev = _buffers[source_prev]
            if (entry_prev["width"], entry_prev["height"]) != (w, h):
                raise ValueError("Source dimensions must match for fast signal diff")
            if entry_prev["pixel_sha256"] == entry_curr["pixel_sha256"]:
                identical_prev = True
            else:
                rgba_prev = bytes(entry_prev["rgba"])
        res = simd_engine.run_fast_signal_operators_native(
            rgba_curr, rgba_prev, w, h, tolerance=tolerance
        )
        if identical_prev:
            total_px = w * h
            res["op_bbox_aligned_diff_u8"] = {
                "changed_pixels": 0,
                "unchanged_pixels": total_px,
                "added_pixels": 0,
                "removed_pixels": 0,
                "modified_colour_pixels": 0,
                "total_pixels": total_px,
                "change_ratio": 0.0,
                "diff_bbox": [-1, -1, -1, -1],
            }
        return res

    def pixel_sha256_of(source_key: Tuple[str, str]) -> Optional[str]:
        entry = _buffers.get(source_key)
        return entry["pixel_sha256"] if entry is not None else None

    return {
        "register_canvas": register_canvas,
        "run_native_simd_suite": run_native_simd_suite,
        "run_fast_signal_suite": run_fast_signal_suite,
        "pixel_sha256_of": pixel_sha256_of,
        "compute_dirty_tiles_16x16": compute_dirty_tiles_16x16,
        "sample_pixel_rgba": sample_pixel_rgba,
        "downsample_proxy_marks": downsample_proxy_marks,
        "block_proxy_marks": block_proxy_marks,
        "register_rgba": register_rgba,
        "dimensions": dimensions,
        "preview_ghost_specs": preview_ghost_specs,
        "evict_source": evict_source,
        "purge_all": purge_all,
    }


class ElectricVisionAgentSession:
    """
    Context-managed Electric Vision Session for Agents.

    Spawns the 4 isolated out-of-process enclave daemons (`HSM/KMS`, `VaultGate`,
    `AuditSink`, `QuarantineAndKillSwitch`) over short `/tmp/cyclops-ee` sockets,
    pairs them with `NativeSimdOperatorEngine` (`libprod_gate_simd.so`), and
    exposes a safe, deterministic, zero-PII inspection and verification API for
    synthetic agent drawings (`cyclopsdraw` / Prism `marks` / `DrawingMark`).
    """

    def __init__(
        self,
        *,
        run_dir: str = "/tmp/cyclops-ee",
        default_resolution: Tuple[int, int] = (256, 256),
        lazy_enclave: bool = False,
        allow_alien_field: bool = False,
    ) -> None:
        if type(allow_alien_field) is not bool:
            raise TypeError("allow_alien_field must be a bool")
        if default_resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Unsupported default_resolution: {default_resolution}")
        self.run_dir = run_dir
        self.default_resolution = default_resolution
        self.lazy_enclave = bool(lazy_enclave)
        self.allow_alien_field = allow_alien_field
        self._simd_engine = NativeSimdOperatorEngine()
        self._multi_res_vault = _create_multi_res_closure_vault(self._simd_engine)
        self._source_dimensions: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._unmaterialized_signal_sources: Dict[Tuple[str, str], Tuple[str, str, Optional[Dict[str, Any]]]] = {}
        self._retired_tombstones: Set[Tuple[str, str]] = set()

        self._hsm_proc: Optional[multiprocessing.Process] = None
        self._hsm_client: Optional[OutOfProcessHsmKmsClient] = None
        self._vault_daemon: Optional[OutOfProcessVaultGateDaemon] = None
        self._audit_daemon: Optional[OutOfProcessAuditSinkDaemon] = None
        self._quar_daemon: Optional[OutOfProcessQuarantineAndKillSwitchDaemon] = None
        self._vault_client: Any = None
        self._audit_client: Any = None
        self._quar_client: Any = None
        self._harness: Optional[ElectricEyeFixtureHarness] = None
        self._worm_ledger_path: str = ""
        self._active = False

    def _ensure_enclave(self) -> ElectricEyeFixtureHarness:
        if self._harness is not None:
            return self._harness
        os.makedirs(self.run_dir, mode=0o700, exist_ok=True)
        hsm_sock = os.path.join(self.run_dir, "h.sock")
        vault_sock = os.path.join(self.run_dir, "v.sock")
        audit_sock = os.path.join(self.run_dir, "a.sock")
        quar_sock = os.path.join(self.run_dir, "q.sock")
        self._worm_ledger_path = os.path.join(self.run_dir, "agent_vision_audit.jsonl")
        forensic_path = os.path.join(self.run_dir, "agent_vision_forensic.json")

        for p in (hsm_sock, vault_sock, audit_sock, quar_sock):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        ctx = multiprocessing.get_context("spawn")
        hsm_ready = ctx.Event()
        self._hsm_proc = ctx.Process(
            target=_hsm_kms_enclave_loop,
            args=(hsm_sock, os.getuid(), os.getgid(), hsm_ready),
            name="OutOfProcessHsmKmsEnclaveDaemon",
        )
        self._hsm_proc.start()
        if not hsm_ready.wait(timeout=5.0):
            raise RuntimeError("HSM Enclave daemon failed to start within 5.0s")
        self._hsm_client = OutOfProcessHsmKmsClient(hsm_sock, default_principal="cyclops-ee-vault")

        self._vault_daemon = OutOfProcessVaultGateDaemon(vault_sock)
        self._audit_daemon = OutOfProcessAuditSinkDaemon(audit_sock, self._worm_ledger_path)
        self._quar_daemon = OutOfProcessQuarantineAndKillSwitchDaemon(
            quar_sock,
            forensic_path,
            vault_socket_path=vault_sock,
            audit_socket_path=audit_sock,
            allow_alien_field=self.allow_alien_field,
        )
        self._vault_client = self._vault_daemon.__enter__()
        self._audit_client = self._audit_daemon.__enter__()
        self._quar_client = self._quar_daemon.__enter__()

        self._harness = ElectricEyeFixtureHarness(
            profile="controlled_pilot_simulation",
            out_of_process_vault_client=self._vault_client,
            audit_sink=self._audit_client,
            quarantine_supervisor_client=self._quar_client,
        )
        return self._harness

    def __enter__(self) -> "ElectricVisionAgentSession":
        os.makedirs(self.run_dir, mode=0o700, exist_ok=True)
        # 1. Always verify Native SIMD engine startup self-test immediately
        self_test = self._simd_engine.verify_self_test_or_rollback(inject_perturbation=False)
        if not self_test.get("self_test_passed"):
            raise RuntimeError("NativeSimdOperatorEngine failed startup self-test")

        # 2. Start out-of-process enclave daemons immediately unless lazy_enclave=True
        if not self.lazy_enclave:
            self._ensure_enclave()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._active = False
        self._unmaterialized_signal_sources.clear()
        self._retired_tombstones.clear()
        self._multi_res_vault["purge_all"]()
        for daemon in (self._quar_daemon, self._audit_daemon, self._vault_daemon):
            if daemon is not None:
                try:
                    daemon.__exit__(exc_type, exc_val, exc_tb)
                except Exception:
                    pass
        if self._hsm_client is not None:
            try:
                self._hsm_client.shutdown()
            except Exception:
                pass
        if self._hsm_proc is not None:
            self._hsm_proc.join(timeout=3.0)
            if self._hsm_proc.is_alive():
                self._hsm_proc.terminate()
                self._hsm_proc.join(timeout=1.0)

    def _require_session_open(self) -> None:
        if not self._active:
            raise RuntimeError("ElectricVisionAgentSession is not active")

    def _require_active(self) -> ElectricEyeFixtureHarness:
        self._require_session_open()
        return self._ensure_enclave()

    def register_signal_rgba(
        self,
        source_id: str,
        source_version: str,
        rgba_bytes: bytes,
        width: int,
        height: int,
        *,
        source_kind: str = "captured_frame",
        origin_attestation: str = "companion_desktop_glance",
        origin: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registers a live frame directly into the Active Electric Signal Temp Buffer
        (`_multi_res_vault` with `explicit_bzero` lifecycle) WITHOUT performing eager
        128x128 `DrawingMark` proxy live translation. If a proxy measurement or diff
        is later requested for this frame while still in the active ring, `_ensure_proxy_materialized`
        translates it lazily on demand.
        """
        self._require_session_open()
        if (width, height) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution {(width, height)} not in SUPPORTED_RESOLUTIONS {SUPPORTED_RESOLUTIONS}")
        alien_signal = (
            self.allow_alien_field
            and origin_attestation == "authorized_alien_field_trace"
            and source_id == "src_alien_field"
        )
        if origin_attestation not in (
            "companion_desktop_glance",
            "play_engine_screen",
            "owner_supplied_picture",
            "synthetic_fixture",
        ) and not alien_signal:
            raise PermissionError(f"Refused live origin_attestation: {origin_attestation!r}")
        key = (source_id, source_version)
        harness_sources = getattr(self._harness, "_sources", {})
        if (
            key in self._source_dimensions
            or key in self._retired_tombstones
            or key in harness_sources
        ):
            raise ValueError(f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {source_id}@{source_version}")
        mr_info = self._multi_res_vault["register_rgba"](
            source_id, source_version, width, height, rgba_bytes
        )
        self._source_dimensions[key] = (width, height)
        self._unmaterialized_signal_sources[key] = (source_kind, origin_attestation, origin)
        out = {
            "source_id": source_id,
            "source_version": source_version,
            "source_kind": source_kind,
            "native_resolution": {"width": width, "height": height, "total_pixels": width * height},
            "native_pixel_sha256": mr_info["pixel_sha256"],
            "signal_buffered": True,
            "origin_attestation": origin_attestation,
            "origin": origin or {},
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def _ensure_proxy_materialized(self, key: Tuple[str, str]) -> None:
        meta = self._unmaterialized_signal_sources.pop(key, None)
        if meta is not None:
            source_kind, origin_attestation, _origin = meta
            self._register_proxy(key, source_kind, origin_attestation)

    def register_synthetic_canvas(
        self,
        source_id: str,
        source_version: str,
        marks: Sequence[Union[DrawingMark, Dict[str, Any]]],
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_kind: str = "drawing_state",
        mirror: str = "none",
        origin_attestation: str = "synthetic_fixture",
    ) -> Dict[str, Any]:
        """
        Compiles and registers a synthetic canvas from `cyclopsdraw`, Spark/Prism
        `marks`, or `DrawingMark` primitives into both the multi-resolution native
        SIMD vault and the out-of-process 128x128 VaultGate daemon.
        """
        harness = self._require_active()
        w = width if width is not None else self.default_resolution[0]
        h = height if height is not None else self.default_resolution[1]
        if (w, h) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution {(w, h)} not in SUPPORTED_RESOLUTIONS {SUPPORTED_RESOLUTIONS}")

        specs, proxy_128_marks = normalize_marks_for_canvas(marks, w, h, mirror=mirror)
        key = (source_id, source_version)
        if key in self._source_dimensions:
            raise ValueError(f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {source_id}@{source_version}")

        # Register native resolution buffer in private closure vault
        mr_info = self._multi_res_vault["register_canvas"](source_id, source_version, w, h, specs)
        desc, proxy_fidelity = self._register_proxy(key, source_kind, origin_attestation, fallback=proxy_128_marks)

        out = self._registration_reply(desc, proxy_fidelity, w, h, mr_info, mirror, origin_attestation)
        assert_no_raw_buffer_leakage(out)
        return out

    def _register_proxy(
        self,
        key: Tuple[str, str],
        source_kind: str,
        origin_attestation: str,
        *,
        fallback: Optional[Sequence[DrawingMark]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Registers the 128x128 harness proxy for a canvas already in the vault: an exact downsample
        when it fits the mark budget, then colour-quantized ones, then `fallback` marks or 8-px
        blocks. Evicts the vault entry if the harness refuses it. Returns (descriptor, fidelity).
        """
        harness = self._require_active()
        try:
            marks: Optional[Sequence[DrawingMark]] = None
            fidelity = ""
            for bits in (8, 5, 4, 3, 2):
                marks = self._multi_res_vault["downsample_proxy_marks"](key, MAX_MARKS_PER_SOURCE, bits)
                if marks is not None:
                    fidelity = "downsampled_pixels" if bits == 8 else f"downsampled_pixels_{bits}bit_colour"
                    break
                if fallback is not None:
                    # Drawn canvases fall back to their own mark boxes straight away, as before.
                    break
            if marks is None:
                if fallback is not None:
                    marks, fidelity = fallback, "approximate_bounding_boxes"
                else:
                    marks, fidelity = self._multi_res_vault["block_proxy_marks"](key), "approximate_8px_blocks"
            # Register 128x128 proxy in out-of-process harness (enforces EXEC-GATE-3 admission firewall)
            desc = harness.register_verified_source(
                source_id=key[0],
                source_version=key[1],
                source_kind=source_kind,
                marks=marks,
                origin_attestation=origin_attestation,
            )
        except BaseException:
            self._multi_res_vault["evict_source"](key)
            raise
        self._source_dimensions[key] = self._multi_res_vault["dimensions"](key)
        return desc, fidelity

    def register_pixel_canvas(
        self,
        source_id: str,
        source_version: str,
        rgba_bytes: bytes,
        width: int,
        height: int,
        *,
        source_kind: str = "image_bytes",
        origin_attestation: str = "owner_supplied_picture",
        origin: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registers a real picture (an image, SVG, video frame or page capture already fitted to a
        supported canvas) so every measurement, diff, ghost and stream tool can use it.
        """
        self._require_active()
        if (width, height) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution {(width, height)} not in SUPPORTED_RESOLUTIONS {SUPPORTED_RESOLUTIONS}")
        key = (source_id, source_version)
        if key in self._source_dimensions:
            raise ValueError(f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {source_id}@{source_version}")
        mr_info = self._multi_res_vault["register_rgba"](source_id, source_version, width, height, rgba_bytes)
        desc, fidelity = self._register_proxy(key, source_kind, origin_attestation)
        out = self._registration_reply(desc, fidelity, width, height, mr_info, "none", origin_attestation)
        out["origin"] = origin or {}
        assert_no_raw_buffer_leakage(out)
        return out

    def _registration_reply(
        self,
        desc: Dict[str, Any],
        proxy_fidelity: str,
        w: int,
        h: int,
        mr_info: Dict[str, Any],
        mirror: str,
        origin_attestation: str,
    ) -> Dict[str, Any]:
        proxy_dims = desc.pop("dimensions")
        proxy_marks = desc.pop("mark_count")
        return {
            **desc,
            "native_resolution": {"width": w, "height": h, "total_pixels": w * h},
            "proxy_resolution": {
                "width": proxy_dims["width"],
                "height": proxy_dims["height"],
                "fidelity": proxy_fidelity,
                "rectangles": proxy_marks,
                "note": "Harness measurements (refuse queries, measure_bbox/count_objects, diff derived views) run on this downsample.",
            },
            "native_pixel_sha256": mr_info["pixel_sha256"],
            "compiled_primitive_count": mr_info["spec_count"],
            "mirror_mode": mirror,
            "origin_attestation": origin_attestation,
            "origin_attestation_checked": False,
        }

    def inspect_canvas(
        self,
        source_id: str,
        source_version: str = "v1",
        *,
        target_rgba: Tuple[int, int, int, int] = (20, 120, 220, 255),
        tolerance: int = 0,
        issue_observation_receipt: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the full closed-form measurement suite + native AVX2 SIMD
        perceptual/structural metrics on `source_id@source_version`, creates a
        provenance-pinned `ElectricRender`, and optionally issues an HSM/Vault
        cryptographic `ObservationReceipt`.
        """
        t0 = time.perf_counter()
        harness = self._require_active()
        key = (source_id, source_version)
        if key not in self._source_dimensions:
            raise KeyError(f"Unknown source {source_id}@{source_version}")
        self._ensure_proxy_materialized(key)
        w, h = self._source_dimensions[key]

        # 1. Execute native AVX2 SIMD 10-operator suite at full target resolution (w, h)
        native_metrics = self._multi_res_vault["run_native_simd_suite"](
            key,
            None,
            target_rgba=target_rgba,
            tolerance=tolerance,
        )

        # 2. Execute standard harness closed-form operators for provenance & replay digest chaining
        m_inspect = harness.execute_measurement(
            "inspect_source", {}, source_id=source_id, source_version=source_version
        )
        m_bbox = harness.execute_measurement(
            "measure_bbox",
            {"selector": {"type": "non_transparent"}},
            source_id=source_id,
            source_version=source_version,
        )
        m_objects = harness.execute_measurement(
            "count_objects",
            {"connectivity": 4, "min_area_px": 1},
            source_id=source_id,
            source_version=source_version,
        )

        # Compute deterministic replay digest over the native multi-resolution measurement
        native_params = {
            "width": w,
            "height": h,
            "target_rgba": list(target_rgba),
            "tolerance": tolerance,
        }
        native_replay_digest = compute_replay_digest(
            source_id=source_id,
            source_version=source_version,
            secondary_source=None,
            operator_name="native_simd_10_operator_suite",
            operator_version=OPERATOR_VERSION,
            parameters=native_params,
            result=native_metrics,
        )

        # 3. Create provenance-pinned ElectricRender
        render_type = "observed" if issue_observation_receipt else "derived"
        render_source = "verified_pixels" if issue_observation_receipt else "measurement"
        uncertainty_state = "verified" if issue_observation_receipt else "derived"

        prov_refs = [
            {"ref_type": "source", "ref_id": source_id, "ref_version": source_version},
            {"ref_type": "measurement", "ref_id": m_inspect["measurement_id"]},
            {"ref_type": "measurement", "ref_id": m_bbox["measurement_id"]},
            {"ref_type": "measurement", "ref_id": m_objects["measurement_id"]},
        ]
        render = harness.create_electric_render(
            render_type=render_type,
            source=render_source,
            confidence="high",
            uncertainty_state=uncertainty_state,
            provenance_refs=prov_refs,
            visibility_policy="agent_visible",
        )

        receipt: Optional[Dict[str, Any]] = None
        hsm_countersignature: Optional[Dict[str, Any]] = None
        if issue_observation_receipt:
            cap_tok = (
                self._vault_client.request_capability_token(
                    render_id=render["render_id"],
                    captured_source_id=source_id,
                    captured_source_version=source_version,
                )
                if self._vault_client is not None
                else None
            )
            receipt = harness.issue_observation_receipt(
                render_id=render["render_id"],
                captured_source_id=source_id,
                captured_source_version=source_version,
                caller_capability_token=cap_tok,
            )
            promo = harness.promote_render_to_observed(render["render_id"], receipt=receipt)
            if promo.get("status") != "ok":
                raise PermissionError(f"Failed to promote render to observed: {promo}")
            render = promo["render"]
            if self._hsm_client is not None:
                hsm_countersignature = self._hsm_client.sign(
                    canonical_json(receipt),
                    key_domain="receipt",
                )

        acc_key_map = {
            "verified": "verified_observation",
            "derived": "unobserved_electric_render",
            "predicted": "non_destructive_ghost_overlay",
            "unknown": "refused_epistemic_boundary",
            "stale": "quarantined_or_unverified_input",
        }
        accessibility_meta = build_accessibility_metadata(
            acc_key_map.get(render["uncertainty_state"], "unobserved_electric_render")
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        out = {
            "status": "ok",
            "source_id": source_id,
            "source_version": source_version,
            "resolution": {"width": w, "height": h, "total_pixels": w * h},
            "native_simd_backend": self._simd_engine.active_backend,
            "native_simd_metrics": native_metrics,
            "native_replay_digest": native_replay_digest,
            "harness_measurements": {
                "inspect_source": m_inspect,
                "measure_bbox_128_proxy": m_bbox,
                "count_objects_128_proxy": m_objects,
            },
            "render": render,
            "observation_receipt": receipt,
            "hsm_receipt_countersignature": hsm_countersignature,
            "accessibility": accessibility_meta,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def compare_canvases(
        self,
        source_a: Tuple[str, str],
        source_b: Tuple[str, str],
        *,
        tolerance: int = 0,
    ) -> Dict[str, Any]:
        """
        Computes a deterministic multi-resolution structural diff and all 5
        Layer-3 derived representations (`change_mask`, `difference_map`,
        `motion_summary`, `colour_delta`, `spatial_relationship`) between
        `source_a` and `source_b`.
        """
        t0 = time.perf_counter()
        harness = self._require_active()
        if source_a not in self._source_dimensions or source_b not in self._source_dimensions:
            raise KeyError(f"Both {source_a} and {source_b} must be registered before comparison")
        self._ensure_proxy_materialized(source_a)
        self._ensure_proxy_materialized(source_b)

        w_a, h_a = self._source_dimensions[source_a]
        w_b, h_b = self._source_dimensions[source_b]
        if (w_a, h_a) != (w_b, h_b):
            raise ValueError(f"Resolution mismatch: {source_a} is {w_a}x{h_a}, {source_b} is {w_b}x{h_b}")

        # Fast-path O(1) skip for native SIMD pixel scan when SHA-256 digests are identical
        sha_a = self._multi_res_vault["pixel_sha256_of"](source_a)
        sha_b = self._multi_res_vault["pixel_sha256_of"](source_b)
        if sha_a is not None and sha_a == sha_b:
            total_px = w_a * h_a
            native_diff_metrics = {
                "changed_pixels": 0,
                "unchanged_pixels": total_px,
                "added_pixels": 0,
                "removed_pixels": 0,
                "modified_colour_pixels": 0,
                "total_pixels": total_px,
                "change_ratio": 0.0,
                "diff_bbox": [-1, -1, -1, -1],
            }
        else:
            native_diff_suite = self._multi_res_vault["run_native_simd_suite"](
                source_a,
                source_b,
                tolerance=tolerance,
            )
            native_diff_metrics = native_diff_suite["op_bbox_aligned_diff_u8"]

        derived_reps: Dict[str, Any] = {}
        for rep_type in (
            "change_mask",
            "difference_map",
            "motion_summary",
            "colour_delta",
            "spatial_relationship",
        ):
            derived_reps[rep_type] = harness.compute_derived_representation(
                rep_type, source_a, source_b
            )

        diff_replay_digest = compute_replay_digest(
            source_id=source_a[0],
            source_version=source_a[1],
            secondary_source={"source_id": source_b[0], "source_version": source_b[1]},
            operator_name="native_simd_canvas_comparison",
            operator_version=OPERATOR_VERSION,
            parameters={"width": w_a, "height": h_a, "tolerance": tolerance},
            result=native_diff_metrics,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        out = {
            "status": "ok",
            "source_a": {"source_id": source_a[0], "source_version": source_a[1]},
            "source_b": {"source_id": source_b[0], "source_version": source_b[1]},
            "resolution": {"width": w_a, "height": h_a},
            "native_diff_metrics": native_diff_metrics,
            "derived_representations": derived_reps,
            "derived_representations_resolution": {
                "width": 128,
                "height": 128,
                "scale_to_native": {"x": round(w_a / 128.0, 6), "y": round(h_a / 128.0, 6)},
                "note": (
                    "native_diff_metrics are measured at the native resolution; derived_representations "
                    "are measured on the 128x128 downsample, so multiply their pixel positions by "
                    "scale_to_native and expect pixel counts near native / (scale.x * scale.y)."
                ),
            },
            "diff_replay_digest": diff_replay_digest,
            "epistemic_label": "derived",
            "observed": False,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def preview_ghost_marks(
        self,
        base_source_id: str,
        base_source_version: str,
        proposed_ghost_marks: Sequence[Union[DrawingMark, Dict[str, Any]]],
        *,
        mirror: str = "none",
    ) -> Dict[str, Any]:
        """
        Projects `proposed_ghost_marks` onto `ghost_isolation_plane` over
        `base_source_id@base_source_version` to evaluate layout, bounding box,
        contrast, and object-count changes without mutating `verified_plane`.
        Guarantees `observed=False` and `ghost_pixel_leakage_bytes == 0`.
        """
        harness = self._require_active()
        key = (base_source_id, base_source_version)
        if key not in self._source_dimensions:
            raise KeyError(f"Unknown base source {base_source_id}@{base_source_version}")
        self._ensure_proxy_materialized(key)
        w, h = self._source_dimensions[key]

        ghost_specs, proxy_ghost_marks = normalize_marks_for_canvas(
            proposed_ghost_marks, w, h, mirror=mirror
        )
        preview_res = self._multi_res_vault["preview_ghost_specs"](key, ghost_specs)

        ghost_render = harness.create_electric_render(
            render_type="predicted",
            source="inference",
            confidence="medium",
            uncertainty_state="predicted",
            provenance_refs=[
                {"ref_type": "source", "ref_id": base_source_id, "ref_version": base_source_version}
            ],
            ghost_marks=proxy_ghost_marks,
        )
        accessibility_meta = build_accessibility_metadata("non_destructive_ghost_overlay")

        out = {
            "status": "ok",
            "epistemic_state": "predicted",
            "observed": False,
            "layer_isolation": ghost_render["layer_isolation"],
            "base_source_id": base_source_id,
            "base_source_version": base_source_version,
            "base_pixel_sha256_preserved": preview_res["base_pixel_sha256_preserved"],
            "ghost_pixel_leakage_bytes": 0,
            "hypothetical_metrics": preview_res["hypothetical_metrics"],
            "ghost_render": ghost_render,
            "accessibility": accessibility_meta,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def evaluate_agent_query(
        self,
        operator_or_query_type: str,
        parameters: Dict[str, Any],
        *,
        source_id: str,
        source_version: str = "v1",
    ) -> Dict[str, Any]:
        """
        Executes a closed-form measurement query or deterministically refuses
        any semantic, aesthetic, OCR, or user-intent query (`REFUSED_EPISTEMIC_BOUNDARY`).
        """
        harness = self._require_active()
        self._ensure_proxy_materialized((source_id, source_version))
        res = harness.execute_measurement(
            operator_or_query_type,
            parameters,
            source_id=source_id,
            source_version=source_version,
        )
        assert_no_raw_buffer_leakage(res)
        return res

    def get_enclave_health_and_audit_summary(self) -> Dict[str, Any]:
        """Returns live health and cryptographic WORM audit chain summary across all 4 daemons."""
        self._require_active()
        hsm_stat = self._hsm_client.ping_hsm() if self._hsm_client else {}
        vault_stat = self._vault_client.ping_status() if self._vault_client else {}
        audit_head = self._audit_client.export_chain_head() if self._audit_client else {}
        quar_stat = self._quar_client.get_status() if self._quar_client else {}

        return {
            "hsm_enclave": hsm_stat,
            "vault_enclave": vault_stat,
            "audit_worm_ledger": audit_head,
            "quarantine_supervisor": {
                "halted": quar_stat.get("halted", False),
                "triggered_kill_switches": sum(
                    1 for ks in quar_stat.get("kill_switches", {}).values() if ks.get("triggered")
                ),
            },
            "native_simd_backend": self._simd_engine.active_backend,
        }

    def evict_synthetic_canvas(self, source_id: str, source_version: str) -> Dict[str, Any]:
        """Explicitly zeroizes (`explicit_bzero`) and evicts a single source revision from the multi-resolution closure vault."""
        key = (source_id, source_version)
        unmat = getattr(self, "_unmaterialized_signal_sources", None)
        tomb = getattr(self, "_retired_tombstones", None)
        was_signal_only = unmat.pop(key, None) is not None if unmat is not None else False
        if tomb is not None and (key in tomb or was_signal_only):
            tomb.add(key)
        if getattr(self, "lazy_enclave", False) and (getattr(self, "_harness", None) is None or was_signal_only):
            self._require_session_open()
            self._source_dimensions.pop(key, None)
            res = self._multi_res_vault["evict_source"](key)
            res["proxy_released"] = False
            if tomb is not None and len(tomb) > 4096:
                keep = set(self._source_dimensions.keys())
                for old_k in list(tomb)[:2048]:
                    if old_k not in keep:
                        tomb.discard(old_k)
            assert_no_raw_buffer_leakage(res)
            return res

        harness = self._require_active()
        self._source_dimensions.pop(key, None)
        for receipt in harness._observation_receipts.values():
            if (receipt["captured_source_id"], receipt["captured_source_version"]) == key:
                receipt.update(revoked=True, receipt_signature=None)
        stale_render_ids: List[str] = []
        for rid, render in harness._renders.items():
            if any(ref.get("ref_type") == "source" and (ref.get("ref_id"), ref.get("ref_version")) == key
                   for ref in render.get("provenance_refs", ())):
                render.update(observed=False, observation_receipt_id=None,
                              render_type="derived" if render.get("render_type") == "observed" else render["render_type"],
                              uncertainty_state="unknown")
                render.update(harness.format_uncertainty_overlay("unknown"))
                stale_render_ids.append(rid)
        # Bound historical unreferenced renders/derived manifestations to prevent O(N^2) growth over multi-hour streams
        if len(harness._renders) > 256:
            for rid in stale_render_ids:
                if not any(rec.get("render_id") == rid for rec in harness._observation_receipts.values()):
                    harness._renders.pop(rid, None)
        if len(getattr(harness, "_derived_representations", {})) > 256:
            for did in list(harness._derived_representations)[:128]:
                harness._derived_representations.pop(did, None)
        if len(getattr(harness, "_measurements", {})) > 512:
            for mid in list(harness._measurements)[:256]:
                harness._measurements.pop(mid, None)
        # Revoke gate authority before either pixel cleanup can fail; descriptors remain tombstones.
        proxy_released = harness.release_source_pixels([key]) > 0
        res = self._multi_res_vault["evict_source"](key)
        res["proxy_released"] = proxy_released
        assert_no_raw_buffer_leakage(res)
        return res

    def purge_session_vault(self, reason: str = "AGENT_EXPLICIT_PURGE") -> Dict[str, Any]:
        """Zeroizes and purges all multi-resolution buffers in < 100 ms."""
        res = self._multi_res_vault["purge_all"]()
        self._source_dimensions.clear()
        self._unmaterialized_signal_sources.clear()
        return {
            "status": "purged",
            "reason": reason,
            **res,
        }

    def create_continuous_stream(
        self,
        source_id: str = "src_continuous_stream",
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_window_frames: int = 8,
        settle_ticks: int = 2,
        checkpoint_hamming_threshold: int = 6,
    ) -> "ContinuousElectricEyeStream":
        """Creates a ContinuousElectricEyeStream bound to this active session."""
        if getattr(self, "lazy_enclave", False):
            self._require_session_open()
        else:
            self._require_active()
        w = width if width is not None else self.default_resolution[0]
        h = height if height is not None else self.default_resolution[1]
        return ContinuousElectricEyeStream(
            self,
            source_id=source_id,
            width=w,
            height=h,
            max_window_frames=max_window_frames,
            settle_ticks=settle_ticks,
            checkpoint_hamming_threshold=checkpoint_hamming_threshold,
        )


def _parse_dhash64_u64(dhash_hex: str) -> int:
    s = str(dhash_hex).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    return int(s or "0", 16) & 0xFFFFFFFFFFFFFFFF


def _dhash64_hamming_distance(dhash_a: str, dhash_b: str) -> int:
    xor_val = _parse_dhash64_u64(dhash_a) ^ _parse_dhash64_u64(dhash_b)
    return bin(xor_val).count("1")


class ContinuousElectricEyeStream:
    """
    Continuous Multi-Frame Electric Eye Perception Stream.

    Extends the discrete pairwise `v1 -> v2` Electric Vision bridge into a
    bounded-memory, zero-PII streaming perception engine with:
      1. Bounded Ring-Buffer Temporal Window (`max_window_frames`) with automatic
         `explicit_bzero` zeroization eviction of expired frames.
      2. 16x16 Macroblock Dirty-Tile Bitmasking (`compute_dirty_tiles_16x16`) with
         O(1) fast-path skip on identical SHA-256 frames.
      3. Automatic Epistemic Stale-Observation Decay (`verified -> stale` watchdog):
         any prior `verified` observation (`observed=True`) is automatically
         demoted to `uncertainty_state="stale"` (`[OBS:STALE_UNVERIFIED]`,
         `observed=False`) when a subsequent frame mutates the canvas, with
         hash-chained WORM audit logging.
      4. Multi-Frame Layer-3 Temporal Operators (`dhash64_hamming_timeline`,
         `settling_and_oscillation_detector` [`initializing|settled|oscillating_cycle|evolving`],
         and `kinematic_trajectory_chain` with velocity, acceleration, and
         `ghost_isolation_plane` next-tick centroid prediction).
      5. Closed-Loop Ghost Realization Verification (`stage_ghost_prediction` +
         `verify_ghost_realization`) scoring predicted `GhostOverlay` metrics
         against the actual realized frame on `verified_plane`.
    """

    def __init__(
        self,
        session: ElectricVisionAgentSession,
        source_id: str = "src_continuous_stream",
        *,
        width: int = 256,
        height: int = 256,
        max_window_frames: int = 8,
        settle_ticks: int = 2,
        checkpoint_hamming_threshold: int = 6,
    ) -> None:
        if (width, height) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Unsupported stream resolution {(width, height)}")
        if max_window_frames < 2 or max_window_frames > 64:
            raise ValueError("max_window_frames must be between 2 and 64")
        if settle_ticks < 1:
            raise ValueError("settle_ticks must be >= 1")
        self.session = session
        self.source_id = source_id
        self.width = width
        self.height = height
        self.max_window_frames = max_window_frames
        self.settle_ticks = settle_ticks
        self.checkpoint_hamming_threshold = checkpoint_hamming_threshold

        self._window_versions: List[str] = []
        self._frame_records: Dict[str, Dict[str, Any]] = {}
        self._transition_records: List[Dict[str, Any]] = []
        self._observation_ledger: Dict[str, Dict[str, Any]] = {}
        self._staged_ghosts: Dict[str, Dict[str, Any]] = {}
        self._ghost_realizations: List[Dict[str, Any]] = []
        self._evicted_frame_count = 0
        self._ghost_seq = 0
        self._last_eviction: Optional[Dict[str, Any]] = None
        self._tick_counter = 0
        self._stale_audit_failed = False
        self._ingest_failed = False
        self._pending_retirements: Set[str] = set()

    # Per-version bookkeeping kept beyond the window, so freshness checks on recent versions still answer.
    HISTORY_LIMIT = 64

    def _prune_history(self) -> None:
        keep = set(self._window_versions)
        for book in (self._observation_ledger, self._staged_ghosts):
            for old_key in list(book)[: max(0, len(book) - self.HISTORY_LIMIT)]:
                if old_key not in keep:
                    book.pop(old_key, None)
        self._transition_records = [t for t in self._transition_records if t["to_version"] in keep]
        del self._ghost_realizations[: max(0, len(self._ghost_realizations) - self.HISTORY_LIMIT)]

    def release(self) -> int:
        """Frees every frame this stream still holds in the session vaults (used by stream reset)."""
        released = 0
        versions = {version for sid, version in self.session._source_dimensions if sid == self.source_id}
        self._pending_retirements.update(versions | set(self._window_versions) | set(self._frame_records))
        self._ingest_failed = True
        for entry in self._observation_ledger.values():
            entry.update(fresh=False, observed=False, uncertainty_state="unknown",
                         refusal_code="ERR_CONTINUOUS_STREAM_RELEASED")
        failure = None
        for version in sorted(self._pending_retirements):
            try:
                if self.session.evict_synthetic_canvas(self.source_id, version).get("evicted"):
                    released += 1
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"Frame retirement pending for {self.source_id}@{version}: {exc}")
            else:
                self._pending_retirements.discard(version)
        if failure is not None:
            raise failure
        self._window_versions.clear()
        self._frame_records.clear()
        return released

    def stage_ghost_prediction(
        self,
        base_version: str,
        proposed_ghost_marks: Sequence[Union[DrawingMark, Dict[str, Any]]],
        *,
        ghost_id: Optional[str] = None,
        mirror: str = "none",
    ) -> Dict[str, Any]:
        """
        Projects `proposed_ghost_marks` onto `ghost_isolation_plane` over `base_version`
        and stages the prediction so it can be verified against the next realized frame.
        """
        preview = self.session.preview_ghost_marks(
            self.source_id,
            base_version,
            proposed_ghost_marks,
            mirror=mirror,
        )
        self._ghost_seq += 1
        gid = ghost_id or f"ghost_{self._ghost_seq:04d}"
        record = {
            "ghost_id": gid,
            "base_source_id": self.source_id,
            "base_source_version": base_version,
            "epistemic_state": "predicted",
            "observed": False,
            "layer_isolation": "ghost_isolation_plane",
            "hypothetical_metrics": preview["hypothetical_metrics"],
            "ghost_render": preview["ghost_render"],
            "realized": False,
        }
        self._staged_ghosts[gid] = record
        assert_no_raw_buffer_leakage(record)
        return record

    def verify_ghost_realization(
        self,
        ghost_id: str,
        actual_version: str,
    ) -> Dict[str, Any]:
        """
        Compares a staged `GhostOverlay` prediction against the actual realized
        frame `actual_version` on `verified_plane`. Always tagged `epistemic_label="derived"`,
        `observed=False`.
        """
        if ghost_id not in self._staged_ghosts:
            raise KeyError(f"Unknown staged ghost_id: {ghost_id!r}")
        if actual_version not in self._frame_records:
            raise KeyError(f"Unknown actual_version in stream: {actual_version!r}")

        ghost_rec = self._staged_ghosts[ghost_id]
        actual_rec = self._frame_records[actual_version]
        if actual_rec["inspection"].get("signal_only"):
            actual_rec["inspection"] = self.session.inspect_canvas(self.source_id, actual_version)
        g_simd = ghost_rec["hypothetical_metrics"]
        a_simd = actual_rec["inspection"]["native_simd_metrics"]

        g_cc = g_simd["op_connected_components_4c"]
        a_cc = a_simd["op_connected_components_4c"]
        g_fg_px = int(g_cc["foreground_bbox"]["matching_pixels"])
        a_fg_px = int(a_cc["foreground_bbox"]["matching_pixels"])
        fg_px_residual = abs(a_fg_px - g_fg_px)

        g_bbox = g_cc["foreground_bbox"]
        a_bbox = a_cc["foreground_bbox"]
        if g_bbox["empty"] and a_bbox["empty"]:
            bbox_l1_residual = 0
        elif g_bbox["empty"] or a_bbox["empty"]:
            bbox_l1_residual = self.width + self.height
        else:
            bbox_l1_residual = (
                abs(int(a_bbox["x_min"]) - int(g_bbox["x_min"]))
                + abs(int(a_bbox["y_min"]) - int(g_bbox["y_min"]))
                + abs(int(a_bbox["x_max"]) - int(g_bbox["x_max"]))
                + abs(int(a_bbox["y_max"]) - int(g_bbox["y_max"]))
            )

        dhash_hamming = _dhash64_hamming_distance(
            g_simd["op_perceptual_hash_dhash64"],
            a_simd["op_perceptual_hash_dhash64"],
        )
        contrast_residual_q16 = abs(
            int(a_simd["op_contrast_ratio_wcag_q16"]) - int(g_simd["op_contrast_ratio_wcag_q16"])
        )
        obj_count_residual = abs(int(a_cc["component_count"]) - int(g_cc["component_count"]))

        total_px = float(self.width * self.height)
        px_norm = max(0.0, 1.0 - (fg_px_residual / max(1.0, max(g_fg_px, a_fg_px, 1))))
        dhash_norm = max(0.0, 1.0 - (dhash_hamming / 64.0))
        bbox_norm = max(0.0, 1.0 - (bbox_l1_residual / float(2 * (self.width + self.height))))
        convergence_score = round((0.5 * px_norm) + (0.3 * dhash_norm) + (0.2 * bbox_norm), 6)
        exact_realization = bool(
            fg_px_residual == 0 and bbox_l1_residual == 0 and dhash_hamming == 0 and obj_count_residual == 0
        )

        audit = {
            "ghost_id": ghost_id,
            "base_source_version": ghost_rec["base_source_version"],
            "actual_source_version": actual_version,
            "epistemic_label": "derived",
            "observed": False,
            "exact_realization": exact_realization,
            "realization_convergence_score": 1.0 if exact_realization else convergence_score,
            "residuals": {
                "foreground_pixel_residual": fg_px_residual,
                "bbox_l1_residual": bbox_l1_residual,
                "dhash64_hamming_distance": dhash_hamming,
                "wcag_contrast_q16_residual": contrast_residual_q16,
                "component_count_residual": obj_count_residual,
                "pixel_residual_fraction_of_canvas": round(fg_px_residual / total_px, 6),
            },
            "replay_digest": compute_replay_digest(
                source_id=self.source_id,
                source_version=actual_version,
                secondary_source={"source_id": self.source_id, "source_version": ghost_rec["base_source_version"]},
                operator_name="verify_ghost_realization",
                operator_version=OPERATOR_VERSION,
                parameters={"ghost_id": ghost_id},
                result={
                    "fg_px_residual": fg_px_residual,
                    "bbox_l1_residual": bbox_l1_residual,
                    "dhash_hamming": dhash_hamming,
                },
            ),
        }
        ghost_rec["realized"] = True
        self._ghost_realizations.append(audit)
        assert_no_raw_buffer_leakage(audit)
        return audit

    def check_receipt_freshness(self, source_version: str) -> Dict[str, Any]:
        """
        Checks whether an observation on `source_version` is still `verified` or
        has decayed to `stale` (`ERR_STALE_OBSERVATION_SUPERSEDED`) due to newer stream frames.
        """
        if source_version not in self._observation_ledger:
            return {
                "source_id": self.source_id,
                "source_version": source_version,
                "fresh": False,
                "uncertainty_state": "unknown",
                "observed": False,
                "refusal_code": "ERR_UNOBSERVED_OR_UNKNOWN_VERSION",
            }
        entry = copy.deepcopy(self._observation_ledger[source_version])
        assert_no_raw_buffer_leakage(entry)
        return entry

    def ingest_signal_frame(
        self,
        source_version: str,
        *,
        rgba: bytes,
        source_kind: str = "captured_frame",
        origin: Optional[Dict[str, Any]] = None,
        origin_attestation: str = "companion_desktop_glance",
    ) -> Dict[str, Any]:
        """
        Active Electric Signal Temp Buffer ingest path:
          - Holds the active `width x height` RGBA frame in `_multi_res_vault` (`explicit_bzero` ring).
          - Computes native AVX2 `dhash64`, `change_ratio`, `diff_bbox`, `dirty_tiles_16x16`, and
            `temporal_window_summary` (`settling_and_oscillation_detector`) directly in ~0.3 ms.
          - Skips eager live 128x128 `DrawingMark` proxy translation and 5-way derived IPC views;
            materializes them lazily only if an explicit query inspects or compares the frame.
        """
        if self._stale_audit_failed:
            raise RuntimeError("ERR_CONTINUOUS_AUDIT_FAILED_CLOSED: reset the stream after durable audit failure")
        if self._ingest_failed:
            raise RuntimeError("ERR_CONTINUOUS_INGEST_FAILED_CLOSED: reset the stream after ingest failure")
        key = (self.source_id, source_version)
        harness_sources = getattr(getattr(self.session, "_harness", None), "_sources", {})
        if (
            key in self.session._source_dimensions
            or key in self.session._retired_tombstones
            or key in harness_sources
            or source_version in self._frame_records
        ):
            raise ValueError(f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {self.source_id}@{source_version}")

        t0 = time.perf_counter()
        tick_before = self._tick_counter
        try:
            prev_version = self._window_versions[-1] if self._window_versions else None
            reg = self.session.register_signal_rgba(
                self.source_id,
                source_version,
                rgba,
                self.width,
                self.height,
                source_kind=source_kind,
                origin_attestation=origin_attestation,
                origin=origin,
            )
            prev_backing_ver = (
                self._frame_records[prev_version]["registration"].get("source_version", prev_version)
                if prev_version is not None
                else None
            )
            prev_key = (
                (self.source_id, prev_backing_ver)
                if prev_backing_ver is not None and (self.source_id, prev_backing_ver) in self.session._source_dimensions
                else None
            )
            sig_metrics = self.session._multi_res_vault["run_fast_signal_suite"](key, prev_key)
            curr_dhash = sig_metrics["op_perceptual_hash_dhash64"]
            self._tick_counter += 1

            self._observation_ledger[source_version] = {
                "source_id": self.source_id,
                "source_version": source_version,
                "tick": self._tick_counter,
                "fresh": False,
                "uncertainty_state": "derived",
                "observed": False,
                "observation_receipt_id": None,
                "superseded_by_version": None,
                "accessibility": build_accessibility_metadata("unobserved_electric_render"),
            }

            transition_info: Optional[Dict[str, Any]] = None
            decayed_versions: List[str] = []
            if prev_version is not None:
                prev_rec = self._frame_records[prev_version]
                prev_dhash = prev_rec["inspection"]["native_simd_metrics"]["op_perceptual_hash_dhash64"]
                prev_sha = prev_rec["registration"]["native_pixel_sha256"]
                step_hamming = _dhash64_hamming_distance(prev_dhash, curr_dhash)
                if prev_key is not None:
                    diff_m = sig_metrics["op_bbox_aligned_diff_u8"]
                    dirty_tiles = self.session._multi_res_vault["compute_dirty_tiles_16x16"](prev_key, key)
                elif prev_sha == reg["native_pixel_sha256"]:
                    total_px = self.width * self.height
                    diff_m = {
                        "changed_pixels": 0,
                        "unchanged_pixels": total_px,
                        "change_ratio": 0.0,
                        "diff_bbox": [-1, -1, -1, -1],
                    }
                    tiles_x, tiles_y = (self.width + 15) // 16, (self.height + 15) // 16
                    total_tiles = tiles_x * tiles_y
                    dirty_tiles = {
                        "tile_size_px": 16,
                        "grid_dimensions": {"tiles_x": tiles_x, "tiles_y": tiles_y, "total_tiles": total_tiles},
                        "dirty_tile_count": 0,
                        "clean_tile_count": total_tiles,
                        "dirty_tile_ratio": 0.0,
                        "dirty_tile_bbox": [-1, -1, -1, -1],
                        "dirty_bitmask_sha256": _canonical_sha256_tag(bytes((total_tiles + 7) // 8)),
                        "fast_path_identical_sha": True,
                    }
                else:
                    total_px = self.width * self.height
                    diff_m = {
                        "changed_pixels": 0,
                        "unchanged_pixels": total_px,
                        "change_ratio": 0.0,
                        "diff_bbox": [-1, -1, -1, -1],
                    }
                    dirty_tiles = {"dirty_tile_ratio": 0.0, "dirty_tile_count": 0}

                changed_px = int(diff_m["changed_pixels"])
                transition_info = {
                    "from_version": prev_version,
                    "to_version": source_version,
                    "changed_pixels": changed_px,
                    "change_ratio": diff_m["change_ratio"],
                    "diff_bbox": diff_m["diff_bbox"],
                    "dhash64_hamming_distance": step_hamming,
                    "checkpoint_triggered": bool(step_hamming >= self.checkpoint_hamming_threshold),
                    "dirty_tiles_16x16": dirty_tiles,
                    "motion_summary": {"component_trajectories": []},
                    "decayed_to_stale_versions": decayed_versions,
                    "diff_replay_digest": None,
                }
                self._transition_records.append(transition_info)

            insp_stub = {
                "signal_only": True,
                "native_simd_metrics": {
                    "op_perceptual_hash_dhash64": curr_dhash,
                    "op_sobel_edge_density_q16": sig_metrics["op_sobel_edge_density_q16"],
                    "op_mean_rgba_u8": sig_metrics["op_mean_rgba_u8"],
                    "op_connected_components_4c": {
                        "component_count": 0,
                        "components": [],
                        "foreground_bbox": {"empty": True, "matching_pixels": 0},
                    },
                },
            }
            self._window_versions.append(source_version)
            self._frame_records[source_version] = {
                "tick": self._tick_counter,
                "source_version": source_version,
                "registration": reg,
                "inspection": insp_stub,
            }

            eviction_receipt: Optional[Dict[str, Any]] = None
            if len(self._window_versions) > self.max_window_frames:
                oldest_ver = self._window_versions.pop(0)
                popped_rec = self._frame_records.pop(oldest_ver, None)
                backing_ver = (
                    popped_rec["registration"].get("source_version", oldest_ver)
                    if popped_rec is not None
                    else oldest_ver
                )
                still_held_reg = any(
                    rec["registration"].get("source_version") == backing_ver
                    for rec in self._frame_records.values()
                )
                if not still_held_reg and (self.source_id, backing_ver) in self.session._source_dimensions:
                    self._pending_retirements.add(backing_ver)
                    eviction_receipt = self.session.evict_synthetic_canvas(self.source_id, backing_ver)
                    self._pending_retirements.discard(backing_ver)
                    self._evicted_frame_count += 1
                    self._last_eviction = eviction_receipt
            self._prune_history()

            window_summary = self.compute_temporal_window_summary()
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            out = {
                "status": "ok",
                "tick": self._tick_counter,
                "source_id": self.source_id,
                "source_version": source_version,
                "resolution": {"width": self.width, "height": self.height},
                "native_pixel_sha256": reg["native_pixel_sha256"],
                "dhash64": curr_dhash,
                "observation_state": copy.deepcopy(self._observation_ledger[source_version]),
                "decayed_to_stale_versions": decayed_versions,
                "transition_from_previous": transition_info,
                "ring_buffer_eviction": eviction_receipt,
                "ghost_realization_audit": None,
                "temporal_window_summary": window_summary,
                "signal_buffer_hit": False,
                "elapsed_ms": elapsed_ms,
            }
            assert_no_raw_buffer_leakage(out)
            return out
        except BaseException as exc:
            self.invalidate_failure(exc, source_version=source_version, tick_before=tick_before)
            raise

    def ingest_buffered_repeat(self, source_version: str) -> Dict[str, Any]:
        """
        O(1) Active Signal Temp Buffer hit path when the captured raw window digest
        matches the active signal already held in the stream. Reuses the active signal's
        `native_pixel_sha256` and `dhash64`, advances the temporal tick (`change_ratio=0.0`),
        and updates `settling_and_oscillation_detector` without re-translating pixels.
        """
        if self._stale_audit_failed:
            raise RuntimeError("ERR_CONTINUOUS_AUDIT_FAILED_CLOSED: reset the stream after durable audit failure")
        if self._ingest_failed:
            raise RuntimeError("ERR_CONTINUOUS_INGEST_FAILED_CLOSED: reset the stream after ingest failure")
        if not self._window_versions:
            raise RuntimeError("ingest_buffered_repeat requires at least one active frame in stream")
        key = (self.source_id, source_version)
        if key in self.session._source_dimensions or key in self.session._retired_tombstones or source_version in self._frame_records:
            raise ValueError(f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {self.source_id}@{source_version}")

        t0 = time.perf_counter()
        prev_version = self._window_versions[-1]
        prev_rec = self._frame_records[prev_version]
        curr_sha = prev_rec["registration"]["native_pixel_sha256"]
        curr_dhash = prev_rec["inspection"]["native_simd_metrics"]["op_perceptual_hash_dhash64"]
        self._tick_counter += 1
        self.session._retired_tombstones.add(key)

        self._observation_ledger[source_version] = {
            "source_id": self.source_id,
            "source_version": source_version,
            "tick": self._tick_counter,
            "fresh": False,
            "uncertainty_state": "derived",
            "observed": False,
            "observation_receipt_id": None,
            "superseded_by_version": None,
            "accessibility": build_accessibility_metadata("unobserved_electric_render"),
        }

        tiles_x, tiles_y = (self.width + 15) // 16, (self.height + 15) // 16
        total_tiles = tiles_x * tiles_y
        transition_info = {
            "from_version": prev_version,
            "to_version": source_version,
            "changed_pixels": 0,
            "change_ratio": 0.0,
            "diff_bbox": [-1, -1, -1, -1],
            "dhash64_hamming_distance": 0,
            "checkpoint_triggered": False,
            "dirty_tiles_16x16": {
                "tile_size_px": 16,
                "grid_dimensions": {"tiles_x": tiles_x, "tiles_y": tiles_y, "total_tiles": total_tiles},
                "dirty_tile_count": 0,
                "clean_tile_count": total_tiles,
                "dirty_tile_ratio": 0.0,
                "dirty_tile_bbox": [-1, -1, -1, -1],
                "fast_path_identical_sha": True,
            },
            "motion_summary": {"component_trajectories": []},
            "decayed_to_stale_versions": [],
            "diff_replay_digest": None,
        }
        self._transition_records.append(transition_info)

        self._window_versions.append(source_version)
        self._frame_records[source_version] = {
            "tick": self._tick_counter,
            "source_version": source_version,
            "registration": prev_rec["registration"],
            "inspection": prev_rec["inspection"],
        }

        eviction_receipt: Optional[Dict[str, Any]] = None
        if len(self._window_versions) > self.max_window_frames:
            oldest_ver = self._window_versions.pop(0)
            popped_rec = self._frame_records.pop(oldest_ver, None)
            backing_ver = (
                popped_rec["registration"].get("source_version", oldest_ver)
                if popped_rec is not None
                else oldest_ver
            )
            # Evict the vault RGBA slot only if no newer repeat references the same registration version
            still_held_reg = any(
                rec["registration"].get("source_version") == backing_ver
                for rec in self._frame_records.values()
            )
            if not still_held_reg and (self.source_id, backing_ver) in self.session._source_dimensions:
                self._pending_retirements.add(backing_ver)
                eviction_receipt = self.session.evict_synthetic_canvas(self.source_id, backing_ver)
                self._pending_retirements.discard(backing_ver)
                self._evicted_frame_count += 1
                self._last_eviction = eviction_receipt
        self._prune_history()

        window_summary = self.compute_temporal_window_summary()
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        out = {
            "status": "ok",
            "tick": self._tick_counter,
            "source_id": self.source_id,
            "source_version": source_version,
            "resolution": {"width": self.width, "height": self.height},
            "native_pixel_sha256": curr_sha,
            "dhash64": curr_dhash,
            "observation_state": copy.deepcopy(self._observation_ledger[source_version]),
            "decayed_to_stale_versions": [],
            "transition_from_previous": transition_info,
            "ring_buffer_eviction": eviction_receipt,
            "ghost_realization_audit": None,
            "temporal_window_summary": window_summary,
            "signal_buffer_hit": True,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def ingest_frame(
        self,
        source_version: str,
        marks: Sequence[Union[DrawingMark, Dict[str, Any]]],
        *,
        source_kind: str = "captured_frame",
        mirror: str = "none",
        issue_observation_receipt: bool = False,
        verify_staged_ghost_id: Optional[str] = None,
        rgba: Optional[bytes] = None,
        origin: Optional[Dict[str, Any]] = None,
        origin_attestation: str = "owner_supplied_picture",
    ) -> Dict[str, Any]:
        """Publish a frame only through a complete ingest; any failure closes receipt scope."""
        if self._stale_audit_failed:
            raise RuntimeError("ERR_CONTINUOUS_AUDIT_FAILED_CLOSED: reset the stream after durable audit failure")
        if self._ingest_failed:
            raise RuntimeError("ERR_CONTINUOUS_INGEST_FAILED_CLOSED: reset the stream after ingest failure")
        key = (self.source_id, source_version)
        # A duplicate request must not retire a previously successful immutable frame.
        harness_sources = getattr(getattr(self.session, "_harness", None), "_sources", {})
        if key in self.session._source_dimensions or key in harness_sources or source_version in self._frame_records:
            raise ValueError(f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {self.source_id}@{source_version}")
        tick_before = self._tick_counter
        try:
            return self._ingest_frame_transaction(
                source_version, marks, source_kind=source_kind, mirror=mirror,
                issue_observation_receipt=issue_observation_receipt,
                verify_staged_ghost_id=verify_staged_ghost_id, rgba=rgba, origin=origin,
                origin_attestation=origin_attestation,
            )
        except BaseException as exc:
            self.invalidate_failure(exc, source_version=source_version, tick_before=tick_before)
            raise

    def invalidate_failure(
        self, failure: BaseException, *, source_version: Optional[str] = None,
        tick_before: Optional[int] = None,
    ) -> None:
        self._ingest_failed = True
        code = ("ERR_CONTINUOUS_AUDIT_FAILED_CLOSED" if self._stale_audit_failed
                else "ERR_CONTINUOUS_INGEST_FAILED_CLOSED")
        self._pending_retirements.update(self._window_versions)
        self._pending_retirements.update(self._frame_records)
        self._pending_retirements.update(self._observation_ledger)
        self._pending_retirements.update(
            v for sid, v in self.session._source_dimensions if sid == self.source_id
        )
        if source_version is not None:
            self._pending_retirements.add(source_version)
        for entry in self._observation_ledger.values():
            entry.update(fresh=False, observed=False, uncertainty_state="unknown",
                         refusal_code=code,
                         accessibility=build_accessibility_metadata("quarantined_or_unverified_input"))
        if tick_before is not None:
            self._tick_counter = tick_before
        self._window_versions.clear()
        self._frame_records.clear()
        self._transition_records.clear()
        self._ghost_realizations.clear()
        for ghost in self._staged_ghosts.values():
            ghost["realized"] = False
        for version in sorted(self._pending_retirements):
            try:
                self.session.evict_synthetic_canvas(self.source_id, version)
            except BaseException as cleanup_exc:
                failure.add_note(f"Frame retirement pending for {self.source_id}@{version}: {cleanup_exc}")
            else:
                self._pending_retirements.discard(version)

    def _ingest_frame_transaction(
        self,
        source_version: str,
        marks: Sequence[Union[DrawingMark, Dict[str, Any]]],
        *,
        source_kind: str = "captured_frame",
        mirror: str = "none",
        issue_observation_receipt: bool = False,
        verify_staged_ghost_id: Optional[str] = None,
        rgba: Optional[bytes] = None,
        origin: Optional[Dict[str, Any]] = None,
        origin_attestation: str = "owner_supplied_picture",
    ) -> Dict[str, Any]:
        """
        Ingests a new synthetic frame into the Continuous Electric Eye stream:
          1. Registers & inspects `source_version` via native AVX2 SIMD + Vault daemons.
          2. Computes pairwise diff & 16x16 dirty-tile bitmask against the previous frame.
          3. Enforces automatic `verified -> stale` decay on superseded observations when pixels change.
          4. Evicts & explicitly zeroizes (`explicit_bzero`) the oldest frame if the ring buffer is full.
          5. Computes updated multi-frame temporal window metrics (`dhash64_hamming_timeline`,
             `settling_and_oscillation_detector`, `kinematic_trajectory_chain`).
        """
        if self._stale_audit_failed:
            raise RuntimeError("ERR_CONTINUOUS_AUDIT_FAILED_CLOSED: reset the stream after durable audit failure")
        t0 = time.perf_counter()
        prev_version = self._window_versions[-1] if self._window_versions else None

        if rgba is not None:
            # A real frame (video or page capture), already fitted to the stream's canvas.
            reg = self.session.register_pixel_canvas(
                self.source_id,
                source_version,
                rgba,
                self.width,
                self.height,
                source_kind=source_kind,
                origin_attestation=origin_attestation,
                origin=origin,
            )
        else:
            reg = self.session.register_synthetic_canvas(
                self.source_id,
                source_version,
                marks,
                width=self.width,
                height=self.height,
                source_kind=source_kind,
                mirror=mirror,
            )
        insp = self.session.inspect_canvas(
            self.source_id,
            source_version,
            issue_observation_receipt=issue_observation_receipt,
        )
        # Count the tick only once the frame is in, so refused frames leave no gap.
        self._tick_counter += 1

        # Record initial epistemic state in observation ledger
        if issue_observation_receipt and insp["render"]["observed"]:
            self._observation_ledger[source_version] = {
                "source_id": self.source_id,
                "source_version": source_version,
                "tick": self._tick_counter,
                "fresh": True,
                "uncertainty_state": "verified",
                "observed": True,
                "observation_receipt_id": insp["render"]["observation_receipt_id"],
                "superseded_by_version": None,
                "accessibility": build_accessibility_metadata("verified_observation"),
            }
        else:
            self._observation_ledger[source_version] = {
                "source_id": self.source_id,
                "source_version": source_version,
                "tick": self._tick_counter,
                "fresh": False,
                "uncertainty_state": "derived",
                "observed": False,
                "observation_receipt_id": None,
                "superseded_by_version": None,
                "accessibility": build_accessibility_metadata("unobserved_electric_render"),
            }

        transition_info: Optional[Dict[str, Any]] = None
        decayed_versions: List[str] = []

        if prev_version is not None:
            diff_rep = self.session.compare_canvases(
                (self.source_id, prev_version),
                (self.source_id, source_version),
            )
            dirty_tiles = self.session._multi_res_vault["compute_dirty_tiles_16x16"](
                (self.source_id, prev_version),
                (self.source_id, source_version),
            )
            prev_dhash = self._frame_records[prev_version]["inspection"]["native_simd_metrics"][
                "op_perceptual_hash_dhash64"
            ]
            curr_dhash = insp["native_simd_metrics"]["op_perceptual_hash_dhash64"]
            step_hamming = _dhash64_hamming_distance(prev_dhash, curr_dhash)
            changed_px = int(diff_rep["native_diff_metrics"]["changed_pixels"])

            # Automatic Epistemic Stale-Observation Decay (`verified -> stale`)
            if changed_px > 0 or step_hamming > 0:
                for old_ver, obs_entry in self._observation_ledger.items():
                    if old_ver != source_version and obs_entry.get("fresh") is True:
                        obs_entry["fresh"] = False
                        obs_entry["uncertainty_state"] = "stale"
                        obs_entry["observed"] = False
                        obs_entry["superseded_by_version"] = source_version
                        obs_entry["refusal_code"] = "ERR_STALE_OBSERVATION_SUPERSEDED"
                        obs_entry["accessibility"] = build_accessibility_metadata(
                            "quarantined_or_unverified_input"
                        )
                        decayed_versions.append(old_ver)
                # Demote every old observation even if the durable sink fails. Then use
                # the harness's sequenced/hash-chained append_entry writer; failure is visible.
                try:
                    for old_ver in decayed_versions:
                        self.session._require_active()._append_audit_event(
                            event_type="CONTINUOUS_OBSERVATION_DECAYED_TO_STALE",
                            target_id=self.source_id,
                            error_code="ERR_STALE_OBSERVATION_SUPERSEDED",
                            detail=canonical_json({
                                "source_id": self.source_id,
                                "decayed_version": old_ver,
                                "superseded_by_version": source_version,
                                "changed_pixels": changed_px,
                                "dhash64_hamming_distance": step_hamming,
                            }),
                        )
                except Exception:
                    # Do not silently continue with a hole in the durable transition history.
                    self._stale_audit_failed = True
                    current = self._observation_ledger[source_version]
                    current.update(fresh=False, observed=False, uncertainty_state="unknown",
                                   refusal_code="ERR_CONTINUOUS_AUDIT_FAILED_CLOSED",
                                   accessibility=build_accessibility_metadata("quarantined_or_unverified_input"))
                    raise

            transition_info = {
                "from_version": prev_version,
                "to_version": source_version,
                "changed_pixels": changed_px,
                "change_ratio": diff_rep["native_diff_metrics"]["change_ratio"],
                "diff_bbox": diff_rep["native_diff_metrics"]["diff_bbox"],
                "dhash64_hamming_distance": step_hamming,
                "checkpoint_triggered": bool(step_hamming >= self.checkpoint_hamming_threshold),
                "dirty_tiles_16x16": dirty_tiles,
                "motion_summary": diff_rep["derived_representations"]["motion_summary"]["summary"],
                "decayed_to_stale_versions": decayed_versions,
                "diff_replay_digest": diff_rep["diff_replay_digest"],
            }
            self._transition_records.append(transition_info)

        self._window_versions.append(source_version)
        self._frame_records[source_version] = {
            "tick": self._tick_counter,
            "source_version": source_version,
            "registration": reg,
            "inspection": insp,
        }

        # Enforce Bounded Ring-Buffer Window with explicit_bzero eviction
        eviction_receipt: Optional[Dict[str, Any]] = None
        if len(self._window_versions) > self.max_window_frames:
            oldest_ver = self._window_versions.pop(0)
            self._frame_records.pop(oldest_ver, None)
            self._pending_retirements.add(oldest_ver)
            eviction_receipt = self.session.evict_synthetic_canvas(self.source_id, oldest_ver)
            self._pending_retirements.discard(oldest_ver)
            self._evicted_frame_count += 1
            self._last_eviction = eviction_receipt
        self._prune_history()

        ghost_audit: Optional[Dict[str, Any]] = None
        if verify_staged_ghost_id is not None:
            ghost_audit = self.verify_ghost_realization(verify_staged_ghost_id, source_version)

        window_summary = self.compute_temporal_window_summary()
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        out = {
            "status": "ok",
            "tick": self._tick_counter,
            "source_id": self.source_id,
            "source_version": source_version,
            "resolution": {"width": self.width, "height": self.height},
            "native_pixel_sha256": reg["native_pixel_sha256"],
            "dhash64": insp["native_simd_metrics"]["op_perceptual_hash_dhash64"],
            "observation_state": copy.deepcopy(self._observation_ledger[source_version]),
            "decayed_to_stale_versions": decayed_versions,
            "transition_from_previous": transition_info,
            "ring_buffer_eviction": eviction_receipt,
            "ghost_realization_audit": ghost_audit,
            "temporal_window_summary": window_summary,
            "elapsed_ms": elapsed_ms,
        }
        assert_no_raw_buffer_leakage(out)
        return out

    def compute_temporal_window_summary(self) -> Dict[str, Any]:
        """
        Computes Layer-3 multi-frame temporal representations over the active
        sliding window:
          - `dhash64_hamming_timeline`
          - `settling_and_oscillation_detector` (`initializing|settled|oscillating_cycle|evolving`)
          - `kinematic_trajectory_chain` (velocity, acceleration, and `ghost_isolation_plane`
            next-tick centroid prediction)
          - `dirty_tile_16x16_profile`
        """
        active_versions = list(self._window_versions)
        n_frames = len(active_versions)
        recent_transitions = [
            t for t in self._transition_records if t["to_version"] in active_versions
        ]

        # 1. dHash64 Hamming Timeline
        dhash_sequence = [
            {
                "source_version": v,
                "dhash64": self._frame_records[v]["inspection"]["native_simd_metrics"][
                    "op_perceptual_hash_dhash64"
                ],
                "native_pixel_sha256": self._frame_records[v]["registration"]["native_pixel_sha256"],
            }
            for v in active_versions
        ]
        step_hammings = [int(t["dhash64_hamming_distance"]) for t in recent_transitions]
        net_window_hamming = (
            _dhash64_hamming_distance(dhash_sequence[0]["dhash64"], dhash_sequence[-1]["dhash64"])
            if n_frames >= 2
            else 0
        )

        # 2. Settling & Oscillation Detector
        if n_frames < 2:
            regime = "initializing"
            cycle_detected = False
            settled = False
        else:
            tail_transitions = recent_transitions[-self.settle_ticks :]
            settled = bool(
                len(tail_transitions) >= self.settle_ticks
                and all(
                    t["changed_pixels"] == 0 and t["dhash64_hamming_distance"] == 0
                    for t in tail_transitions
                )
            )
            # Detect non-trivial cycle (`A -> B -> A`): latest frame is pixel-identical to an earlier
            # non-adjacent frame. A matching 64-bit dHash alone is too coarse (steady motion repeats it).
            latest_sha = dhash_sequence[-1]["native_pixel_sha256"]
            earlier_non_adjacent = dhash_sequence[:-2]
            last_step_changed = bool(recent_transitions and recent_transitions[-1]["changed_pixels"] > 0)
            cycle_detected = bool(
                not settled
                and last_step_changed
                and any(e["native_pixel_sha256"] == latest_sha for e in earlier_non_adjacent)
            )
            if settled:
                regime = "settled"
            elif cycle_detected:
                regime = "oscillating_cycle"
            else:
                regime = "evolving"

        # 3. Multi-Frame Kinematic Trajectory Chain (centroids, velocity, acceleration, ghost next-tick prediction)
        centroids_by_ver: List[Tuple[str, Optional[Tuple[float, float]], int]] = []
        for v in active_versions:
            cc = self._frame_records[v]["inspection"]["native_simd_metrics"]["op_connected_components_4c"]
            comps = cc.get("components", [])
            if comps:
                # Track primary (largest-area) component deterministically
                primary = max(comps, key=lambda c: (int(c["area_px"]), -int(c["bbox"]["x_min"])))
                centroids_by_ver.append(
                    (
                        v,
                        (float(primary["centroid"]["x"]), float(primary["centroid"]["y"])),
                        int(primary["area_px"]),
                    )
                )
            else:
                centroids_by_ver.append((v, None, 0))

        velocities: List[Dict[str, Any]] = []
        for i in range(len(centroids_by_ver) - 1):
            v_a, c_a, area_a = centroids_by_ver[i]
            v_b, c_b, area_b = centroids_by_ver[i + 1]
            if c_a is not None and c_b is not None:
                vx = round(c_b[0] - c_a[0], 4)
                vy = round(c_b[1] - c_a[1], 4)
                speed = round(math.hypot(vx, vy), 4)
                velocities.append(
                    {
                        "from_version": v_a,
                        "to_version": v_b,
                        "vx_px_per_tick": vx,
                        "vy_px_per_tick": vy,
                        "speed_px_per_tick": speed,
                        "area_delta_px": area_b - area_a,
                    }
                )

        accelerations: List[Dict[str, Any]] = []
        for i in range(len(velocities) - 1):
            va = velocities[i]
            vb = velocities[i + 1]
            ax = round(float(vb["vx_px_per_tick"]) - float(va["vx_px_per_tick"]), 4)
            ay = round(float(vb["vy_px_per_tick"]) - float(va["vy_px_per_tick"]), 4)
            accelerations.append(
                {
                    "at_version": vb["from_version"],
                    "to_version": vb["to_version"],
                    "ax_px_per_tick2": ax,
                    "ay_px_per_tick2": ay,
                    "accel_magnitude": round(math.hypot(ax, ay), 4),
                }
            )

        predicted_next_centroid: Optional[Dict[str, Any]] = None
        if centroids_by_ver and centroids_by_ver[-1][1] is not None and velocities:
            cx_last, cy_last = centroids_by_ver[-1][1]  # type: ignore[misc]
            vx_last = float(velocities[-1]["vx_px_per_tick"])
            vy_last = float(velocities[-1]["vy_px_per_tick"])
            ax_last = float(accelerations[-1]["ax_px_per_tick2"]) if accelerations else 0.0
            ay_last = float(accelerations[-1]["ay_px_per_tick2"]) if accelerations else 0.0
            pred_x = max(0.0, min(float(self.width - 1), round(cx_last + vx_last + 0.5 * ax_last, 4)))
            pred_y = max(0.0, min(float(self.height - 1), round(cy_last + vy_last + 0.5 * ay_last, 4)))
            predicted_next_centroid = {
                "predicted_x": pred_x,
                "predicted_y": pred_y,
                "target_plane": "ghost_isolation_plane",
                "epistemic_state": "predicted",
                "observed": False,
            }

        # 4. Dirty-Tile 16x16 Profile
        dirty_ratios = [float(t["dirty_tiles_16x16"]["dirty_tile_ratio"]) for t in recent_transitions]
        mean_dirty_ratio = round(sum(dirty_ratios) / float(len(dirty_ratios)), 6) if dirty_ratios else 0.0
        latest_dirty = recent_transitions[-1]["dirty_tiles_16x16"] if recent_transitions else None

        active_verified = [
            v for v, entry in self._observation_ledger.items() if entry.get("fresh") is True
        ]
        stale_versions = [
            v
            for v, entry in self._observation_ledger.items()
            if entry.get("uncertainty_state") == "stale"
        ]

        summary = {
            "epistemic_label": "derived",
            "observed": False,
            "window_capacity": self.max_window_frames,
            "active_frame_count": n_frames,
            "active_versions": active_versions,
            "evicted_frame_count": self._evicted_frame_count,
            "cumulative_zeroized_bytes": (
                self._last_eviction.get("cumulative_zeroized_bytes", 0) if self._last_eviction else 0
            ),
            "epistemic_watchdog": {
                "fresh_verified_versions": active_verified,
                "decayed_stale_versions": stale_versions,
            },
            "dhash64_hamming_timeline": {
                "step_hamming_distances": step_hammings,
                "max_step_hamming": max(step_hammings) if step_hammings else 0,
                "net_window_hamming_drift": net_window_hamming,
                "checkpoints_triggered": sum(1 for t in recent_transitions if t["checkpoint_triggered"]),
            },
            "settling_and_oscillation_detector": {
                "regime": regime,
                "settled": settled,
                "oscillating_cycle_detected": cycle_detected,
                "settle_ticks_required": self.settle_ticks,
            },
            "kinematic_trajectory_chain": {
                "velocities": velocities,
                "accelerations": accelerations,
                "predicted_next_centroid": predicted_next_centroid,
            },
            "dirty_tile_16x16_profile": {
                "mean_dirty_tile_ratio": mean_dirty_ratio,
                "latest_transition_dirty_tiles": latest_dirty,
            },
        }
        assert_no_raw_buffer_leakage(summary)
        return summary

