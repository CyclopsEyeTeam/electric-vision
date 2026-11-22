#!/usr/bin/env python3
"""
Disposable Fixture Harness for the "Electric Eye" Pixel Perception
and Render Labelling System (Gates 1-5 Remediated).

Design-contract implementation only:
- Isolated, synthetic, in-memory 128x128 RGBA fixture world.
- Zero external calls, network requests, model APIs, daemons, or live Cyclops edits.
- Raw pixel buffers / cache bytes and HMAC signing keys are isolated inside a
  private lexical closure vault and never exposed on instance attributes or
  in any agent-facing return payload.
- Enforces the 5-layer epistemic architecture:
  1. VerifiedSource
  2. DeterministicMeasurement
  3. DerivedRepresentation
  4. SemanticInterpretation (boundary/refusal)
  5. ElectricRender + ObservationWindowGate + Hash-Chained Security Audit Ledger
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import resource
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CANVAS_WIDTH = 128
CANVAS_HEIGHT = 128
CANVAS_TOTAL_PIXELS = CANVAS_WIDTH * CANVAS_HEIGHT
ROW_BYTE_STRIDE = CANVAS_WIDTH * 4
OPERATOR_VERSION = "1.0.0"
FIXED_LOGICAL_TIMESTAMP = "2026-10-29T22:35:26.000Z"
DISPOSABLE_FIXTURE_STATIC_HMAC_KEY = b"electric-eye-disposable-fixture-gate-key-v1"

MAX_CACHE_ENTRIES = 64
MAX_MARKS_PER_SOURCE = 1024
MAX_REPORTED_REGIONS = 512

VALID_HARNESS_PROFILES = frozenset({
    "disposable_fixture",
    "non_fixture",
    "controlled_pilot_simulation",
})

# NEXT-GATE-2: Strict lowercase opaque handle regex; rejects uppercase, paths, dots, and emails.
SOURCE_ID_REGEX = re.compile(r"^src_[a-z0-9_]{1,48}$")
SOURCE_VERSION_REGEX = re.compile(r"^v([1-9][0-9]{0,6})$")
MARK_ID_REGEX = re.compile(r"^[a-zA-Z0-9_]{1,48}$")
LONG_DIGIT_RUN_REGEX = re.compile(r"\d{6,}")

# NEXT-GATE-2: Prohibited semantic, filename, path, or user-identifying tokens in source_id.
PROHIBITED_SOURCE_ID_TOKENS = frozenset({
    "user",
    "username",
    "userid",
    "uid",
    "owner",
    "person",
    "human",
    "client",
    "customer",
    "patient",
    "subject",
    "member",
    "account",
    "acct",
    "email",
    "mail",
    "mailto",
    "phone",
    "tel",
    "ssn",
    "passport",
    "tax",
    "secret",
    "token",
    "passwd",
    "password",
    "cred",
    "credential",
    "auth",
    "session",
    "cookie",
    "file",
    "filename",
    "filepath",
    "path",
    "dir",
    "folder",
    "home",
    "desktop",
    "documents",
    "downloads",
    "tmp",
    "var",
    "etc",
    "usr",
    "mnt",
    "media",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "webp",
    "bmp",
    "svg",
    "tiff",
    "pdf",
    "json",
    "txt",
    "csv",
    "mp4",
    "mov",
    "charlie",
    "alice",
    "bob",
    "admin",
})

VALID_SOURCE_KINDS = frozenset({
    "verified_pixels",
    "drawing_state",
    "canvas_state",
    "image_bytes",
    "captured_frame",
})

VALID_RENDER_TYPES = frozenset({
    "observed",
    "derived",
    "predicted",
    "diagnostic",
    "decorative",
})

VALID_RENDER_SOURCES = frozenset({
    "verified_pixels",
    "drawing_state",
    "measurement",
    "inference",
    "generation",
})

VALID_VISIBILITY_POLICIES = frozenset({
    "agent_visible",
    "silent_diagnostic",
    "human_only",
})

VALID_LAYER_ISOLATION_PLANES = frozenset({
    "verified_plane",
    "derived_overlay_plane",
    "ghost_isolation_plane",
    "diagnostic_plane",
    "decorative_plane",
})

VALID_CONFIDENCE_LEVELS = frozenset({
    "low",
    "medium",
    "high",
    "unknown",
})

VALID_UNCERTAINTY_STATES = frozenset({
    "verified",
    "derived",
    "predicted",
    "stale",
    "unknown",
})

VALID_SEMANTIC_EPISTEMIC_LABELS = frozenset({
    "derived",
    "inferred",
    "uncertain",
})

CLOSED_FORM_OPERATORS = frozenset({
    "inspect_source",
    "measure_dimensions",
    "measure_bbox",
    "sample_colour",
    "measure_distance",
    "measure_angle",
    "count_pixels",
    "count_changed_pixels",
    "extract_changed_regions",
    "count_objects",
})

ALLOWED_OPERATOR_PARAM_KEYS: Dict[str, frozenset[str]] = {
    "inspect_source": frozenset(),
    "measure_dimensions": frozenset(),
    "measure_bbox": frozenset({"selector"}),
    "sample_colour": frozenset({"x", "y"}),
    "measure_distance": frozenset({"p1", "p2", "metric"}),
    "measure_angle": frozenset({"p1", "p2"}),
    "count_pixels": frozenset({"predicate"}),
    "count_changed_pixels": frozenset({"channel_tolerance"}),
    "extract_changed_regions": frozenset({"connectivity", "channel_tolerance"}),
    "count_objects": frozenset({"connectivity", "min_area_px"}),
}

# NEXT-GATE-4: Closed inner key allowlists for nested parameter dictionaries.
ALLOWED_SELECTOR_KEYS = frozenset({"type", "rgba", "tolerance"})
ALLOWED_PREDICATE_KEYS = frozenset({"type", "rgba", "tolerance"})
ALLOWED_POINT_KEYS = frozenset({"x", "y"})
ALLOWED_SECONDARY_SOURCE_KEYS = frozenset({"source_id", "source_version"})

PROHIBITED_SEMANTIC_QUERY_TYPES = frozenset({
    "semantic_meaning",
    "aesthetic_judgment",
    "agent_intent",
    "free_text_interpretation",
})

SEMANTIC_KEY_INDICATORS = frozenset({
    "prompt",
    "question",
    "query_type",
    "semantic_hypothesis",
    "intent",
    "meaning",
    "interpretation",
    "aesthetic",
})

REFUSAL_CODE_SEMANTIC = "ERR_SEMANTIC_INTERPRETATION_OUT_OF_SCOPE"
REFUSAL_REASON_SEMANTIC = (
    "Pixel Perception measurement operators compute exact geometric and "
    "colorimetric facts only. Semantic meaning, evaluation, and intent "
    "inference are prohibited."
)

UNCERTAINTY_STYLE_SPEC: Dict[str, Dict[str, str]] = {
    "verified": {
        "border_pattern": "═══",
        "pattern_name": "solid_double_line",
        "glow_colour_hex": "#00F5D4",
        "glow_name": "steady_crisp_cyan",
        "glyph": "◉",
    },
    "derived": {
        "border_pattern": "─── ───",
        "pattern_name": "dashed_line",
        "glow_colour_hex": "#FFB703",
        "glow_name": "electric_amber",
        "glyph": "◈",
    },
    "predicted": {
        "border_pattern": "··· ///",
        "pattern_name": "dotted_sparse_hatch_45deg",
        "glow_colour_hex": "#9B5DE5",
        "glow_name": "pulsing_violet_ghost",
        "glyph": "◌",
    },
    "stale": {
        "border_pattern": "⋀⋁⋀⋁",
        "pattern_name": "sawtooth_notched",
        "glow_colour_hex": "#9C6644",
        "glow_name": "desaturated_ochre",
        "glyph": "⌛",
    },
    "unknown": {
        "border_pattern": "╳╳╳ ?",
        "pattern_name": "cross_hatched_question",
        "glow_colour_hex": "#FF006E",
        "glow_name": "high_contrast_magenta_grey",
        "glyph": "?",
    },
}


def canonical_json(obj: Any) -> str:
    """Deterministic, sorted-key JSON serialization with no whitespace drift."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def parse_version_num(version_str: str) -> int:
    m = SOURCE_VERSION_REGEX.match(version_str)
    if not m:
        raise ValueError(
            f"ERR_INVALID_SOURCE_VERSION: '{version_str}' must match ^v[1-9][0-9]*$"
        )
    return int(m.group(1))


def compute_replay_digest(
    source_id: Optional[str],
    source_version: Optional[str],
    secondary_source: Optional[Dict[str, str]],
    operator_name: str,
    operator_version: str,
    parameters: Dict[str, Any],
    result: Any,
) -> str:
    """
    Compute deterministic replay digest over:
    source_id, source_version, secondary_source, operator_name,
    operator_version, canonical_json(parameters), canonical_json(result),
    excluding measurement_id and wall-clock timestamp.
    """
    payload = {
        "source_id": source_id,
        "source_version": source_version,
        "secondary_source": secondary_source,
        "operator_name": operator_name,
        "operator_version": operator_version,
        "parameters": parameters,
        "result": result,
    }
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def assert_no_raw_buffer_leakage(payload: Any) -> None:
    """
    Guardrail ensuring no raw bytes, bytearray, memoryview, or full-canvas
    pixel arrays are ever exposed in agent-facing structured outputs.
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raise RuntimeError("SECURITY_VIOLATION: Raw buffer bytes leaked in output.")
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in (
                "_raw_pixels",
                "_vault",
                "raw_bytes",
                "row_bytes",
                "cache_bytes",
                "pixel_buffer",
            ):
                raise RuntimeError(f"SECURITY_VIOLATION: Forbidden raw buffer key '{k}'.")
            assert_no_raw_buffer_leakage(v)
    elif isinstance(payload, (list, tuple)):
        if len(payload) > MAX_REPORTED_REGIONS:
            raise RuntimeError("SECURITY_VIOLATION: Unbounded array leaked in output.")
        for item in payload:
            assert_no_raw_buffer_leakage(item)


@dataclass(frozen=True)
class DrawingMark:
    """Deterministic drawing mark primitive for synthetic 128x128 canvases."""
    mark_id: str
    shape: str  # "rect" | "line" | "dot"
    x0: int
    y0: int
    x1: int
    y1: int
    rgba: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not MARK_ID_REGEX.match(self.mark_id):
            raise ValueError(f"ERR_INVALID_MARK_ID: '{self.mark_id}'")
        if self.shape not in ("rect", "line", "dot"):
            raise ValueError(f"ERR_UNSUPPORTED_MARK_SHAPE: '{self.shape}'")
        if len(self.rgba) != 4 or any(not (0 <= int(c) <= 255) for c in self.rgba):
            raise ValueError(f"ERR_INVALID_RGBA: {self.rgba}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mark_id": self.mark_id,
            "shape": self.shape,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "rgba": list(self.rgba),
        }


@dataclass(frozen=True)
class _RasterizedSurface:
    """
    Internal pre-indexed representation of a 128x128 RGBA surface inside the closure vault.
    Uses packed immutable bytes + row slices + horizontal RLE runs to avoid
    allocating 16,384 Python tuple objects per canvas.
    """
    raw_bytes: bytes
    row_bytes: Tuple[bytes, ...]
    fg_runs: Tuple[Tuple[int, int, int, Tuple[int, int, int, int]], ...]


def _create_isolated_vault_and_gate(
    *,
    profile: str = "disposable_fixture",
    hmac_key_provider: Optional[Callable[[], bytes]] = None,
) -> Tuple[
    Callable[[str, str, str, Sequence[DrawingMark], str], str],
    Callable[[str, Sequence[DrawingMark], str], str],
    Callable[[str, str], _RasterizedSurface],
    Callable[[str, str, int, int], Tuple[int, int, int, int]],
    Callable[[], int],
    Callable[[str, str, str, str, str], str],
    Callable[[str, str, str, str, str, str], bool],
    Callable[[], int],
]:
    """
    GATE-2 & NEXT-GATE-1 Remediation (Fixes SEC-01, SEC-02, SEC-R1):
    - In 'disposable_fixture' profile, uses the deterministic fixture HMAC key so
      scenarios E1-E10 and receipt digests replay byte-identically.
    - In any non-disposable profile ('non_fixture', 'controlled_pilot_simulation'),
      strictly forbids the static fixture key and requires either a >=32-byte key
      from an external KMS/HSM/daemon key provider or a fresh CSPRNG 32-byte key.
    """
    if profile not in VALID_HARNESS_PROFILES:
        raise ValueError(f"ERR_INVALID_PROFILE: Unsupported profile '{profile}'.")

    if profile == "disposable_fixture":
        if hmac_key_provider is not None:
            secret_hmac_key = hmac_key_provider()
        else:
            secret_hmac_key = DISPOSABLE_FIXTURE_STATIC_HMAC_KEY
    else:
        if hmac_key_provider is not None:
            candidate_key = hmac_key_provider()
            if (
                not isinstance(candidate_key, bytes)
                or len(candidate_key) < 32
                or candidate_key == DISPOSABLE_FIXTURE_STATIC_HMAC_KEY
            ):
                raise PermissionError(
                    "ERR_STATIC_OR_WEAK_KEY_PROHIBITED: Non-disposable profiles forbid "
                    "the static fixture HMAC key and require >=32-byte KMS/daemon key material."
                )
            secret_hmac_key = candidate_key
        else:
            secret_hmac_key = secrets.token_bytes(32)

    surfaces: Dict[Tuple[str, str], _RasterizedSurface] = {}
    ghost_surfaces: Dict[str, _RasterizedSurface] = {}
    raster_cache: "OrderedDict[Tuple[Tuple[DrawingMark, ...], Tuple[int, int, int, int]], _RasterizedSurface]" = OrderedDict()

    def _rasterize(
        marks: Sequence[DrawingMark],
        background: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> _RasterizedSurface:
        if len(marks) > MAX_MARKS_PER_SOURCE:
            raise ValueError(f"ERR_MARK_COUNT_EXCEEDED: {len(marks)} > {MAX_MARKS_PER_SOURCE}")
        marks_key = (tuple(marks), background)
        cached = raster_cache.get(marks_key)
        if cached is not None:
            raster_cache.move_to_end(marks_key)
            return cached

        bg_bytes = bytes(background)
        empty_row_bytes = bg_bytes * CANVAS_WIDTH
        raw = bytearray(empty_row_bytes * CANVAS_HEIGHT)

        for m in marks:
            m_bytes = bytes(m.rgba)
            if m.shape == "rect":
                xmin = max(0, min(m.x0, m.x1))
                xmax = min(CANVAS_WIDTH - 1, max(m.x0, m.x1))
                ymin = max(0, min(m.y0, m.y1))
                ymax = min(CANVAS_HEIGHT - 1, max(m.y0, m.y1))
                if xmin <= xmax and ymin <= ymax:
                    span = xmax - xmin + 1
                    row_byte_slice = m_bytes * span
                    span_bytes = span << 2
                    for y in range(ymin, ymax + 1):
                        start_b = ((y << 7) + xmin) << 2
                        raw[start_b : start_b + span_bytes] = row_byte_slice
            elif m.shape == "dot":
                if 0 <= m.x0 < CANVAS_WIDTH and 0 <= m.y0 < CANVAS_HEIGHT:
                    start_b = ((m.y0 << 7) + m.x0) << 2
                    raw[start_b : start_b + 4] = m_bytes
            elif m.shape == "line":
                x0, y0, x1, y1 = m.x0, m.y0, m.x1, m.y1
                dx = abs(x1 - x0)
                sx = 1 if x0 < x1 else -1
                dy = -abs(y1 - y0)
                sy = 1 if y0 < y1 else -1
                err = dx + dy
                while True:
                    if 0 <= x0 < CANVAS_WIDTH and 0 <= y0 < CANVAS_HEIGHT:
                        start_b = ((y0 << 7) + x0) << 2
                        raw[start_b : start_b + 4] = m_bytes
                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 >= dy:
                        err += dy
                        x0 += sx
                    if e2 <= dx:
                        err += dx
                        y0 += sy
            else:
                raise ValueError(f"Unsupported mark shape: {m.shape}")

        raw_frozen = bytes(raw)
        row_bytes_list: List[bytes] = []
        fg_runs: List[Tuple[int, int, int, Tuple[int, int, int, int]]] = []

        for y in range(CANVAS_HEIGHT):
            r_bytes = raw_frozen[y * ROW_BYTE_STRIDE : (y + 1) * ROW_BYTE_STRIDE]
            row_bytes_list.append(r_bytes)
            if r_bytes == empty_row_bytes:
                continue
            x = 0
            while x < CANVAS_WIDTH:
                x4 = x << 2
                if r_bytes[x4 + 3] > 0:
                    px_bytes = r_bytes[x4 : x4 + 4]
                    rgba = (r_bytes[x4], r_bytes[x4 + 1], r_bytes[x4 + 2], r_bytes[x4 + 3])
                    x_start = x
                    x += 1
                    while x < CANVAS_WIDTH and r_bytes[(x << 2) : (x << 2) + 4] == px_bytes:
                        x += 1
                    fg_runs.append((y, x_start, x - 1, rgba))
                else:
                    x += 1

        surface = _RasterizedSurface(
            raw_bytes=raw_frozen,
            row_bytes=tuple(row_bytes_list),
            fg_runs=tuple(fg_runs),
        )
        raster_cache[marks_key] = surface
        if len(raster_cache) > MAX_CACHE_ENTRIES:
            raster_cache.popitem(last=False)
        return surface

    def _compute_sha256(raw_bytes: bytes, marks: Sequence[DrawingMark], source_kind: str) -> str:
        meta = canonical_json({
            "source_kind": source_kind,
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "marks": [m.to_dict() for m in marks],
        }).encode("utf-8")
        h = hashlib.sha256()
        h.update(meta)
        h.update(b"|")
        h.update(raw_bytes)
        return "sha256:" + h.hexdigest()

    def store_verified_source(
        source_id: str,
        source_version: str,
        source_kind: str,
        marks: Sequence[DrawingMark],
        target_plane: str = "verified_plane",
    ) -> str:
        if target_plane != "verified_plane":
            raise ValueError(
                f"ERR_PLANE_VIOLATION: Verified sources must reside in verified_plane, got {target_plane}"
            )
        key = (source_id, source_version)
        if key in surfaces:
            raise ValueError(
                f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Source {source_id}@{source_version} already exists."
            )
        surface = _rasterize(marks)
        surfaces[key] = surface
        return _compute_sha256(surface.raw_bytes, marks, source_kind)

    def store_ghost_render_pixels(
        render_id: str,
        marks: Sequence[DrawingMark],
        target_plane: str,
    ) -> str:
        if target_plane == "verified_plane":
            raise ValueError(
                "ERR_GHOST_BUFFER_CONTAMINATION: Predicted/ghost pixels must never enter verified_plane."
            )
        if target_plane != "ghost_isolation_plane":
            raise ValueError(
                f"ERR_INVALID_GHOST_PLANE: Expected ghost_isolation_plane, got {target_plane}."
            )
        surface = _rasterize(marks)
        ghost_surfaces[render_id] = surface
        return _compute_sha256(surface.raw_bytes, marks, "ghost_prediction")

    def get_surface(source_id: str, source_version: str) -> _RasterizedSurface:
        key = (source_id, source_version)
        if key not in surfaces:
            raise KeyError(f"ERR_UNVERIFIED_SOURCE: {source_id}@{source_version} not found.")
        return surfaces[key]

    def sample_pixel(source_id: str, source_version: str, x: int, y: int) -> Tuple[int, int, int, int]:
        raw = get_surface(source_id, source_version).raw_bytes
        off = ((y << 7) + x) << 2
        return (raw[off], raw[off + 1], raw[off + 2], raw[off + 3])

    def estimate_vault_bytes() -> int:
        total = 0
        for s in surfaces.values():
            total += len(s.raw_bytes) + sum(len(rb) for rb in s.row_bytes) + len(s.fg_runs) * 64
        for gs in ghost_surfaces.values():
            total += len(gs.raw_bytes) + sum(len(rb) for rb in gs.row_bytes) + len(gs.fg_runs) * 64
        return total

    def sign_receipt(
        observation_receipt_id: str,
        render_id: str,
        captured_source_id: str,
        captured_source_version: str,
        pipeline_stage: str,
    ) -> str:
        msg = canonical_json({
            "observation_receipt_id": observation_receipt_id,
            "render_id": render_id,
            "captured_source_id": captured_source_id,
            "captured_source_version": captured_source_version,
            "pipeline_stage": pipeline_stage,
        }).encode("utf-8")
        return "hmac-sha256:" + hmac.new(secret_hmac_key, msg, hashlib.sha256).hexdigest()

    def verify_receipt_sig(
        observation_receipt_id: str,
        render_id: str,
        captured_source_id: str,
        captured_source_version: str,
        pipeline_stage: str,
        candidate_signature: str,
    ) -> bool:
        expected = sign_receipt(
            observation_receipt_id,
            render_id,
            captured_source_id,
            captured_source_version,
            pipeline_stage,
        )
        return hmac.compare_digest(candidate_signature, expected)

    def get_raster_cache_size() -> int:
        return len(raster_cache)

    def evict_surfaces(source_keys: Sequence[Sequence[str]], render_ids: Sequence[str] = ()) -> int:
        """Drops stored proxy surfaces the caller no longer serves; their descriptors stay immutable."""
        dropped = 0
        for key in source_keys:
            if surfaces.pop((str(key[0]), str(key[1])), None) is not None:
                dropped += 1
        for render_id in render_ids:
            if ghost_surfaces.pop(str(render_id), None) is not None:
                dropped += 1
        return dropped

    return (
        store_verified_source,
        store_ghost_render_pixels,
        get_surface,
        sample_pixel,
        estimate_vault_bytes,
        sign_receipt,
        verify_receipt_sig,
        get_raster_cache_size,
        evict_surfaces,
    )


class SimulatedPrivilegedGateBoundary:
    """
    NEXT-GATE-1 Remediation (Addresses SEC-R1 in disposable scope):
    Simulated IPC/RPC boundary abstraction modelling a narrow capability-gated contract
    between the unprivileged harness and a privileged capture/signing daemon without
    claiming real OS process isolation inside the single-process Python test suite.
    """

    def __init__(
        self,
        *,
        hmac_key_provider: Optional[Callable[[], bytes]] = None,
        capture_service_token: Optional[str] = None,
    ) -> None:
        if hmac_key_provider is not None:
            key = hmac_key_provider()
        else:
            key = secrets.token_bytes(32)
        if (
            not isinstance(key, bytes)
            or len(key) < 32
            or key == DISPOSABLE_FIXTURE_STATIC_HMAC_KEY
        ):
            raise PermissionError(
                "ERR_STATIC_OR_WEAK_KEY_PROHIBITED: SimulatedPrivilegedGateBoundary "
                "forbids the static fixture key and requires >=32-byte key material."
            )
        self._capture_token = capture_service_token or ("cap_tok_" + secrets.token_hex(16))

        def _sign(
            observation_receipt_id: str,
            render_id: str,
            captured_source_id: str,
            captured_source_version: str,
            pipeline_stage: str,
        ) -> str:
            msg = canonical_json({
                "observation_receipt_id": observation_receipt_id,
                "render_id": render_id,
                "captured_source_id": captured_source_id,
                "captured_source_version": captured_source_version,
                "pipeline_stage": pipeline_stage,
            }).encode("utf-8")
            return "hmac-sha256:" + hmac.new(key, msg, hashlib.sha256).hexdigest()

        def _verify(
            observation_receipt_id: str,
            render_id: str,
            captured_source_id: str,
            captured_source_version: str,
            pipeline_stage: str,
            candidate_signature: str,
        ) -> bool:
            expected = _sign(
                observation_receipt_id,
                render_id,
                captured_source_id,
                captured_source_version,
                pipeline_stage,
            )
            return hmac.compare_digest(candidate_signature, expected)

        self._rpc_sign = _sign
        self._rpc_verify = _verify

    def get_boundary_descriptor(self) -> Dict[str, Any]:
        return {
            "boundary_mode": "simulated_in_memory_rpc",
            "claims_real_process_isolation": False,
            "ipc_transport_spec": "unix_domain_socket_length_prefixed_canonical_json_v1",
            "key_management": "ephemeral_or_kms_injected_non_static_32b",
            "requires_privileged_capture_capability_token": True,
        }

    def rpc_issue_observation_receipt(
        self,
        *,
        caller_capability_token: Optional[str],
        observation_receipt_id: str,
        render: Dict[str, Any],
        captured_source_desc: VerifiedSourceDescriptor,
    ) -> Dict[str, Any]:
        if not isinstance(caller_capability_token, str) or not hmac.compare_digest(
            caller_capability_token, self._capture_token
        ):
            raise PermissionError(
                "ERR_UNAUTHORIZED_RECEIPT_ISSUER: rpc_issue_observation_receipt "
                "is restricted to the privileged capture service capability token."
            )
        if render.get("render_type") != "observed" or captured_source_desc.source_kind != "captured_frame":
            raise PermissionError(
                "ERR_UNVERIFIED_OBSERVATION_CLAIM: Simulated gate boundary requires "
                "render_type='observed' and source_kind='captured_frame'."
            )
        binds = any(
            ref.get("ref_type") == "source"
            and ref.get("ref_id") == captured_source_desc.source_id
            and ref.get("ref_version") == captured_source_desc.source_version
            for ref in render.get("provenance_refs", ())
        )
        if not binds:
            raise PermissionError(
                "ERR_UNVERIFIED_OBSERVATION_CLAIM: Render provenance_refs do not bind to captured_frame."
            )
        stage = "verified_observation_window"
        sig = self._rpc_sign(
            observation_receipt_id,
            render["render_id"],
            captured_source_desc.source_id,
            captured_source_desc.source_version,
            stage,
        )
        return {
            "observation_receipt_id": observation_receipt_id,
            "render_id": render["render_id"],
            "captured_source_id": captured_source_desc.source_id,
            "captured_source_version": captured_source_desc.source_version,
            "pipeline_stage": stage,
            "receipt_signature": sig,
        }

    def rpc_verify_observation_receipt(self, receipt: Dict[str, Any]) -> bool:
        return self._rpc_verify(
            receipt["observation_receipt_id"],
            receipt["render_id"],
            receipt["captured_source_id"],
            receipt["captured_source_version"],
            receipt["pipeline_stage"],
            receipt["receipt_signature"],
        )


class InMemoryAppendOnlyAuditSink:
    """
    NEXT-GATE-3 Remediation (Addresses AUD-R1 in disposable scope):
    In-memory implementation of the external append-only audit sink contract.
    Enforces strict sequential ordering, hash-chain continuity, and dual-clock presence.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def append_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        expected_seq = len(self._entries) + 1
        expected_prev = (
            self._entries[-1]["entry_hash"]
            if self._entries
            else "sha256:" + ("0" * 64)
        )
        if entry.get("entry_seq") != expected_seq:
            raise ValueError(
                f"ERR_AUDIT_SEQUENCE_VIOLATION: Expected seq {expected_seq}, got {entry.get('entry_seq')}"
            )
        if entry.get("prev_hash") != expected_prev:
            raise ValueError("ERR_AUDIT_CHAIN_BROKEN: prev_hash mismatch on append.")
        for req_clock in ("logical_clock", "logical_timestamp", "utc_wall_timestamp"):
            if not entry.get(req_clock):
                raise ValueError(f"ERR_AUDIT_CLOCK_MISSING: Missing {req_clock}")
        body = {k: v for k, v in entry.items() if k != "entry_hash"}
        recomputed = sha256_hex(canonical_json(body).encode("utf-8"))
        if recomputed != entry.get("entry_hash"):
            raise ValueError("ERR_AUDIT_HASH_MISMATCH: Entry hash failed verification on sink ingest.")
        stored = copy.deepcopy(entry)
        self._entries.append(stored)
        return copy.deepcopy(stored)

    def get_entries(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._entries)

    def export_chain_head(self) -> Dict[str, Any]:
        return {
            "sink_type": "in_memory_append_only_audit_sink_v1",
            "entry_count": len(self._entries),
            "head_seq": self._entries[-1]["entry_seq"] if self._entries else 0,
            "chain_head_digest": (
                self._entries[-1]["entry_hash"]
                if self._entries
                else "sha256:" + ("0" * 64)
            ),
        }


@dataclass(frozen=True)
class VerifiedSourceDescriptor:
    """Public, opaque handle for a VerifiedSource (Layer 1). Contains no raw bytes."""
    source_id: str
    source_version: str
    source_kind: str
    content_sha256: str
    width: int
    height: int
    mark_count: int
    plane: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_kind": self.source_kind,
            "content_sha256": self.content_sha256,
            "dimensions": {"width": self.width, "height": self.height},
            "mark_count": self.mark_count,
            "plane": self.plane,
            "created_at": self.created_at,
            "observed": False,
            "observation_claim": False,
        }


class ElectricEyeFixtureHarness:
    """
    Isolated, deterministic 128x128 RGBA fixture harness implementing:
    - Layer 1: VerifiedSource Registry (immutable versioned handles, opaque non-semantic IDs)
    - Layer 2: DeterministicMeasurement Engine (10 closed-form operators + closed nested schemas + refusal contract)
    - Layer 3: DerivedRepresentation Engine (5 structural comparators with provenance/confidence)
    - Layer 4: SemanticInterpretation Boundary (strict refusal & labelling enforcement)
    - Layer 5: ElectricRender Engine & ObservationWindowGate (receipt-gated sight with strict
               source-to-render provenance binding, default-on dynamic staleness detection,
               and dual-clock hash-chained security audit ledger)
    """

    def __init__(
        self,
        redact_refused_prompts: Optional[bool] = None,
        *,
        profile: str = "non_fixture",
        hmac_key_provider: Optional[Callable[[], bytes]] = None,
        simulated_gate_boundary: Optional[SimulatedPrivilegedGateBoundary] = None,
        audit_sink: Optional[Any] = None,
        utc_clock_fn: Optional[Callable[[], str]] = None,
        out_of_process_vault_client: Optional[Any] = None,
        quarantine_supervisor_client: Optional[Any] = None,
    ) -> None:
        if profile not in VALID_HARNESS_PROFILES:
            raise ValueError(f"ERR_INVALID_PROFILE: Unsupported profile '{profile}'.")

        # NEXT-GATE-2: redact_refused_prompts defaults to True for all non-fixture profiles,
        # and is permitted to be False only in profile='disposable_fixture'.
        if redact_refused_prompts is None:
            resolved_redact = profile != "disposable_fixture"
        else:
            if not redact_refused_prompts and profile != "disposable_fixture":
                raise ValueError(
                    "ERR_PRIVACY_POLICY_VIOLATION: redact_refused_prompts=False is permitted "
                    "only in profile='disposable_fixture' for E3 deterministic replay."
                )
            resolved_redact = bool(redact_refused_prompts)

        self._out_of_process_vault_client = out_of_process_vault_client
        self._quarantine_supervisor_client = quarantine_supervisor_client

        if out_of_process_vault_client is not None:
            # EXEC-GATE-1: Zero raw RGBA pixel buffers and zero HMAC signing keys reside in
            # the unprivileged harness process; all vault/signing operations go over AF_UNIX IPC.
            self._store_verified_source = out_of_process_vault_client.store_verified_source
            self._store_ghost_pixels = out_of_process_vault_client.store_ghost_pixels
            self._get_surface = None  # type: ignore[assignment]
            self._sample_pixel = None  # type: ignore[assignment]
            self._estimate_vault_bytes = lambda: int(
                out_of_process_vault_client.ping_status().get("vault_memory_bytes", 0)
            )
            self._sign_receipt = None  # type: ignore[assignment]
            self._verify_receipt_sig = None  # type: ignore[assignment]
            self._get_raster_cache_size = lambda: int(
                out_of_process_vault_client.ping_status().get("raster_cache_size", 0)
            )
            self.has_local_vault_or_hmac_keys = False
        else:
            (
                self._store_verified_source,
                self._store_ghost_pixels,
                self._get_surface,
                self._sample_pixel,
                self._estimate_vault_bytes,
                self._sign_receipt,
                self._verify_receipt_sig,
                self._get_raster_cache_size,
                self._evict_surfaces,
            ) = _create_isolated_vault_and_gate(
                profile=profile,
                hmac_key_provider=hmac_key_provider,
            )
            self.has_local_vault_or_hmac_keys = True

        self._profile = profile
        self._redact_refused_prompts = resolved_redact
        self._simulated_gate_boundary = simulated_gate_boundary
        self._audit_sink = audit_sink
        if utc_clock_fn is not None:
            self._utc_clock_fn = utc_clock_fn
        elif profile == "disposable_fixture":
            self._utc_clock_fn = lambda: FIXED_LOGICAL_TIMESTAMP
        else:
            self._utc_clock_fn = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        self._sources: Dict[Tuple[str, str], VerifiedSourceDescriptor] = {}
        self._latest_version_by_source: Dict[str, str] = {}
        self._measurements: List[Dict[str, Any]] = []
        self._derived_representations: Dict[str, Dict[str, Any]] = {}
        self._renders: Dict[str, Dict[str, Any]] = {}
        self._observation_receipts: Dict[str, Dict[str, Any]] = {}
        self._audit_ledger: List[Dict[str, Any]] = []

        # Bounded LRU cache for connected components per (source_id, source_version, connectivity)
        self._cc_cache: "OrderedDict[Tuple[str, str, int], List[Dict[str, Any]]]" = OrderedDict()

        # Counters for telemetry and acceptance verification
        self._measurement_seq = 0
        self._derived_seq = 0
        self._render_seq = 0
        self._receipt_seq = 0
        self._audit_seq = 0
        self.blocked_observation_attempts = 0
        self.false_observation_count = 0
        self.blocked_ghost_contamination_attempts = 0
        self.semantic_queries_total = 0
        self.semantic_queries_refused = 0
        self._replayed_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def _assert_supervisor_not_halted(self) -> None:
        if self._quarantine_supervisor_client is not None:
            self._quarantine_supervisor_client.assert_not_halted()

    # -------------------------------------------------------------------------
    # Hash-Chained Security Audit Ledger (GATE-4, NEXT-GATE-3 & EXEC-GATE-2)
    # -------------------------------------------------------------------------

    def _append_audit_event(
        self,
        *,
        event_type: str,
        target_id: str,
        error_code: str,
        detail: str,
    ) -> Dict[str, Any]:
        next_seq = self._audit_seq + 1
        prev_hash = (
            self._audit_ledger[-1]["entry_hash"]
            if self._audit_ledger
            else "sha256:" + ("0" * 64)
        )
        utc_ts = self._utc_clock_fn()
        logical_clock = f"seq:{next_seq:08d}@{FIXED_LOGICAL_TIMESTAMP}"
        body = {
            "entry_seq": next_seq,
            "logical_clock": logical_clock,
            "logical_timestamp": FIXED_LOGICAL_TIMESTAMP,
            "utc_wall_timestamp": utc_ts,
            "event_type": event_type,
            "target_id": target_id,
            "error_code": error_code,
            "detail": detail,
            "prev_hash": prev_hash,
        }
        entry_hash = sha256_hex(canonical_json(body).encode("utf-8"))
        record = {**body, "entry_hash": entry_hash}
        # EXEC-GATE-2 Fail-Closed: External sink must accept and fsync before local commit
        if self._audit_sink is not None:
            self._audit_sink.append_entry(record)
        self._audit_seq = next_seq
        self._audit_ledger.append(record)
        return copy.deepcopy(record)

    def get_security_audit_ledger(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._audit_ledger)

    def verify_security_audit_ledger(self) -> bool:
        expected_prev = "sha256:" + ("0" * 64)
        for idx, entry in enumerate(self._audit_ledger, start=1):
            if entry.get("entry_seq") != idx or entry.get("prev_hash") != expected_prev:
                return False
            expected_logical_clock = f"seq:{idx:08d}@{FIXED_LOGICAL_TIMESTAMP}"
            if entry.get("logical_clock") != expected_logical_clock:
                return False
            if not entry.get("utc_wall_timestamp"):
                return False
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            recomputed = sha256_hex(canonical_json(body).encode("utf-8"))
            if recomputed != entry.get("entry_hash"):
                return False
            expected_prev = entry["entry_hash"]
        return True

    def export_audit_chain_summary(self) -> Dict[str, Any]:
        chain_valid = self.verify_security_audit_ledger()
        head_entry = self._audit_ledger[-1] if self._audit_ledger else None
        return {
            "chain_valid": chain_valid,
            "entry_count": len(self._audit_ledger),
            "head_seq": head_entry["entry_seq"] if head_entry else 0,
            "head_logical_clock": head_entry["logical_clock"] if head_entry else None,
            "head_utc_wall_timestamp": head_entry["utc_wall_timestamp"] if head_entry else None,
            "chain_head_digest": (
                head_entry["entry_hash"]
                if head_entry
                else "sha256:" + ("0" * 64)
            ),
            "genesis_prev_hash": "sha256:" + ("0" * 64),
        }

    # -------------------------------------------------------------------------
    # Layer 1: VerifiedSource Management (GATE-3 & NEXT-GATE-2: Opaque Handles)
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_source_handle(source_id: str, source_version: str) -> int:
        if not isinstance(source_id, str) or not SOURCE_ID_REGEX.match(source_id):
            raise ValueError(
                f"ERR_INVALID_SOURCE_ID: '{source_id}' must match ^src_[a-z0-9_]{{1,48}}$"
            )
        if LONG_DIGIT_RUN_REGEX.search(source_id):
            raise ValueError(
                f"ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID: '{source_id}' contains a long digit sequence "
                "capable of encoding timestamps, dates, or user identifiers."
            )
        suffix = source_id[4:]
        tokens = [t for t in suffix.split("_") if t]
        if not tokens or any(t in PROHIBITED_SOURCE_ID_TOKENS for t in tokens):
            raise ValueError(
                f"ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID: '{source_id}' contains prohibited semantic, "
                "filename, path, or user-identifying token."
            )
        if not isinstance(source_version, str):
            raise ValueError("ERR_INVALID_SOURCE_VERSION: source_version must be str")
        return parse_version_num(source_version)

    def register_verified_source(
        self,
        source_id: str,
        source_version: str,
        source_kind: str,
        marks: Sequence[DrawingMark],
        *,
        origin_attestation: str = "synthetic_fixture",
    ) -> Dict[str, Any]:
        self._assert_supervisor_not_halted()
        if self._quarantine_supervisor_client is not None:
            self._quarantine_supervisor_client.inspect_source(
                source_id=source_id,
                source_version=source_version,
                source_kind=source_kind,
                marks=marks,
                origin_attestation=origin_attestation,
            )
        ver_num = self._validate_source_handle(source_id, source_version)
        if source_kind not in VALID_SOURCE_KINDS:
            raise ValueError(f"Unsupported source_kind: {source_kind}")
        key = (source_id, source_version)
        if key in self._sources:
            raise ValueError(
                f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Cannot overwrite {source_id}@{source_version}"
            )
        digest = self._store_verified_source(
            source_id,
            source_version,
            source_kind,
            marks,
            "verified_plane",
        )
        desc = VerifiedSourceDescriptor(
            source_id=source_id,
            source_version=source_version,
            source_kind=source_kind,
            content_sha256=digest,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            mark_count=len(marks),
            plane="verified_plane",
            created_at=FIXED_LOGICAL_TIMESTAMP,
        )
        self._sources[key] = desc
        prev_latest = self._latest_version_by_source.get(source_id)
        if prev_latest is None or ver_num > parse_version_num(prev_latest):
            self._latest_version_by_source[source_id] = source_version
        out = desc.to_dict()
        assert_no_raw_buffer_leakage(out)
        return out

    def release_source_pixels(self, source_keys: Sequence[Tuple[str, str]]) -> int:
        """
        Frees the stored 128x128 pixels of sources the caller has retired. Their descriptors stay
        registered, so a retired source_id@source_version can never be reused for other pixels.
        """
        keys = [[sid, sver] for sid, sver in source_keys]
        if not keys:
            return 0
        if self._out_of_process_vault_client is not None:
            return int(self._out_of_process_vault_client.evict_sources(keys))
        return int(self._evict_surfaces(keys))

    def mutate_canvas(
        self,
        source_id: str,
        base_version: str,
        new_version: str,
        updated_marks: Sequence[DrawingMark],
    ) -> Dict[str, Any]:
        """
        Mutating a canvas creates a new immutable source_version and never
        overwrites the base version in place.
        """
        base_num = self._validate_source_handle(source_id, base_version)
        new_num = self._validate_source_handle(source_id, new_version)
        base_key = (source_id, base_version)
        if base_key not in self._sources:
            raise KeyError(f"ERR_UNVERIFIED_SOURCE: Base {source_id}@{base_version} not found.")
        if new_num <= base_num or (source_id, new_version) in self._sources:
            raise ValueError(
                f"ERR_IMMUTABLE_SOURCE_OVERWRITE: Mutation must produce a strictly newer version (got {new_version} from {base_version})."
            )
        base_desc = self._sources[base_key]
        return self.register_verified_source(
            source_id=source_id,
            source_version=new_version,
            source_kind=base_desc.source_kind,
            marks=updated_marks,
        )

    # -------------------------------------------------------------------------
    # Layer 2: DeterministicMeasurement Engine (10 Closed-Form Operators)
    # -------------------------------------------------------------------------

    def _emit_measurement_envelope(
        self,
        *,
        status: str,
        source_id: Optional[str],
        source_version: Optional[str],
        secondary_source: Optional[Dict[str, str]],
        operator_name: str,
        parameters: Dict[str, Any],
        result: Any,
        confidence: str,
        refusal_code: Optional[str] = None,
        refusal_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._measurement_seq += 1
        meas_id = f"meas_{self._measurement_seq:04d}"
        digest = compute_replay_digest(
            source_id=source_id,
            source_version=source_version,
            secondary_source=secondary_source,
            operator_name=operator_name,
            operator_version=OPERATOR_VERSION,
            parameters=parameters,
            result=result,
        )
        envelope: Dict[str, Any] = {
            "measurement_id": meas_id,
            "status": status,
            "source_id": source_id,
            "source_version": source_version,
            "secondary_source": secondary_source,
            "operator_name": operator_name,
            "operator_version": OPERATOR_VERSION,
            "parameters": copy.deepcopy(parameters),
            "result": copy.deepcopy(result),
            "confidence": confidence,
            "timestamp": FIXED_LOGICAL_TIMESTAMP,
            "replay_digest": digest,
        }
        if refusal_code is not None:
            envelope["refusal_code"] = refusal_code
        if refusal_reason is not None:
            envelope["refusal_reason"] = refusal_reason
        assert_no_raw_buffer_leakage(envelope)
        self._measurements.append(copy.deepcopy(envelope))
        return envelope

    def _sanitize_refused_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not self._redact_refused_prompts:
            return parameters
        sanitized: Dict[str, Any] = {}
        for k, v in parameters.items():
            if k in ("prompt", "question", "semantic_hypothesis", "intent", "meaning", "interpretation", "aesthetic") and isinstance(v, str):
                sanitized[f"{k}_sha256"] = sha256_hex(v.encode("utf-8"))
                sanitized[f"{k}_length"] = len(v)
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_refused_parameters(v)
            else:
                sanitized[k] = v
        return sanitized

    @classmethod
    def _contains_nested_semantic_key(cls, obj: Any) -> bool:
        if isinstance(obj, dict):
            if set(obj.keys()) & SEMANTIC_KEY_INDICATORS:
                return True
            return any(cls._contains_nested_semantic_key(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(cls._contains_nested_semantic_key(item) for item in obj)
        return False

    @staticmethod
    def _validate_strict_int(val: Any, name: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(f"Parameter '{name}' must be an integer (got {type(val).__name__}).")
        if min_val is not None and val < min_val:
            raise ValueError(f"Parameter '{name}'={val} is below minimum {min_val}.")
        if max_val is not None and val > max_val:
            raise ValueError(f"Parameter '{name}'={val} exceeds maximum {max_val}.")
        return val

    @classmethod
    def _validate_nested_operator_parameters(
        cls,
        operator_name: str,
        parameters: Dict[str, Any],
        secondary_source: Optional[Dict[str, str]],
    ) -> None:
        """
        NEXT-GATE-4 (Addresses API-R1):
        Enforce closed inner key allowlists and strict value types/bounds for
        nested selector, predicate, point, and secondary_source dictionaries.
        """
        if secondary_source is not None:
            if not isinstance(secondary_source, dict):
                raise TypeError("secondary_source must be a dict.")
            sec_keys = set(secondary_source.keys())
            if sec_keys != ALLOWED_SECONDARY_SOURCE_KEYS:
                raise ValueError(
                    f"secondary_source must have exact keys {sorted(ALLOWED_SECONDARY_SOURCE_KEYS)}, got {sorted(sec_keys)}"
                )
            cls._validate_source_handle(secondary_source["source_id"], secondary_source["source_version"])

        if operator_name == "measure_bbox" and "selector" in parameters:
            sel = parameters["selector"]
            if not isinstance(sel, dict):
                raise TypeError("selector must be a dict.")
            extra_sel = sorted(set(sel.keys()) - ALLOWED_SELECTOR_KEYS)
            if extra_sel:
                raise ValueError(f"Unsupported nested keys in selector: {extra_sel}")
            sel_type = sel.get("type", "non_transparent")
            if sel_type not in ("non_transparent", "colour_match"):
                raise ValueError(f"Unsupported bbox selector type: {sel_type}")
            if "tolerance" in sel:
                cls._validate_strict_int(sel["tolerance"], "selector.tolerance", 0, 255)
            if "rgba" in sel:
                rgba = sel["rgba"]
                if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
                    raise ValueError("selector.rgba must be a 4-element RGBA list/tuple.")
                for idx, ch in enumerate(rgba):
                    cls._validate_strict_int(ch, f"selector.rgba[{idx}]", 0, 255)

        elif operator_name == "count_pixels" and "predicate" in parameters:
            pred = parameters["predicate"]
            if not isinstance(pred, dict):
                raise TypeError("predicate must be a dict.")
            extra_pred = sorted(set(pred.keys()) - ALLOWED_PREDICATE_KEYS)
            if extra_pred:
                raise ValueError(f"Unsupported nested keys in predicate: {extra_pred}")
            pred_type = pred.get("type", "non_transparent")
            if pred_type not in ("non_transparent", "colour_match"):
                raise ValueError(f"Unsupported predicate type: {pred_type}")
            if "tolerance" in pred:
                cls._validate_strict_int(pred["tolerance"], "predicate.tolerance", 0, 255)
            if "rgba" in pred:
                rgba = pred["rgba"]
                if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
                    raise ValueError("predicate.rgba must be a 4-element RGBA list/tuple.")
                for idx, ch in enumerate(rgba):
                    cls._validate_strict_int(ch, f"predicate.rgba[{idx}]", 0, 255)

        elif operator_name in ("measure_distance", "measure_angle"):
            for pt_name in ("p1", "p2"):
                if pt_name not in parameters:
                    raise ValueError(f"{operator_name} requires '{pt_name}'.")
                pt = parameters[pt_name]
                if not isinstance(pt, dict):
                    raise TypeError(f"'{pt_name}' must be a dict.")
                pt_keys = set(pt.keys())
                if pt_keys != ALLOWED_POINT_KEYS:
                    extra_pt = sorted(pt_keys - ALLOWED_POINT_KEYS)
                    missing_pt = sorted(ALLOWED_POINT_KEYS - pt_keys)
                    raise ValueError(
                        f"Nested point '{pt_name}' violates closed schema (extra={extra_pt}, missing={missing_pt})."
                    )
                cls._validate_strict_int(pt["x"], f"{pt_name}.x", 0, CANVAS_WIDTH - 1)
                cls._validate_strict_int(pt["y"], f"{pt_name}.y", 0, CANVAS_HEIGHT - 1)
            if operator_name == "measure_distance" and "metric" in parameters:
                metric = parameters["metric"]
                if not isinstance(metric, str) or metric not in ("euclidean", "manhattan", "chebyshev"):
                    raise ValueError(f"Unsupported metric: {metric}")

        elif operator_name == "sample_colour":
            if "x" not in parameters or "y" not in parameters:
                raise ValueError("sample_colour requires 'x' and 'y'.")
            cls._validate_strict_int(parameters["x"], "x", 0, CANVAS_WIDTH - 1)
            cls._validate_strict_int(parameters["y"], "y", 0, CANVAS_HEIGHT - 1)

        elif operator_name == "count_changed_pixels":
            if "channel_tolerance" in parameters:
                cls._validate_strict_int(parameters["channel_tolerance"], "channel_tolerance", 0, 255)

        elif operator_name == "extract_changed_regions":
            if "connectivity" in parameters:
                conn = cls._validate_strict_int(parameters["connectivity"], "connectivity")
                if conn not in (4, 8):
                    raise ValueError("connectivity must be 4 or 8")
            if "channel_tolerance" in parameters:
                cls._validate_strict_int(parameters["channel_tolerance"], "channel_tolerance", 0, 255)

        elif operator_name == "count_objects":
            if "connectivity" in parameters:
                conn = cls._validate_strict_int(parameters["connectivity"], "connectivity")
                if conn not in (4, 8):
                    raise ValueError("connectivity must be 4 or 8")
            if "min_area_px" in parameters:
                cls._validate_strict_int(parameters["min_area_px"], "min_area_px", 1, CANVAS_TOTAL_PIXELS)

    def execute_measurement(
        self,
        operator_name: str,
        parameters: Dict[str, Any],
        source_id: Optional[str] = None,
        source_version: Optional[str] = None,
        secondary_source: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Single entry point for all measurement queries.
        Strictly enforces:
        - Refusal Contract for semantic, aesthetic, intent, or free-text interpretive queries.
        - Closed per-operator top-level and nested parameter key schemas (GATE-3 & NEXT-GATE-4).
        """
        self._assert_supervisor_not_halted()
        if not isinstance(parameters, dict):
            return self._emit_measurement_envelope(
                status="error",
                source_id=source_id,
                source_version=source_version,
                secondary_source=secondary_source,
                operator_name=operator_name,
                parameters={},
                result="unknown",
                confidence="unknown",
                refusal_code="ERR_INVALID_PARAMETERS",
                refusal_reason="parameters must be a JSON object / dict.",
            )

        query_type = parameters.get("query_type")
        param_keys = set(parameters.keys())
        has_semantic_key = self._contains_nested_semantic_key(parameters)

        if (
            operator_name in PROHIBITED_SEMANTIC_QUERY_TYPES
            or query_type in PROHIBITED_SEMANTIC_QUERY_TYPES
            or operator_name not in CLOSED_FORM_OPERATORS
            or has_semantic_key
        ):
            self.semantic_queries_total += 1
            self.semantic_queries_refused += 1
            recorded_params = self._sanitize_refused_parameters(parameters)
            refused_env = self._emit_measurement_envelope(
                status="refused",
                source_id=source_id,
                source_version=source_version,
                secondary_source=secondary_source,
                operator_name=operator_name,
                parameters=recorded_params,
                result="unknown",
                confidence="unknown",
                refusal_code=REFUSAL_CODE_SEMANTIC,
                refusal_reason=REFUSAL_REASON_SEMANTIC,
            )
            self._append_audit_event(
                event_type="semantic_query_refused",
                target_id=refused_env["measurement_id"],
                error_code=REFUSAL_CODE_SEMANTIC,
                detail=f"Refused semantic/interpretive query on operator '{operator_name}'.",
            )
            return refused_env

        # GATE-3 (Fixes API-01): Enforce closed top-level parameter schema per operator
        allowed_keys = ALLOWED_OPERATOR_PARAM_KEYS[operator_name]
        extra_keys = sorted(param_keys - allowed_keys)
        if extra_keys:
            return self._emit_measurement_envelope(
                status="error",
                source_id=source_id,
                source_version=source_version,
                secondary_source=secondary_source,
                operator_name=operator_name,
                parameters=parameters,
                result="unknown",
                confidence="unknown",
                refusal_code="ERR_INVALID_PARAMETERS",
                refusal_reason=f"Unsupported parameter keys for {operator_name}: {extra_keys}",
            )

        # Resolve primary source if required by operator
        if operator_name not in ("measure_distance", "measure_angle"):
            if source_id is None or source_version is None:
                return self._emit_measurement_envelope(
                    status="error",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=secondary_source,
                    operator_name=operator_name,
                    parameters=parameters,
                    result="unknown",
                    confidence="unknown",
                    refusal_code="ERR_UNVERIFIED_SOURCE",
                    refusal_reason="Missing source_id or source_version.",
                )
            if (source_id, source_version) not in self._sources:
                return self._emit_measurement_envelope(
                    status="error",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=secondary_source,
                    operator_name=operator_name,
                    parameters=parameters,
                    result="unknown",
                    confidence="unknown",
                    refusal_code="ERR_UNVERIFIED_SOURCE",
                    refusal_reason=f"Source {source_id}@{source_version} is not a registered VerifiedSource.",
                )

        try:
            # NEXT-GATE-4: Enforce closed nested schemas and strict value types/bounds
            self._validate_nested_operator_parameters(operator_name, parameters, secondary_source)
            if operator_name == "inspect_source":
                desc = self._sources[(source_id, source_version)]  # type: ignore[index]
                result = {
                    "source_id": desc.source_id,
                    "source_version": desc.source_version,
                    "source_kind": desc.source_kind,
                    "content_sha256": desc.content_sha256,
                    "dimensions": {"width": desc.width, "height": desc.height},
                    "mark_count": desc.mark_count,
                    "plane": desc.plane,
                    "observed": False,
                    "observation_claim": False,
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "measure_dimensions":
                desc = self._sources[(source_id, source_version)]  # type: ignore[index]
                result = {
                    "width": desc.width,
                    "height": desc.height,
                    "total_pixels": desc.width * desc.height,
                    "aspect_ratio": "1:1",
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if self._out_of_process_vault_client is not None and operator_name in (
                "measure_bbox",
                "sample_colour",
                "count_pixels",
                "count_changed_pixels",
                "extract_changed_regions",
                "count_objects",
            ):
                if operator_name in ("count_changed_pixels", "extract_changed_regions"):
                    if secondary_source is None:
                        raise ValueError(f"{operator_name} requires secondary_source")
                    sec_key = (secondary_source["source_id"], secondary_source["source_version"])
                    if sec_key not in self._sources:
                        raise KeyError(
                            f"ERR_UNVERIFIED_SOURCE: Secondary {sec_key[0]}@{sec_key[1]} not found."
                        )
                result = self._out_of_process_vault_client.run_pixel_operator(
                    operator_name=operator_name,
                    parameters=parameters,
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=secondary_source,
                )
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=(
                        secondary_source
                        if operator_name in ("count_changed_pixels", "extract_changed_regions")
                        else None
                    ),
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "measure_bbox":
                surface = self._get_surface(source_id, source_version)  # type: ignore[arg-type]
                selector = parameters.get("selector", {"type": "non_transparent"})
                sel_type = selector.get("type", "non_transparent")
                target_rgba = tuple(selector.get("rgba", [0, 0, 0, 0]))
                tolerance = int(selector.get("tolerance", 0))

                x_min, y_min = CANVAS_WIDTH, CANVAS_HEIGHT
                x_max, y_max = -1, -1
                matched = 0

                if sel_type == "non_transparent":
                    for y, x0, x1, _rgba in surface.fg_runs:
                        matched += x1 - x0 + 1
                        if x0 < x_min:
                            x_min = x0
                        if x1 > x_max:
                            x_max = x1
                        if y < y_min:
                            y_min = y
                        if y > y_max:
                            y_max = y
                elif sel_type == "colour_match" and target_rgba[3] > tolerance:
                    tr, tg, tb, ta = target_rgba
                    for y, x0, x1, rgba in surface.fg_runs:
                        if tolerance == 0:
                            hit = rgba == target_rgba
                        else:
                            hit = (
                                abs(rgba[0] - tr) <= tolerance
                                and abs(rgba[1] - tg) <= tolerance
                                and abs(rgba[2] - tb) <= tolerance
                                and abs(rgba[3] - ta) <= tolerance
                            )
                        if hit:
                            matched += x1 - x0 + 1
                            if x0 < x_min:
                                x_min = x0
                            if x1 > x_max:
                                x_max = x1
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y
                elif sel_type == "colour_match":
                    tr, tg, tb, ta = target_rgba
                    raw = surface.raw_bytes
                    for idx in range(CANVAS_TOTAL_PIXELS):
                        off = idx << 2
                        if (
                            abs(raw[off] - tr) <= tolerance
                            and abs(raw[off + 1] - tg) <= tolerance
                            and abs(raw[off + 2] - tb) <= tolerance
                            and abs(raw[off + 3] - ta) <= tolerance
                        ):
                            matched += 1
                            y = idx >> 7
                            x = idx & 127
                            if x < x_min:
                                x_min = x
                            if x > x_max:
                                x_max = x
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y
                else:
                    raise ValueError(f"Unsupported bbox selector type: {sel_type}")

                if matched == 0:
                    result = {
                        "empty": True,
                        "x_min": None,
                        "y_min": None,
                        "x_max": None,
                        "y_max": None,
                        "width": 0,
                        "height": 0,
                        "matching_pixels": 0,
                    }
                else:
                    result = {
                        "empty": False,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "width": x_max - x_min + 1,
                        "height": y_max - y_min + 1,
                        "matching_pixels": matched,
                    }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "sample_colour":
                x = int(parameters["x"])
                y = int(parameters["y"])
                if not (0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT):
                    raise ValueError(f"Coordinates ({x}, {y}) out of 128x128 bounds.")
                r, g, b, a = self._sample_pixel(source_id, source_version, x, y)  # type: ignore[arg-type]
                result = {
                    "x": x,
                    "y": y,
                    "rgba": [r, g, b, a],
                    "hex": f"#{r:02X}{g:02X}{b:02X}{a:02X}",
                    "sampled_plane": "verified_plane",
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "measure_distance":
                p1 = parameters["p1"]
                p2 = parameters["p2"]
                metric = parameters.get("metric", "euclidean")
                dx = int(p2["x"]) - int(p1["x"])
                dy = int(p2["y"]) - int(p1["y"])
                dist_sq = dx * dx + dy * dy
                if metric == "euclidean":
                    dist_val = round(math.sqrt(dist_sq), 6)
                elif metric == "manhattan":
                    dist_val = float(abs(dx) + abs(dy))
                elif metric == "chebyshev":
                    dist_val = float(max(abs(dx), abs(dy)))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                result = {
                    "dx": dx,
                    "dy": dy,
                    "distance_exact_sq": dist_sq,
                    "distance": dist_val,
                    "metric": metric,
                    "unit": "px",
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "measure_angle":
                p1 = parameters["p1"]
                p2 = parameters["p2"]
                dx = int(p2["x"]) - int(p1["x"])
                dy = int(p2["y"]) - int(p1["y"])
                rad = math.atan2(dy, dx)
                deg = math.degrees(rad)
                result = {
                    "dx": dx,
                    "dy": dy,
                    "radians": round(rad, 6),
                    "degrees": round(deg, 6),
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "count_pixels":
                surface = self._get_surface(source_id, source_version)  # type: ignore[arg-type]
                predicate = parameters.get("predicate", {"type": "non_transparent"})
                pred_type = predicate.get("type", "non_transparent")
                target_rgba = tuple(predicate.get("rgba", [0, 0, 0, 0]))
                tolerance = int(predicate.get("tolerance", 0))

                matching = 0
                if pred_type == "non_transparent":
                    for _y, x0, x1, _rgba in surface.fg_runs:
                        matching += x1 - x0 + 1
                elif pred_type == "colour_match" and target_rgba[3] > tolerance:
                    tr, tg, tb, ta = target_rgba
                    for _y, x0, x1, rgba in surface.fg_runs:
                        if tolerance == 0:
                            if rgba == target_rgba:
                                matching += x1 - x0 + 1
                        else:
                            if (
                                abs(rgba[0] - tr) <= tolerance
                                and abs(rgba[1] - tg) <= tolerance
                                and abs(rgba[2] - tb) <= tolerance
                                and abs(rgba[3] - ta) <= tolerance
                            ):
                                matching += x1 - x0 + 1
                elif pred_type == "colour_match":
                    tr, tg, tb, ta = target_rgba
                    raw = surface.raw_bytes
                    for idx in range(CANVAS_TOTAL_PIXELS):
                        off = idx << 2
                        if (
                            abs(raw[off] - tr) <= tolerance
                            and abs(raw[off + 1] - tg) <= tolerance
                            and abs(raw[off + 2] - tb) <= tolerance
                            and abs(raw[off + 3] - ta) <= tolerance
                        ):
                            matching += 1
                else:
                    raise ValueError(f"Unsupported predicate type: {pred_type}")

                result = {
                    "matching_pixels": matching,
                    "total_pixels": CANVAS_TOTAL_PIXELS,
                    "fraction": round(matching / CANVAS_TOTAL_PIXELS, 6),
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "count_changed_pixels":
                if secondary_source is None:
                    raise ValueError("count_changed_pixels requires secondary_source")
                sec_id = secondary_source["source_id"]
                sec_ver = secondary_source["source_version"]
                surf_a = self._get_surface(source_id, source_version)  # type: ignore[arg-type]
                surf_b = self._get_surface(sec_id, sec_ver)
                tol = int(parameters.get("channel_tolerance", 0))
                if tol < 0 or tol > 255:
                    raise ValueError("channel_tolerance must be in [0, 255]")

                changed = 0
                added = 0
                removed = 0
                modified_colour = 0
                rows_a = surf_a.row_bytes
                rows_b = surf_b.row_bytes

                for y in range(CANVAS_HEIGHT):
                    ra = rows_a[y]
                    rb = rows_b[y]
                    if ra == rb:
                        continue
                    if tol == 0:
                        for x4 in range(0, ROW_BYTE_STRIDE, 4):
                            if ra[x4 : x4 + 4] != rb[x4 : x4 + 4]:
                                changed += 1
                                aa = ra[x4 + 3]
                                ab = rb[x4 + 3]
                                if aa == 0 and ab > 0:
                                    added += 1
                                elif aa > 0 and ab == 0:
                                    removed += 1
                                else:
                                    modified_colour += 1
                    else:
                        for x4 in range(0, ROW_BYTE_STRIDE, 4):
                            if (
                                abs(ra[x4] - rb[x4]) > tol
                                or abs(ra[x4 + 1] - rb[x4 + 1]) > tol
                                or abs(ra[x4 + 2] - rb[x4 + 2]) > tol
                                or abs(ra[x4 + 3] - rb[x4 + 3]) > tol
                            ):
                                changed += 1
                                aa = ra[x4 + 3]
                                ab = rb[x4 + 3]
                                if aa == 0 and ab > 0:
                                    added += 1
                                elif aa > 0 and ab == 0:
                                    removed += 1
                                else:
                                    modified_colour += 1

                result = {
                    "changed_pixels": changed,
                    "unchanged_pixels": CANVAS_TOTAL_PIXELS - changed,
                    "added_pixels": added,
                    "removed_pixels": removed,
                    "modified_colour_pixels": modified_colour,
                    "total_pixels": CANVAS_TOTAL_PIXELS,
                    "change_ratio": round(changed / CANVAS_TOTAL_PIXELS, 6),
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=secondary_source,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "extract_changed_regions":
                if secondary_source is None:
                    raise ValueError("extract_changed_regions requires secondary_source")
                sec_id = secondary_source["source_id"]
                sec_ver = secondary_source["source_version"]
                surf_a = self._get_surface(source_id, source_version)  # type: ignore[arg-type]
                surf_b = self._get_surface(sec_id, sec_ver)
                connectivity = int(parameters.get("connectivity", 8))
                if connectivity not in (4, 8):
                    raise ValueError("connectivity must be 4 or 8")
                tol = int(parameters.get("channel_tolerance", 0))

                diff_runs = self._extract_diff_runs(surf_a, surf_b, tol=tol)
                regions = self._connected_components_from_runs(diff_runs, connectivity=connectivity)
                result = {
                    "region_count": len(regions),
                    "connectivity": connectivity,
                    "regions": regions[:MAX_REPORTED_REGIONS],
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=secondary_source,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            if operator_name == "count_objects":
                connectivity = int(parameters.get("connectivity", 8))
                if connectivity not in (4, 8):
                    raise ValueError("connectivity must be 4 or 8")
                min_area = int(parameters.get("min_area_px", 1))
                all_objs = self._get_foreground_components(
                    source_id, source_version, connectivity=connectivity  # type: ignore[arg-type]
                )
                filtered = [copy.deepcopy(o) for o in all_objs if o["area_px"] >= min_area]
                for idx, obj in enumerate(filtered, start=1):
                    obj["object_id"] = f"obj_{idx:02d}"
                result = {
                    "object_count": len(filtered),
                    "connectivity": connectivity,
                    "min_area_px": min_area,
                    "objects": filtered[:MAX_REPORTED_REGIONS],
                }
                return self._emit_measurement_envelope(
                    status="ok",
                    source_id=source_id,
                    source_version=source_version,
                    secondary_source=None,
                    operator_name=operator_name,
                    parameters=parameters,
                    result=result,
                    confidence="high",
                )

            raise ValueError(f"Unreachable operator: {operator_name}")

        except (KeyError, ValueError, TypeError) as exc:
            # GATE-3 (Fixes SEC-03): Return deterministic structured error envelope
            return self._emit_measurement_envelope(
                status="error",
                source_id=source_id,
                source_version=source_version,
                secondary_source=secondary_source,
                operator_name=operator_name,
                parameters=parameters,
                result="unknown",
                confidence="unknown",
                refusal_code="ERR_INVALID_PARAMETERS",
                refusal_reason=str(exc),
            )

    @staticmethod
    def _extract_diff_runs(
        surf_a: _RasterizedSurface,
        surf_b: _RasterizedSurface,
        tol: int = 0,
    ) -> List[Tuple[int, int, int]]:
        rows_a = surf_a.row_bytes
        rows_b = surf_b.row_bytes
        runs: List[Tuple[int, int, int]] = []

        for y in range(CANVAS_HEIGHT):
            ra = rows_a[y]
            rb = rows_b[y]
            if ra == rb:
                continue
            x = 0
            if tol == 0:
                while x < CANVAS_WIDTH:
                    x4 = x << 2
                    if ra[x4 : x4 + 4] != rb[x4 : x4 + 4]:
                        x0 = x
                        x += 1
                        while x < CANVAS_WIDTH:
                            nx4 = x << 2
                            if ra[nx4 : nx4 + 4] != rb[nx4 : nx4 + 4]:
                                x += 1
                            else:
                                break
                        runs.append((y, x0, x - 1))
                    else:
                        x += 1
            else:
                while x < CANVAS_WIDTH:
                    x4 = x << 2
                    if (
                        abs(ra[x4] - rb[x4]) > tol
                        or abs(ra[x4 + 1] - rb[x4 + 1]) > tol
                        or abs(ra[x4 + 2] - rb[x4 + 2]) > tol
                        or abs(ra[x4 + 3] - rb[x4 + 3]) > tol
                    ):
                        x0 = x
                        x += 1
                        while x < CANVAS_WIDTH:
                            nx4 = x << 2
                            if (
                                abs(ra[nx4] - rb[nx4]) > tol
                                or abs(ra[nx4 + 1] - rb[nx4 + 1]) > tol
                                or abs(ra[nx4 + 2] - rb[nx4 + 2]) > tol
                                or abs(ra[nx4 + 3] - rb[nx4 + 3]) > tol
                            ):
                                x += 1
                            else:
                                break
                        runs.append((y, x0, x - 1))
                    else:
                        x += 1
        return runs

    def _get_foreground_components(
        self,
        source_id: str,
        source_version: str,
        connectivity: int = 8,
    ) -> List[Dict[str, Any]]:
        cache_key = (source_id, source_version, connectivity)
        cached = self._cc_cache.get(cache_key)
        if cached is not None:
            self._cc_cache.move_to_end(cache_key)
            return cached
        surface = self._get_surface(source_id, source_version)
        merged_runs: List[Tuple[int, int, int]] = []
        for y, x0, x1, _rgba in surface.fg_runs:
            if merged_runs and merged_runs[-1][0] == y and merged_runs[-1][2] + 1 == x0:
                merged_runs[-1] = (y, merged_runs[-1][1], x1)
            else:
                merged_runs.append((y, x0, x1))
        comps = self._connected_components_from_runs(merged_runs, connectivity=connectivity)
        self._cc_cache[cache_key] = comps
        if len(self._cc_cache) > MAX_CACHE_ENTRIES:
            self._cc_cache.popitem(last=False)
        return comps

    @staticmethod
    def _connected_components_from_runs(
        runs: Sequence[Tuple[int, int, int]],
        connectivity: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Exact connected-components algorithm over sorted horizontal RLE runs (y, x0, x1).
        Produces byte-identical region_id, area_px, bbox, centroid, rle_run_count,
        and rle_mask_digest as pixel-by-pixel scanline BFS in O(runs) time.
        """
        n_runs = len(runs)
        if n_runs == 0:
            return []

        runs_by_row: Dict[int, List[int]] = {}
        for idx, (y, _x0, _x1) in enumerate(runs):
            runs_by_row.setdefault(y, []).append(idx)

        visited = [False] * n_runs
        components: List[Dict[str, Any]] = []
        diag_pad = 1 if connectivity == 8 else 0
        comp_idx = 0

        for start_i in range(n_runs):
            if visited[start_i]:
                continue
            comp_idx += 1
            visited[start_i] = True
            queue: deque[int] = deque([start_i])
            comp_runs: List[Tuple[int, int, int]] = []

            x_min = y_min = 1 << 62  # (was CANVAS_WIDTH, CANVAS_HEIGHT, 128: the native components run this at full resolution, so a region wholly beyond 128 kept 128; fixed 2026-10-30)
            x_max, y_max = -1, -1
            sum_x = 0
            sum_y = 0
            area = 0

            while queue:
                curr_i = queue.popleft()
                cy, cx0, cx1 = runs[curr_i]
                comp_runs.append((cy, cx0, cx1))

                span = cx1 - cx0 + 1
                area += span
                sum_x += (cx0 + cx1) * span // 2
                sum_y += cy * span
                if cx0 < x_min:
                    x_min = cx0
                if cx1 > x_max:
                    x_max = cx1
                if cy < y_min:
                    y_min = cy
                if cy > y_max:
                    y_max = cy

                for adj_y in (cy - 1, cy + 1):
                    for cand_i in runs_by_row.get(adj_y, ()):
                        if not visited[cand_i]:
                            _ny, nx0, nx1 = runs[cand_i]
                            if cx0 <= nx1 + diag_pad and nx0 <= cx1 + diag_pad:
                                visited[cand_i] = True
                                queue.append(cand_i)

            comp_runs.sort()
            rle_digest = sha256_hex(canonical_json(comp_runs).encode("utf-8"))
            components.append({
                "region_id": f"reg_{comp_idx:02d}",
                "area_px": area,
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": x_max - x_min + 1,
                    "height": y_max - y_min + 1,
                },
                "centroid": {
                    "x": round(sum_x / area, 4),
                    "y": round(sum_y / area, 4),
                },
                "rle_run_count": len(comp_runs),
                "rle_mask_digest": rle_digest,
            })
        return components

    def record_replay_pair(self, run_a: Dict[str, Any], run_b: Dict[str, Any]) -> bool:
        self._replayed_pairs.append((copy.deepcopy(run_a), copy.deepcopy(run_b)))
        return (
            canonical_json(run_a["result"]) == canonical_json(run_b["result"])
            and run_a["replay_digest"] == run_b["replay_digest"]
            and canonical_json(run_a["parameters"]) == canonical_json(run_b["parameters"])
        )

    @staticmethod
    def verify_measurement_provenance(envelope: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = (
            "measurement_id",
            "source_id",
            "source_version",
            "secondary_source",
            "operator_name",
            "operator_version",
            "parameters",
            "result",
            "confidence",
            "timestamp",
            "replay_digest",
        )
        missing = [
            f for f in required_fields
            if f not in envelope or (
                f not in ("secondary_source", "source_id", "source_version")
                and (envelope[f] is None or envelope[f] == "")
            )
        ]
        if missing:
            return {
                "valid": False,
                "error_code": "ERR_INCOMPLETE_PROVENANCE",
                "missing_fields": missing,
            }
        expected_digest = compute_replay_digest(
            source_id=envelope["source_id"],
            source_version=envelope["source_version"],
            secondary_source=envelope["secondary_source"],
            operator_name=envelope["operator_name"],
            operator_version=envelope["operator_version"],
            parameters=envelope["parameters"],
            result=envelope["result"],
        )
        if expected_digest != envelope["replay_digest"]:
            return {
                "valid": False,
                "error_code": "ERR_PROVENANCE_DIGEST_MISMATCH",
                "expected_digest": expected_digest,
                "actual_digest": envelope["replay_digest"],
            }
        return {
            "valid": True,
            "error_code": None,
            "verified_digest": expected_digest,
        }

    # -------------------------------------------------------------------------
    # Layer 3: DerivedRepresentation Engine
    # -------------------------------------------------------------------------

    def _emit_derived_representation(
        self,
        *,
        representation_type: str,
        source_refs: List[Dict[str, str]],
        parameters: Dict[str, Any],
        summary: Dict[str, Any],
        confidence: str,
        rationale: str,
    ) -> Dict[str, Any]:
        if confidence not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence: {confidence}")
        if confidence != "high" and not rationale.strip():
            raise ValueError("Rationale is mandatory when confidence < high.")
        self._derived_seq += 1
        rep_id = f"rep_{self._derived_seq:04d}_{representation_type}"
        manifest = {
            "representation_id": rep_id,
            "epistemic_label": "derived",
            "representation_type": representation_type,
            "source_refs": copy.deepcopy(source_refs),
            "operator_version": OPERATOR_VERSION,
            "parameters": copy.deepcopy(parameters),
            "confidence": confidence,
            "rationale": rationale,
            "summary": copy.deepcopy(summary),
            "timestamp": FIXED_LOGICAL_TIMESTAMP,
        }
        assert_no_raw_buffer_leakage(manifest)
        self._derived_representations[rep_id] = copy.deepcopy(manifest)
        return manifest

    def compute_derived_representation(
        self,
        representation_type: str,
        source_a: Tuple[str, str],
        source_b: Tuple[str, str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._assert_supervisor_not_halted()
        params = parameters or {}
        desc_a = self._sources[source_a]
        desc_b = self._sources[source_b]
        source_refs = [
            {
                "source_id": desc_a.source_id,
                "source_version": desc_a.source_version,
                "content_sha256": desc_a.content_sha256,
            },
            {
                "source_id": desc_b.source_id,
                "source_version": desc_b.source_version,
                "content_sha256": desc_b.content_sha256,
            },
        ]
        if self._out_of_process_vault_client is not None:
            derived_info = self._out_of_process_vault_client.compute_derived_summary(
                representation_type=representation_type,
                source_a=source_a,
                source_b=source_b,
                parameters=params,
            )
            return self._emit_derived_representation(
                representation_type=representation_type,
                source_refs=source_refs,
                parameters=params,
                summary=derived_info["summary"],
                confidence=derived_info["confidence"],
                rationale=derived_info["rationale"],
            )
        surf_a = self._get_surface(*source_a)
        surf_b = self._get_surface(*source_b)
        rows_a = surf_a.row_bytes
        rows_b = surf_b.row_bytes

        if representation_type == "change_mask":
            added, removed, modified, unchanged = 0, 0, 0, 0
            rle_changed: List[Tuple[int, int, str]] = []
            for y in range(CANVAS_HEIGHT):
                ra = rows_a[y]
                rb = rows_b[y]
                if ra == rb:
                    unchanged += CANVAS_WIDTH
                    continue
                row_start = y << 7
                for x in range(CANVAS_WIDTH):
                    x4 = x << 2
                    if ra[x4 : x4 + 4] == rb[x4 : x4 + 4]:
                        unchanged += 1
                    else:
                        idx = row_start + x
                        aa = ra[x4 + 3]
                        ab = rb[x4 + 3]
                        if aa == 0 and ab > 0:
                            added += 1
                            state = "added"
                        elif aa > 0 and ab == 0:
                            removed += 1
                            state = "removed"
                        else:
                            modified += 1
                            state = "modified"
                        if rle_changed and rle_changed[-1][2] == state and rle_changed[-1][1] + 1 == idx:
                            rle_changed[-1] = (rle_changed[-1][0], idx, state)
                        else:
                            rle_changed.append((idx, idx, state))

            mask_digest = sha256_hex(canonical_json(rle_changed).encode("utf-8"))
            summary = {
                "added_pixels": added,
                "removed_pixels": removed,
                "modified_pixels": modified,
                "changed_pixels": added + removed + modified,
                "unchanged_pixels": unchanged,
                "rle_span_count": len(rle_changed),
                "change_mask_digest": mask_digest,
            }
            return self._emit_derived_representation(
                representation_type="change_mask",
                source_refs=source_refs,
                parameters=params,
                summary=summary,
                confidence="high",
                rationale="Exact pixel-by-pixel ternary comparison across two immutable VerifiedSources.",
            )

        if representation_type == "difference_map":
            max_l1 = 0
            sum_l1 = 0
            non_zero_deltas = 0
            delta_spans: List[Tuple[int, int, int, int, int]] = []
            for y in range(CANVAS_HEIGHT):
                ra = rows_a[y]
                rb = rows_b[y]
                if ra == rb:
                    continue
                row_start = y << 7
                for x in range(CANVAS_WIDTH):
                    x4 = x << 2
                    if ra[x4 : x4 + 4] != rb[x4 : x4 + 4]:
                        dr = abs(ra[x4] - rb[x4])
                        dg = abs(ra[x4 + 1] - rb[x4 + 1])
                        db = abs(ra[x4 + 2] - rb[x4 + 2])
                        da = abs(ra[x4 + 3] - rb[x4 + 3])
                        l1 = dr + dg + db + da
                        non_zero_deltas += 1
                        sum_l1 += l1
                        if l1 > max_l1:
                            max_l1 = l1
                        delta_spans.append((row_start + x, dr, dg, db, da))

            diff_digest = sha256_hex(canonical_json(delta_spans).encode("utf-8"))
            summary = {
                "non_zero_delta_pixels": non_zero_deltas,
                "max_channel_l1_delta": max_l1,
                "mean_channel_l1_delta_on_changed": round(sum_l1 / max(1, non_zero_deltas), 4),
                "difference_map_digest": diff_digest,
            }
            return self._emit_derived_representation(
                representation_type="difference_map",
                source_refs=source_refs,
                parameters=params,
                summary=summary,
                confidence="high",
                rationale="Exact per-channel absolute difference map over verified RGBA buffers.",
            )

        if representation_type == "motion_summary":
            objs_a = self._get_foreground_components(source_a[0], source_a[1], connectivity=8)
            objs_b = self._get_foreground_components(source_b[0], source_b[1], connectivity=8)
            vectors = []
            for oa, ob in zip(objs_a, objs_b):
                dx = round(ob["centroid"]["x"] - oa["centroid"]["x"], 4)
                dy = round(ob["centroid"]["y"] - oa["centroid"]["y"], 4)
                vectors.append({
                    "region_id_a": oa["region_id"],
                    "region_id_b": ob["region_id"],
                    "area_px_a": oa["area_px"],
                    "area_px_b": ob["area_px"],
                    "displacement": {"dx": dx, "dy": dy},
                })
            return self._emit_derived_representation(
                representation_type="motion_summary",
                source_refs=source_refs,
                parameters=params,
                summary={"component_trajectories": vectors},
                confidence="medium",
                rationale=(
                    "Component correspondence between discrete frames matches connected components "
                    "by area and scanline order; non-rigid or occluded motion is not uniquely identifiable."
                ),
            )

        if representation_type == "colour_delta":
            changed_count = 0
            sum_delta_e = 0.0
            max_delta_e = 0.0
            for y in range(CANVAS_HEIGHT):
                ra = rows_a[y]
                rb = rows_b[y]
                if ra == rb:
                    continue
                for x4 in range(0, ROW_BYTE_STRIDE, 4):
                    if ra[x4 : x4 + 4] != rb[x4 : x4 + 4]:
                        changed_count += 1
                        dr = ra[x4] - rb[x4]
                        dg = ra[x4 + 1] - rb[x4 + 1]
                        db = ra[x4 + 2] - rb[x4 + 2]
                        da = ra[x4 + 3] - rb[x4 + 3]
                        de = math.sqrt(dr * dr + dg * dg + db * db + da * da)
                        sum_delta_e += de
                        if de > max_delta_e:
                            max_delta_e = de
            summary = {
                "changed_pixel_count": changed_count,
                "mean_rgba_euclidean_delta": round(sum_delta_e / max(1, changed_count), 4),
                "max_rgba_euclidean_delta": round(max_delta_e, 4),
            }
            return self._emit_derived_representation(
                representation_type="colour_delta",
                source_refs=source_refs,
                parameters=params,
                summary=summary,
                confidence="high",
                rationale="Deterministic Euclidean RGBA colour delta across changed pixels.",
            )

        if representation_type == "spatial_relationship":
            objs_b = self._get_foreground_components(source_b[0], source_b[1], connectivity=8)
            if len(objs_b) >= 2:
                o1, o2 = objs_b[0], objs_b[1]
                b1, b2 = o1["bbox"], o2["bbox"]
                intersects = not (
                    b1["x_max"] < b2["x_min"]
                    or b2["x_max"] < b1["x_min"]
                    or b1["y_max"] < b2["y_min"]
                    or b2["y_max"] < b1["y_min"]
                )
                relation = "intersects" if intersects else "disjoint"
                dx_gap = max(0, b2["x_min"] - b1["x_max"], b1["x_min"] - b2["x_max"])
                dy_gap = max(0, b2["y_min"] - b1["y_max"], b1["y_min"] - b2["y_max"])
                clearance_px = round(math.sqrt(dx_gap ** 2 + dy_gap ** 2), 4)
                bearing_deg = round(
                    math.degrees(
                        math.atan2(
                            o2["centroid"]["y"] - o1["centroid"]["y"],
                            o2["centroid"]["x"] - o1["centroid"]["x"],
                        )
                    ),
                    4,
                )
                summary = {
                    "primary_region": o1["region_id"],
                    "secondary_region": o2["region_id"],
                    "topological_relation": relation,
                    "edge_clearance_px": clearance_px,
                    "centroid_bearing_deg": bearing_deg,
                }
            else:
                summary = {
                    "topological_relation": "single_region_only",
                    "region_count": len(objs_b),
                }
            return self._emit_derived_representation(
                representation_type="spatial_relationship",
                source_refs=source_refs,
                parameters=params,
                summary=summary,
                confidence="high",
                rationale="Exact bounding-box topology, clearance, and centroid bearing on verified mask.",
            )

        raise ValueError(f"Unsupported DerivedRepresentation type: {representation_type}")

    # -------------------------------------------------------------------------
    # Layer 4: SemanticInterpretation Boundary Validator
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_external_semantic_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Layer 4 contract: Measurement operators never produce semantic interpretations.
        If external semantic metadata is evaluated at the boundary, it must be
        explicitly labelled as derived, inferred, or uncertain with explicit confidence,
        and can never claim observed=True or VerifiedSource status.
        """
        label = annotation.get("epistemic_label")
        conf = annotation.get("confidence")
        if label not in VALID_SEMANTIC_EPISTEMIC_LABELS:
            raise ValueError(
                f"ERR_INVALID_SEMANTIC_LABEL: External semantic interpretation must be labelled "
                f"derived, inferred, or uncertain (got {label!r})."
            )
        if conf not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(
                f"ERR_INVALID_SEMANTIC_CONFIDENCE: External semantic interpretation must carry "
                f"explicit confidence in {sorted(VALID_CONFIDENCE_LEVELS)} (got {conf!r})."
            )
        if annotation.get("observed") is True or annotation.get("is_verified_source") is True:
            raise PermissionError(
                "ERR_SEMANTIC_OBSERVATION_CONFLATION: Semantic interpretations cannot claim verified observation or source status."
            )
        return {
            "epistemic_label": label,
            "confidence": conf,
            "observed": False,
            "is_verified_source": False,
            "is_measurement": False,
        }

    # -------------------------------------------------------------------------
    # Layer 5: ElectricRender & ObservationWindowGate
    # -------------------------------------------------------------------------

    def format_uncertainty_overlay(
        self,
        uncertainty_state: str,
        *,
        detail: str = "",
        color_mode: bool = True,
    ) -> Dict[str, Any]:
        if uncertainty_state not in VALID_UNCERTAINTY_STATES:
            raise ValueError(f"Invalid uncertainty_state: {uncertainty_state}")
        spec = UNCERTAINTY_STYLE_SPEC[uncertainty_state]
        suffix = f" · {detail}" if detail else ""
        badge_text = f"{spec['glyph']} [STATE: {uncertainty_state.upper()}{suffix}]"
        overlay: Dict[str, Any] = {
            "uncertainty_state": uncertainty_state,
            "uncertainty_badge_text": badge_text,
            "border_pattern": spec["border_pattern"],
            "pattern_name": spec["pattern_name"],
            "glyph": spec["glyph"],
            "color_mode": color_mode,
            "glow_colour_hex": spec["glow_colour_hex"] if color_mode else None,
            "glow_name": spec["glow_name"] if color_mode else "monochrome_pattern_only",
        }
        return overlay

    def evaluate_render_staleness(
        self,
        render_id: str,
        color_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        GATE-4 & NEXT-GATE-4 (Fixes ACC-01, ACC-R1): Dynamically checks whether any
        source referenced in a render's provenance_refs has advanced beyond the render's
        referenced version(s). If so, returns an updated copy of the render with
        uncertainty_state='stale', sawtooth border pattern '⋀⋁⋀⋁', and version-lag badge text.
        """
        if render_id not in self._renders:
            raise KeyError(f"Unknown render_id: {render_id}")
        render = copy.deepcopy(self._renders[render_id])
        max_ref_ver_by_source: Dict[str, int] = {}
        for ref in render["provenance_refs"]:
            if ref.get("ref_type") == "source":
                sid = ref["ref_id"]
                ref_ver_num = parse_version_num(ref["ref_version"])
                prev_max = max_ref_ver_by_source.get(sid, 0)
                if ref_ver_num > prev_max:
                    max_ref_ver_by_source[sid] = ref_ver_num

        max_lag = 0
        for sid, ref_ver_num in max_ref_ver_by_source.items():
            latest_ver = self._latest_version_by_source.get(sid)
            if latest_ver is not None:
                lag = parse_version_num(latest_ver) - ref_ver_num
                if lag > max_lag:
                    max_lag = lag

        if max_lag > 0:
            overlay = self.format_uncertainty_overlay(
                "stale",
                detail=f"BEHIND BY {max_lag} VERSION{'S' if max_lag != 1 else ''}",
                color_mode=color_mode,
            )
            render["uncertainty_state"] = "stale"
            render["uncertainty_badge_text"] = overlay["uncertainty_badge_text"]
            render["border_pattern"] = overlay["border_pattern"]
            render["glow_colour_hex"] = overlay["glow_colour_hex"]
        return render

    def create_electric_render(
        self,
        *,
        render_type: str,
        source: str,
        confidence: str,
        uncertainty_state: str,
        provenance_refs: List[Dict[str, str]],
        uncertainty_detail: str = "",
        visibility_policy: Optional[str] = None,
        layer_isolation: Optional[str] = None,
        ghost_marks: Optional[Sequence[DrawingMark]] = None,
        claim_observed_without_receipt: bool = False,
        color_mode: bool = True,
    ) -> Dict[str, Any]:
        self._assert_supervisor_not_halted()
        if render_type not in VALID_RENDER_TYPES:
            raise ValueError(f"Invalid render_type: {render_type}")
        if source not in VALID_RENDER_SOURCES:
            raise ValueError(f"Invalid render source: {source}")
        if confidence not in VALID_CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence: {confidence}")
        if uncertainty_state not in VALID_UNCERTAINTY_STATES:
            raise ValueError(f"Invalid uncertainty_state: {uncertainty_state}")
        if not provenance_refs:
            raise ValueError("provenance_refs must be non-empty for every ElectricRender.")

        # Enforce default visibility_policy and layer_isolation by render_type
        if visibility_policy is None:
            if render_type == "diagnostic":
                visibility_policy = "silent_diagnostic"
            elif render_type == "decorative":
                visibility_policy = "human_only"
            else:
                visibility_policy = "agent_visible"

        if layer_isolation is None:
            plane_map = {
                "observed": "verified_plane",
                "derived": "derived_overlay_plane",
                "predicted": "ghost_isolation_plane",
                "diagnostic": "diagnostic_plane",
                "decorative": "decorative_plane",
            }
            layer_isolation = plane_map[render_type]

        # Enforce ghost isolation: predicted renders MUST use ghost_isolation_plane
        if render_type == "predicted" and layer_isolation != "ghost_isolation_plane":
            self.blocked_ghost_contamination_attempts += 1
            self._append_audit_event(
                event_type="ghost_plane_violation",
                target_id=f"pending_{render_type}",
                error_code="ERR_GHOST_BUFFER_CONTAMINATION",
                detail=f"Attempted predicted render on {layer_isolation}.",
            )
            raise ValueError(
                "ERR_GHOST_BUFFER_CONTAMINATION: predicted render must use ghost_isolation_plane."
            )

        # Enforce Observation Contract: a render cannot self-certify as observed at creation
        if claim_observed_without_receipt:
            self.blocked_observation_attempts += 1
            self._append_audit_event(
                event_type="self_certification_blocked",
                target_id=f"pending_{render_type}",
                error_code="ERR_UNVERIFIED_OBSERVATION_CLAIM",
                detail="Attempted create_electric_render with claim_observed_without_receipt=True.",
            )
            raise PermissionError(
                "ERR_UNVERIFIED_OBSERVATION_CLAIM: A render cannot self-certify as observed=True without a valid ObservationReceipt."
            )

        self._render_seq += 1
        render_id = f"rnd_{self._render_seq:04d}_{render_type}"

        ghost_digest: Optional[str] = None
        if render_type == "predicted" and ghost_marks is not None:
            ghost_digest = self._store_ghost_pixels(
                render_id,
                ghost_marks,
                layer_isolation,
            )

        overlay = self.format_uncertainty_overlay(
            uncertainty_state,
            detail=uncertainty_detail or f"CONF:{confidence.upper()}",
            color_mode=color_mode,
        )

        envelope: Dict[str, Any] = {
            "render_id": render_id,
            "render_type": render_type,
            "source": source,
            "observed": False,
            "observation_receipt_id": None,
            "visibility_policy": visibility_policy,
            "layer_isolation": layer_isolation,
            "confidence": confidence,
            "uncertainty_state": uncertainty_state,
            "uncertainty_badge_text": overlay["uncertainty_badge_text"],
            "border_pattern": overlay["border_pattern"],
            "glow_colour_hex": overlay["glow_colour_hex"],
            "provenance_refs": copy.deepcopy(provenance_refs),
            "created_at": FIXED_LOGICAL_TIMESTAMP,
        }
        if ghost_digest is not None:
            envelope["ghost_plane_digest"] = ghost_digest

        assert_no_raw_buffer_leakage(envelope)
        self._renders[render_id] = copy.deepcopy(envelope)
        return copy.deepcopy(envelope)

    def _render_binds_to_captured_source(
        self,
        render: Dict[str, Any],
        captured_source_id: str,
        captured_source_version: str,
    ) -> bool:
        """
        GATE-1 (Fixes GOV-01): Verify that:
        1. render["render_type"] == "observed"
        2. (captured_source_id, captured_source_version) is registered with source_kind == "captured_frame"
        3. render["provenance_refs"] explicitly contains a source ref matching (captured_source_id, captured_source_version)
        """
        if render["render_type"] != "observed":
            return False
        desc = self._sources.get((captured_source_id, captured_source_version))
        if desc is None or desc.source_kind != "captured_frame":
            return False
        return any(
            ref.get("ref_type") == "source"
            and ref.get("ref_id") == captured_source_id
            and ref.get("ref_version") == captured_source_version
            for ref in render.get("provenance_refs", ())
        )

    def issue_observation_receipt(
        self,
        *,
        render_id: str,
        captured_source_id: str,
        captured_source_version: str,
        caller_capability_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Issued exclusively by the verified ObservationWindowGate when a render has
        render_type == 'observed' and provenance_refs bound to a registered
        captured_frame VerifiedSource at (captured_source_id, captured_source_version).
        When a SimulatedPrivilegedGateBoundary is attached (NEXT-GATE-1), requires a
        valid privileged capture service capability token.
        """
        self._assert_supervisor_not_halted()
        if render_id not in self._renders:
            raise KeyError(f"Unknown render_id: {render_id}")
        if (captured_source_id, captured_source_version) not in self._sources:
            raise KeyError(
                f"ERR_UNVERIFIED_SOURCE: {captured_source_id}@{captured_source_version} not registered."
            )
        render = self._renders[render_id]
        if not self._render_binds_to_captured_source(render, captured_source_id, captured_source_version):
            self._append_audit_event(
                event_type="receipt_issuance_rejected",
                target_id=render_id,
                error_code="ERR_UNVERIFIED_OBSERVATION_CLAIM",
                detail=f"Render {render_id} (type={render['render_type']}) does not bind to captured_frame {captured_source_id}@{captured_source_version}.",
            )
            raise PermissionError(
                "ERR_UNVERIFIED_OBSERVATION_CLAIM: Only render_type='observed' with provenance_refs bound to a registered captured_frame source may receive an ObservationReceipt."
            )

        active_boundary = (
            self._out_of_process_vault_client
            if self._out_of_process_vault_client is not None
            else self._simulated_gate_boundary
        )
        if active_boundary is not None:
            try:
                rcpt_id = f"obs_rcpt_{self._receipt_seq + 1:04d}"
                receipt = active_boundary.rpc_issue_observation_receipt(
                    caller_capability_token=caller_capability_token,
                    observation_receipt_id=rcpt_id,
                    render=render,
                    captured_source_desc=self._sources[(captured_source_id, captured_source_version)],
                )
            except PermissionError as exc:
                self._append_audit_event(
                    event_type="unauthorized_or_invalid_rpc_receipt_request",
                    target_id=render_id,
                    error_code="ERR_UNAUTHORIZED_RECEIPT_ISSUER",
                    detail=str(exc),
                )
                raise
            self._receipt_seq += 1
            self._observation_receipts[rcpt_id] = copy.deepcopy(receipt)
            return copy.deepcopy(receipt)

        self._receipt_seq += 1
        rcpt_id = f"obs_rcpt_{self._receipt_seq:04d}"
        stage = "verified_observation_window"
        sig = self._sign_receipt(
            rcpt_id,
            render_id,
            captured_source_id,
            captured_source_version,
            stage,
        )
        receipt = {
            "observation_receipt_id": rcpt_id,
            "render_id": render_id,
            "captured_source_id": captured_source_id,
            "captured_source_version": captured_source_version,
            "pipeline_stage": stage,
            "receipt_signature": sig,
        }
        self._observation_receipts[rcpt_id] = copy.deepcopy(receipt)
        return copy.deepcopy(receipt)

    def promote_render_to_observed(
        self,
        render_id: str,
        receipt: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Attempt to mark a render as observed: True.
        Requires a valid, ledger-verified ObservationReceipt whose captured_source_id
        and captured_source_version match the render's provenance_refs.
        Any missing, forged, or mismatched receipt returns ERR_UNVERIFIED_OBSERVATION_CLAIM,
        keeps the render at observed: False, and increments blocked_observation_attempts.
        """
        self._assert_supervisor_not_halted()
        if render_id not in self._renders:
            raise KeyError(f"Unknown render_id: {render_id}")

        render = self._renders[render_id]
        required_keys = (
            "observation_receipt_id",
            "render_id",
            "captured_source_id",
            "captured_source_version",
            "pipeline_stage",
            "receipt_signature",
        )
        if receipt is None or not isinstance(receipt, dict) or any(k not in receipt for k in required_keys):
            self.blocked_observation_attempts += 1
            self._append_audit_event(
                event_type="unreceipted_promotion_blocked",
                target_id=render_id,
                error_code="ERR_UNVERIFIED_OBSERVATION_CLAIM",
                detail="Missing or incomplete ObservationReceipt.",
            )
            return {
                "status": "rejected",
                "error_code": "ERR_UNVERIFIED_OBSERVATION_CLAIM",
                "reason": "Missing or incomplete ObservationReceipt.",
                "render_id": render_id,
                "observed": render["observed"],
            }

        rcpt_id = receipt["observation_receipt_id"]
        cap_sid = receipt["captured_source_id"]
        cap_sver = receipt["captured_source_version"]
        stored_rcpt = self._observation_receipts.get(rcpt_id)
        active_boundary = (
            self._out_of_process_vault_client
            if self._out_of_process_vault_client is not None
            else self._simulated_gate_boundary
        )
        if active_boundary is not None:
            sig_valid = active_boundary.rpc_verify_observation_receipt(receipt)
        else:
            sig_valid = self._verify_receipt_sig(
                rcpt_id,
                receipt["render_id"],
                cap_sid,
                cap_sver,
                receipt["pipeline_stage"],
                receipt["receipt_signature"],
            )

        if (
            stored_rcpt is None
            or receipt["render_id"] != render_id
            or receipt["pipeline_stage"] != "verified_observation_window"
            or not sig_valid
            or stored_rcpt["receipt_signature"] != receipt["receipt_signature"]
            or not self._render_binds_to_captured_source(render, cap_sid, cap_sver)
        ):
            self.blocked_observation_attempts += 1
            self._append_audit_event(
                event_type="invalid_or_mismatched_receipt_blocked",
                target_id=render_id,
                error_code="ERR_UNVERIFIED_OBSERVATION_CLAIM",
                detail="ObservationReceipt signature, ledger binding, or captured_frame provenance binding failed verification.",
            )
            return {
                "status": "rejected",
                "error_code": "ERR_UNVERIFIED_OBSERVATION_CLAIM",
                "reason": "ObservationReceipt signature or ledger binding failed verification.",
                "render_id": render_id,
                "observed": render["observed"],
            }

        render["observed"] = True
        render["observation_receipt_id"] = rcpt_id
        return {
            "status": "ok",
            "error_code": None,
            "render_id": render_id,
            "observed": True,
            "observation_receipt_id": rcpt_id,
            "render": copy.deepcopy(render),
        }

    def list_agent_visible_renders(
        self,
        include_diagnostic_renders: bool = False,
        evaluate_staleness: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Default agent feed excludes silent_diagnostic and human_only renders, and
        evaluates dynamic staleness by default (NEXT-GATE-4: evaluate_staleness=True).
        Explicit opt-out (evaluate_staleness=False) is preserved for raw creation-time inspection.
        """
        results: List[Dict[str, Any]] = []
        for r_id, r in self._renders.items():
            pol = r["visibility_policy"]
            if pol == "agent_visible" or (pol == "silent_diagnostic" and include_diagnostic_renders):
                item = self.evaluate_render_staleness(r_id) if evaluate_staleness else copy.deepcopy(r)
                results.append(item)
        assert_no_raw_buffer_leakage(results)
        return results

    def get_render(
        self,
        render_id: str,
        include_diagnostic_renders: bool = False,
        evaluate_staleness: bool = True,
    ) -> Dict[str, Any]:
        if render_id not in self._renders:
            raise KeyError(f"Unknown render_id: {render_id}")
        r = self._renders[render_id]
        if r["visibility_policy"] == "silent_diagnostic" and not include_diagnostic_renders:
            raise PermissionError(
                "ERR_SILENT_DIAGNOSTIC_HIDDEN: Diagnostic render is not exposed by default; set include_diagnostic_renders=True."
            )
        if evaluate_staleness:
            return self.evaluate_render_staleness(render_id)
        return copy.deepcopy(r)

    # -------------------------------------------------------------------------
    # Provenance & Telemetry Audit
    # -------------------------------------------------------------------------

    def compute_provenance_completeness(self) -> float:
        total = 0
        valid = 0
        for m in self._measurements:
            total += 1
            if self.verify_measurement_provenance(m)["valid"]:
                valid += 1
        for d in self._derived_representations.values():
            total += 1
            req = ("representation_id", "epistemic_label", "representation_type", "source_refs", "operator_version", "confidence", "rationale", "timestamp")
            if all(k in d and d[k] is not None and d[k] != "" for k in req) and d["epistemic_label"] == "derived" and len(d["source_refs"]) >= 1:
                valid += 1
        for r in self._renders.values():
            total += 1
            req = (
                "render_id",
                "render_type",
                "source",
                "observed",
                "observation_receipt_id",
                "visibility_policy",
                "layer_isolation",
                "confidence",
                "uncertainty_state",
                "uncertainty_badge_text",
                "provenance_refs",
                "created_at",
            )
            has_req = all(k in r for k in req) and bool(r["provenance_refs"])
            receipt_ok = (not r["observed"] and r["observation_receipt_id"] is None) or (
                r["observed"] and r["observation_receipt_id"] in self._observation_receipts
            )
            if has_req and receipt_ok:
                valid += 1
        return round(valid / max(1, total), 6)

    def compute_deterministic_replay_equality(self) -> float:
        if not self._replayed_pairs:
            return 0.0
        equal_count = sum(
            1 for a, b in self._replayed_pairs
            if canonical_json(a["result"]) == canonical_json(b["result"])
            and a["replay_digest"] == b["replay_digest"]
            and canonical_json(a["parameters"]) == canonical_json(b["parameters"])
        )
        return round(equal_count / len(self._replayed_pairs), 6)

    def compute_render_count_by_type(self) -> Dict[str, int]:
        counts = {t: 0 for t in sorted(VALID_RENDER_TYPES)}
        for r in self._renders.values():
            counts[r["render_type"]] += 1
        return counts

    def estimate_memory_footprint_bytes(self) -> int:
        """
        Deterministic calculation of allocated fixture buffer bytes + serialized
        artifact state inside the harness, avoiding tracemalloc per-op hooks
        during latency-critical execution.
        """
        surface_bytes = self._estimate_vault_bytes()
        state_bytes = (
            len(canonical_json(self._measurements))
            + len(canonical_json(self._derived_representations))
            + len(canonical_json(self._renders))
            + len(canonical_json(self._observation_receipts))
        )
        return surface_bytes + state_bytes


def run_all_scenarios() -> Dict[str, Any]:
    """
    Execute scenarios E1 through E10 in an isolated in-memory 128x128 RGBA fixture world,
    evaluate pass criteria C1 through C9, and return the complete structured report.
    Explicitly uses profile='disposable_fixture' and redact_refused_prompts=False
    (NEXT-GATE-2) to preserve byte-identical E1-E10 scenario replay.
    """
    ru_start = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.monotonic()

    audit_sink = InMemoryAppendOnlyAuditSink()
    harness = ElectricEyeFixtureHarness(
        redact_refused_prompts=False,
        profile="disposable_fixture",
        audit_sink=audit_sink,
    )
    scenarios: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # E1: Verified source inspection
    # -------------------------------------------------------------------------
    marks_v1 = [
        DrawingMark("m_square", "rect", 16, 16, 63, 63, (0, 255, 170, 255)),
        DrawingMark("m_marker", "rect", 96, 96, 103, 103, (255, 180, 0, 255)),
    ]
    marks_v2 = [
        DrawingMark("m_square_shifted", "rect", 26, 16, 73, 63, (0, 255, 170, 255)),
        DrawingMark("m_marker", "rect", 96, 96, 103, 103, (255, 180, 0, 255)),
    ]

    src_v1_desc = harness.register_verified_source("src_fix_01", "v1", "verified_pixels", marks_v1)
    src_v2_desc = harness.mutate_canvas("src_fix_01", "v1", "v2", marks_v2)

    # Register the remaining 4 source kinds to cover all 5 required source_kinds
    harness.register_verified_source("src_draw_01", "v1", "drawing_state", marks_v1)
    harness.register_verified_source("src_canvas_01", "v1", "canvas_state", marks_v1)
    harness.register_verified_source("src_img_01", "v1", "image_bytes", marks_v1)
    harness.register_verified_source("src_cap_01", "v1", "captured_frame", marks_v2)

    e1_inspect_v1 = harness.execute_measurement(
        "inspect_source", {}, source_id="src_fix_01", source_version="v1"
    )
    e1_inspect_v2 = harness.execute_measurement(
        "inspect_source", {}, source_id="src_fix_01", source_version="v2"
    )
    e1_missing = harness.execute_measurement(
        "inspect_source", {}, source_id="src_nonexistent_99", source_version="v1"
    )

    e1_pass = (
        e1_inspect_v1["status"] == "ok"
        and e1_inspect_v1["result"]["source_id"] == "src_fix_01"
        and e1_inspect_v1["result"]["source_version"] == "v1"
        and e1_inspect_v1["result"]["content_sha256"] == src_v1_desc["content_sha256"]
        and e1_inspect_v1["result"]["observed"] is False
        and e1_inspect_v1["result"]["observation_claim"] is False
        and e1_inspect_v2["result"]["content_sha256"] == src_v2_desc["content_sha256"]
        and e1_inspect_v1["result"]["content_sha256"] != e1_inspect_v2["result"]["content_sha256"]
        and e1_missing["status"] == "error"
        and e1_missing["refusal_code"] == "ERR_UNVERIFIED_SOURCE"
    )
    scenarios["E1_verified_source_inspection"] = {
        "pass": e1_pass,
        "inspected_v1": e1_inspect_v1,
        "inspected_v2": e1_inspect_v2,
        "unverified_source_rejection": e1_missing,
    }

    # -------------------------------------------------------------------------
    # E2: Deterministic measurement replay (all 10 closed-form operators)
    # -------------------------------------------------------------------------
    operator_calls = [
        ("inspect_source", {}, "src_fix_01", "v1", None),
        ("measure_dimensions", {}, "src_fix_01", "v1", None),
        (
            "measure_bbox",
            {"selector": {"type": "colour_match", "rgba": [0, 255, 170, 255], "tolerance": 0}},
            "src_fix_01",
            "v1",
            None,
        ),
        ("sample_colour", {"x": 32, "y": 32}, "src_fix_01", "v1", None),
        (
            "measure_distance",
            {"p1": {"x": 16, "y": 16}, "p2": {"x": 63, "y": 63}, "metric": "euclidean"},
            "src_fix_01",
            "v1",
            None,
        ),
        (
            "measure_angle",
            {"p1": {"x": 16, "y": 16}, "p2": {"x": 63, "y": 63}},
            "src_fix_01",
            "v1",
            None,
        ),
        (
            "count_pixels",
            {"predicate": {"type": "non_transparent"}},
            "src_fix_01",
            "v1",
            None,
        ),
        (
            "count_changed_pixels",
            {"channel_tolerance": 0},
            "src_fix_01",
            "v1",
            {"source_id": "src_fix_01", "source_version": "v2"},
        ),
        (
            "extract_changed_regions",
            {"connectivity": 8, "channel_tolerance": 0},
            "src_fix_01",
            "v1",
            {"source_id": "src_fix_01", "source_version": "v2"},
        ),
        (
            "count_objects",
            {"connectivity": 8, "min_area_px": 1},
            "src_fix_01",
            "v1",
            None,
        ),
    ]

    e2_replays = []
    e2_all_equal = True
    for op_name, params, sid, sver, sec in operator_calls:
        run_a = harness.execute_measurement(op_name, params, source_id=sid, source_version=sver, secondary_source=sec)
        run_b = harness.execute_measurement(op_name, params, source_id=sid, source_version=sver, secondary_source=sec)
        is_eq = harness.record_replay_pair(run_a, run_b)
        e2_all_equal = e2_all_equal and is_eq
        e2_replays.append({
            "operator_name": op_name,
            "run_a_measurement_id": run_a["measurement_id"],
            "run_b_measurement_id": run_b["measurement_id"],
            "replay_digest": run_a["replay_digest"],
            "byte_identical": is_eq,
            "result": run_a["result"],
        })

    scenarios["E2_deterministic_measurement_replay"] = {
        "pass": e2_all_equal and len(e2_replays) == 10,
        "operator_replay_count": len(e2_replays),
        "replays": e2_replays,
    }

    # -------------------------------------------------------------------------
    # E3: Measurement refusal
    # -------------------------------------------------------------------------
    refusal_queries = [
        ("semantic_meaning", {"query_type": "semantic_meaning", "question": "What does this image mean?"}),
        ("aesthetic_judgment", {"query_type": "aesthetic_judgment", "question": "Is this composition good?"}),
        ("agent_intent", {"query_type": "agent_intent", "question": "What is the agent thinking?"}),
        ("free_text_interpretation", {"prompt": "Interpret the emotional narrative of the cyan square."}),
    ]
    e3_envelopes = []
    for op_name, params in refusal_queries:
        env = harness.execute_measurement(op_name, params, source_id="src_fix_01", source_version="v1")
        e3_envelopes.append(env)

    e3_pass = all(
        env["status"] == "refused"
        and env["result"] == "unknown"
        and env["refusal_code"] == REFUSAL_CODE_SEMANTIC
        and env["refusal_reason"] == REFUSAL_REASON_SEMANTIC
        for env in e3_envelopes
    )
    scenarios["E3_measurement_refusal"] = {
        "pass": e3_pass,
        "refusal_envelopes": e3_envelopes,
    }

    # -------------------------------------------------------------------------
    # E4: Derived representation (all 5 required types)
    # -------------------------------------------------------------------------
    rep_change_mask = harness.compute_derived_representation(
        "change_mask", ("src_fix_01", "v1"), ("src_fix_01", "v2")
    )
    rep_diff_map = harness.compute_derived_representation(
        "difference_map", ("src_fix_01", "v1"), ("src_fix_01", "v2")
    )
    rep_motion = harness.compute_derived_representation(
        "motion_summary", ("src_fix_01", "v1"), ("src_fix_01", "v2")
    )
    rep_colour = harness.compute_derived_representation(
        "colour_delta", ("src_fix_01", "v1"), ("src_fix_01", "v2")
    )
    rep_spatial = harness.compute_derived_representation(
        "spatial_relationship", ("src_fix_01", "v1"), ("src_fix_01", "v2")
    )
    e4_reps = [rep_change_mask, rep_diff_map, rep_motion, rep_colour, rep_spatial]
    e4_pass = (
        all(r["epistemic_label"] == "derived" for r in e4_reps)
        and all(len(r["source_refs"]) == 2 for r in e4_reps)
        and all(r["confidence"] in VALID_CONFIDENCE_LEVELS and bool(r["rationale"]) for r in e4_reps)
        and rep_change_mask["summary"]["changed_pixels"] == 960
    )
    scenarios["E4_derived_representation"] = {
        "pass": e4_pass,
        "representations": e4_reps,
    }

    # -------------------------------------------------------------------------
    # E5: Electric render labelling
    # -------------------------------------------------------------------------
    rnd_derived = harness.create_electric_render(
        render_type="derived",
        source="measurement",
        confidence="high",
        uncertainty_state="derived",
        provenance_refs=[
            {"ref_type": "derived_representation", "ref_id": rep_change_mask["representation_id"], "ref_version": "1.0.0"},
            {"ref_type": "source", "ref_id": "src_fix_01", "ref_version": "v1"},
            {"ref_type": "source", "ref_id": "src_fix_01", "ref_version": "v2"},
        ],
    )
    rnd_decorative = harness.create_electric_render(
        render_type="decorative",
        source="generation",
        confidence="low",
        uncertainty_state="unknown",
        provenance_refs=[
            {"ref_type": "source", "ref_id": "src_canvas_01", "ref_version": "v1"},
        ],
    )
    e5_pass = (
        rnd_derived["render_type"] == "derived"
        and rnd_derived["source"] == "measurement"
        and rnd_derived["observed"] is False
        and rnd_derived["observation_receipt_id"] is None
        and len(rnd_derived["provenance_refs"]) == 3
    )
    scenarios["E5_electric_render_labelling"] = {
        "pass": e5_pass,
        "derived_render": rnd_derived,
        "decorative_render": rnd_decorative,
    }

    # -------------------------------------------------------------------------
    # E6: Ghost render isolation
    # -------------------------------------------------------------------------
    ghost_marks_v3 = [
        DrawingMark("m_ghost_square", "rect", 36, 16, 83, 63, (155, 93, 229, 180)),
    ]
    rnd_ghost = harness.create_electric_render(
        render_type="predicted",
        source="inference",
        confidence="medium",
        uncertainty_state="predicted",
        provenance_refs=[
            {"ref_type": "derived_representation", "ref_id": rep_motion["representation_id"], "ref_version": "1.0.0"},
            {"ref_type": "source", "ref_id": "src_fix_01", "ref_version": "v2"},
        ],
        ghost_marks=ghost_marks_v3,
    )
    sample_at_ghost_coord = harness.execute_measurement(
        "sample_colour", {"x": 80, "y": 32}, source_id="src_fix_01", source_version="v2"
    )
    inspect_v2_after_ghost = harness.execute_measurement(
        "inspect_source", {}, source_id="src_fix_01", source_version="v2"
    )

    ghost_plane_violation_caught = False
    try:
        harness.create_electric_render(
            render_type="predicted",
            source="inference",
            confidence="low",
            uncertainty_state="predicted",
            layer_isolation="verified_plane",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_fix_01", "ref_version": "v2"}],
        )
    except ValueError as exc:
        if "ERR_GHOST_BUFFER_CONTAMINATION" in str(exc):
            ghost_plane_violation_caught = True

    e6_pass = (
        rnd_ghost["render_type"] == "predicted"
        and rnd_ghost["layer_isolation"] == "ghost_isolation_plane"
        and rnd_ghost["observed"] is False
        and sample_at_ghost_coord["result"]["rgba"] == [0, 0, 0, 0]
        and inspect_v2_after_ghost["result"]["content_sha256"] == src_v2_desc["content_sha256"]
        and ghost_plane_violation_caught
    )
    scenarios["E6_ghost_render_isolation"] = {
        "pass": e6_pass,
        "ghost_render": rnd_ghost,
        "verified_sample_at_ghost_coord": sample_at_ghost_coord,
        "ghost_plane_violation_blocked": ghost_plane_violation_caught,
    }

    # -------------------------------------------------------------------------
    # E7: Uncertainty glow and multi-modal status
    # -------------------------------------------------------------------------
    states_ordered = ["verified", "derived", "predicted", "stale", "unknown"]
    overlays_color = [
        harness.format_uncertainty_overlay(st, detail=f"MODE:COLOR:{st.upper()}", color_mode=True)
        for st in states_ordered
    ]
    overlays_no_color = [
        harness.format_uncertainty_overlay(st, detail=f"MODE:PLAIN:{st.upper()}", color_mode=False)
        for st in states_ordered
    ]
    unique_badges_color = {o["uncertainty_badge_text"] for o in overlays_color}
    unique_badges_plain = {o["uncertainty_badge_text"] for o in overlays_no_color}
    unique_borders = {o["border_pattern"] for o in overlays_no_color}
    plain_has_no_colour = all(o["glow_colour_hex"] is None for o in overlays_no_color)

    e7_pass = (
        len(unique_badges_color) == 5
        and len(unique_badges_plain) == 5
        and len(unique_borders) == 5
        and plain_has_no_colour
    )
    scenarios["E7_uncertainty_glow"] = {
        "pass": e7_pass,
        "colour_overlays": overlays_color,
        "no_colour_overlays": overlays_no_color,
    }

    # -------------------------------------------------------------------------
    # E8: Silent diagnostic render
    # -------------------------------------------------------------------------
    rnd_diag = harness.create_electric_render(
        render_type="diagnostic",
        source="measurement",
        confidence="high",
        uncertainty_state="derived",
        provenance_refs=[
            {"ref_type": "source", "ref_id": "src_fix_01", "ref_version": "v2"},
        ],
    )
    default_feed = harness.list_agent_visible_renders(include_diagnostic_renders=False)
    opt_in_feed = harness.list_agent_visible_renders(include_diagnostic_renders=True)
    explicit_diag = harness.get_render(rnd_diag["render_id"], include_diagnostic_renders=True)

    diag_in_default = any(r["render_type"] == "diagnostic" for r in default_feed)
    diag_in_opt_in = any(r["render_id"] == rnd_diag["render_id"] for r in opt_in_feed)

    e8_pass = (
        rnd_diag["visibility_policy"] == "silent_diagnostic"
        and rnd_diag["observed"] is False
        and not diag_in_default
        and diag_in_opt_in
        and explicit_diag["observed"] is False
    )
    scenarios["E8_silent_diagnostic_render"] = {
        "pass": e8_pass,
        "diagnostic_render": rnd_diag,
        "default_agent_feed_count": len(default_feed),
        "diagnostic_in_default_feed": diag_in_default,
        "diagnostic_retrievable_on_opt_in": diag_in_opt_in,
    }

    # -------------------------------------------------------------------------
    # E9: Observation window
    # -------------------------------------------------------------------------
    attempt_missing_rcpt = harness.promote_render_to_observed(rnd_derived["render_id"], receipt=None)

    forged_receipt = {
        "observation_receipt_id": "obs_rcpt_9999",
        "render_id": rnd_derived["render_id"],
        "captured_source_id": "src_cap_01",
        "captured_source_version": "v1",
        "pipeline_stage": "verified_observation_window",
        "receipt_signature": "hmac-sha256:0000000000000000000000000000000000000000000000000000000000000000",
    }
    attempt_forged_rcpt = harness.promote_render_to_observed(rnd_derived["render_id"], receipt=forged_receipt)

    self_cert_blocked = False
    try:
        harness.create_electric_render(
            render_type="observed",
            source="verified_pixels",
            confidence="high",
            uncertainty_state="verified",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_cap_01", "ref_version": "v1"}],
            claim_observed_without_receipt=True,
        )
    except PermissionError as exc:
        if "ERR_UNVERIFIED_OBSERVATION_CLAIM" in str(exc):
            self_cert_blocked = True

    rnd_observed_candidate = harness.create_electric_render(
        render_type="observed",
        source="verified_pixels",
        confidence="high",
        uncertainty_state="verified",
        provenance_refs=[{"ref_type": "source", "ref_id": "src_cap_01", "ref_version": "v1"}],
    )
    valid_receipt = harness.issue_observation_receipt(
        render_id=rnd_observed_candidate["render_id"],
        captured_source_id="src_cap_01",
        captured_source_version="v1",
    )
    legit_promotion = harness.promote_render_to_observed(
        rnd_observed_candidate["render_id"], receipt=valid_receipt
    )

    rnd_derived_after = harness.get_render(rnd_derived["render_id"])
    e9_pass = (
        attempt_missing_rcpt["status"] == "rejected"
        and attempt_missing_rcpt["error_code"] == "ERR_UNVERIFIED_OBSERVATION_CLAIM"
        and attempt_forged_rcpt["status"] == "rejected"
        and attempt_forged_rcpt["error_code"] == "ERR_UNVERIFIED_OBSERVATION_CLAIM"
        and self_cert_blocked
        and rnd_derived_after["observed"] is False
        and legit_promotion["status"] == "ok"
        and legit_promotion["observed"] is True
        and harness.false_observation_count == 0
        and harness.blocked_observation_attempts == 3
        and harness.verify_security_audit_ledger()
    )
    scenarios["E9_observation_window"] = {
        "pass": e9_pass,
        "unreceipted_attempt": attempt_missing_rcpt,
        "forged_receipt_attempt": attempt_forged_rcpt,
        "self_certification_blocked": self_cert_blocked,
        "valid_receipt": valid_receipt,
        "legitimate_observed_promotion": legit_promotion,
        "blocked_observation_attempts": harness.blocked_observation_attempts,
        "false_observation_count": harness.false_observation_count,
    }

    # -------------------------------------------------------------------------
    # E10: Measurement provenance and tamper detection
    # -------------------------------------------------------------------------
    e10_meas = harness.execute_measurement(
        "measure_distance",
        {"p1": {"x": 26, "y": 16}, "p2": {"x": 73, "y": 63}, "metric": "euclidean"},
        source_id="src_fix_01",
        source_version="v2",
    )
    e10_valid_check = harness.verify_measurement_provenance(e10_meas)

    tampered_missing_op_ver = copy.deepcopy(e10_meas)
    tampered_missing_op_ver["operator_version"] = ""
    e10_tamper_missing_check = harness.verify_measurement_provenance(tampered_missing_op_ver)

    tampered_altered_result = copy.deepcopy(e10_meas)
    tampered_altered_result["result"]["distance"] = 9999.0
    e10_tamper_digest_check = harness.verify_measurement_provenance(tampered_altered_result)

    e10_pass = (
        e10_valid_check["valid"] is True
        and e10_tamper_missing_check["valid"] is False
        and e10_tamper_missing_check["error_code"] == "ERR_INCOMPLETE_PROVENANCE"
        and e10_tamper_digest_check["valid"] is False
        and e10_tamper_digest_check["error_code"] == "ERR_PROVENANCE_DIGEST_MISMATCH"
    )
    scenarios["E10_measurement_provenance"] = {
        "pass": e10_pass,
        "measurement": e10_meas,
        "valid_provenance_check": e10_valid_check,
        "tamper_missing_field_check": e10_tamper_missing_check,
        "tamper_altered_result_check": e10_tamper_digest_check,
    }

    # -------------------------------------------------------------------------
    # Pass Criteria C1 - C9
    # -------------------------------------------------------------------------
    replay_eq = harness.compute_deterministic_replay_equality()
    prov_comp = harness.compute_provenance_completeness()
    render_counts = harness.compute_render_count_by_type()

    pass_criteria = {
        "C1_verified_sources_identified_and_versioned": e1_pass,
        "C2_measurements_deterministic": e2_all_equal and replay_eq == 1.0,
        "C3_semantic_interpretation_refused": (
            e3_pass and harness.semantic_queries_total > 0
            and harness.semantic_queries_refused == harness.semantic_queries_total
        ),
        "C4_electric_renders_clearly_labelled": e5_pass and prov_comp == 1.0,
        "C5_ghost_renders_isolated": e6_pass,
        "C6_uncertainty_visible_and_textual": e7_pass,
        "C7_silent_renders_not_exposed_by_default": e8_pass,
        "C8_non_observed_renders_cannot_become_observations": (
            e9_pass and harness.false_observation_count == 0
        ),
        "C9_measurement_provenance_complete": e10_pass and prov_comp == 1.0,
    }

    # Stop timing & compute memory footprint
    t1 = time.monotonic()
    ru_end = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem = harness.estimate_memory_footprint_bytes()

    cpu_time_ms = round(
        ((ru_end.ru_utime - ru_start.ru_utime) + (ru_end.ru_stime - ru_start.ru_stime)) * 1000.0,
        4,
    )
    wall_latency_ms = round((t1 - t0) * 1000.0, 4)

    audit_summary = harness.export_audit_chain_summary()

    report: Dict[str, Any] = {
        "harness_version": OPERATOR_VERSION,
        "fixture_world": {
            "width": CANVAS_WIDTH,
            "height": CANVAS_HEIGHT,
            "color_model": "RGBA8888",
            "isolation": "in_memory_only",
        },
        "all_scenarios_passed": all(s["pass"] for s in scenarios.values()),
        "all_criteria_passed": all(pass_criteria.values()),
        "pass_criteria": pass_criteria,
        "audit_ledger_summary": audit_summary,
        "metrics": {
            "source_count": len(harness._sources),
            "measurement_count": len(harness._measurements),
            "deterministic_replay_equality": replay_eq,
            "derived_representation_count": len(harness._derived_representations),
            "render_count_by_type": render_counts,
            "provenance_completeness": prov_comp,
            "false_observation_count": harness.false_observation_count,
            "blocked_observation_attempts": harness.blocked_observation_attempts,
            "audit_entry_count": audit_summary["entry_count"],
            "audit_chain_head_digest": audit_summary["chain_head_digest"],
            "cpu_time": cpu_time_ms,
            "wall_latency": wall_latency_ms,
            "memory_footprint": peak_mem,
            "output_size": 0,  # populated below from serialized JSON byte length
        },
        "scenarios": scenarios,
        "non_claims": [
            "Does not establish production readiness or live Cyclops Eye integration.",
            "Does not establish real-world visual understanding or scene comprehension.",
            "Does not establish neural model accuracy or statistical confidence calibration.",
            "Does not establish natural-language generation quality.",
            "Does not establish subjective visual perception or agent consciousness.",
            "Does not claim any render is an observation unless backed by a verified ObservationReceipt and complete provenance.",
        ],
    }

    serialized_prelim = canonical_json(report).encode("utf-8")
    report["metrics"]["output_size"] = len(serialized_prelim)
    assert_no_raw_buffer_leakage(report)
    return report


if __name__ == "__main__":
    report_data = run_all_scenarios()
    print(json.dumps(report_data, indent=2, sort_keys=True))
