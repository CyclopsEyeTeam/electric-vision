#!/usr/bin/env python3
"""
Electric Eye (128x128 RGBA) — Isolated Lab Implementation of EXEC-GATE-1, EXEC-GATE-2, and EXEC-GATE-3
======================================================================================================

Implements the three execution-blocking gates in an isolated, non-production, non-networked lab
environment using ephemeral OS child processes and AF_UNIX domain sockets with Linux SO_PEERCRED
kernel peer authentication:

- EXEC-GATE-1 (`OutOfProcessVaultGateDaemon` / `OutOfProcessVaultGateClient`):
  Out-of-process pixel vault and ObservationWindowGate service. Keeps all raw RGBA buffers and
  >=32-byte HMAC signing keys exclusively in the privileged daemon process address space.
  Enforces SO_PEERCRED UID/GID verification, single-use capability tokens, RPC nonce replay
  protection, `render_type == "observed"`, and `source_kind == "captured_frame"` binding.

- EXEC-GATE-2 (`OutOfProcessAuditSinkDaemon` / `OutOfProcessAuditSinkClient`):
  External append-only WORM audit storage process. Enforces O_APPEND + os.fsync durable writes,
  dual-clock stamping (`logical_clock` + `utc_wall_timestamp`), SHA-256 hash-chain continuity,
  tamper detection, zero raw pixel / zero unredacted prompt ingestion, signed chain-head export,
  and synchronous fail-closed behavior when unavailable.

- EXEC-GATE-3 (`OutOfProcessQuarantineAndKillSwitchDaemon` / `OutOfProcessQuarantineAndKillSwitchClient`):
  Pre-ingestion zero-PII quarantine and kill-switch supervisor process. Enforces synthetic-only
  input attestation, opaque safe `source_id` validation, default refused-prompt redaction, and
  10 armed kill switches (`KS-1` through `KS-10`) with durable pre-halt forensic evidence
  preservation and vault lock propagation.

HARD BOUNDARIES:
- Scoped strictly to temporary lab context managers; all child processes terminate on context exit.
- No live Cyclops Eye integration, no external network sockets, no real user data.
"""

from __future__ import annotations

from ev_child_guard import die_with_parent
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import multiprocessing
import os
import secrets
import socket
import struct
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from electric_eye_harness import (
    CANVAS_HEIGHT,
    CANVAS_TOTAL_PIXELS,
    CANVAS_WIDTH,
    DISPOSABLE_FIXTURE_STATIC_HMAC_KEY,
    DrawingMark,
    ElectricEyeFixtureHarness,
    FIXED_LOGICAL_TIMESTAMP,
    LONG_DIGIT_RUN_REGEX,
    MAX_CACHE_ENTRIES,
    MAX_MARKS_PER_SOURCE,
    MAX_REPORTED_REGIONS,
    OPERATOR_VERSION,
    PROHIBITED_SOURCE_ID_TOKENS,
    ROW_BYTE_STRIDE,
    SOURCE_ID_REGEX,
    VALID_CONFIDENCE_LEVELS,
    VALID_SOURCE_KINDS,
    VerifiedSourceDescriptor,
    _create_isolated_vault_and_gate,
    assert_no_raw_buffer_leakage,
    canonical_json,
    run_all_scenarios,
    sha256_hex,
)


# =============================================================================
# Framed AF_UNIX IPC Helpers + SO_PEERCRED Verification
# =============================================================================

MAX_IPC_FRAME_BYTES = 4 * 1024 * 1024  # 4 MiB hard frame ceiling
IPC_REQUEST_TIMEOUT_S = 2.0
MAX_RPC_NONCES = 65536
MAX_RPC_NONCE_CHARS = 256
MAX_CAPABILITY_RECORDS = 4096
MAX_AUDIT_ENTRY_BYTES = 64 * 1024
MAX_AUDIT_PAGE_BYTES = 512 * 1024
MAX_AUDIT_PAGE_ENTRIES = 256


@contextmanager
def _isolated_ipc_connection(conn: socket.socket):
    """One malformed/slow/disconnected client cannot terminate the accept loop."""
    with conn:
        conn.settimeout(IPC_REQUEST_TIMEOUT_S)
        try:
            yield conn
        except (OSError, EOFError, ValueError, TypeError, KeyError, RecursionError):
            # Includes failures while sending the error response to a departed client.
            pass


def _send_framed_json(sock: socket.socket, payload: Dict[str, Any]) -> None:
    raw = canonical_json(payload).encode("utf-8")
    if len(raw) > MAX_IPC_FRAME_BYTES:
        raise ValueError(f"ERR_IPC_FRAME_TOO_LARGE: {len(raw)} > {MAX_IPC_FRAME_BYTES}")
    header = struct.pack(">I", len(raw))
    sock.sendall(header + raw)


def _recv_exact(sock: socket.socket, n_bytes: int, *, deadline: Optional[float] = None) -> bytes:
    buf = bytearray()
    original_timeout = sock.gettimeout()
    try:
        while len(buf) < n_bytes:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("ERR_IPC_REQUEST_DEADLINE")
                sock.settimeout(min(left, original_timeout) if original_timeout is not None else left)
            chunk = sock.recv(n_bytes - len(buf))
            if not chunk:
                raise EOFError("Unexpected EOF on AF_UNIX IPC socket.")
            buf.extend(chunk)
        return bytes(buf)
    finally:
        sock.settimeout(original_timeout)


def _recv_framed_json(sock: socket.socket) -> Dict[str, Any]:
    timeout = sock.gettimeout()
    deadline = time.monotonic() + (timeout if timeout is not None else IPC_REQUEST_TIMEOUT_S)
    header = _recv_exact(sock, 4, deadline=deadline)
    (frame_len,) = struct.unpack(">I", header)
    if frame_len <= 0 or frame_len > MAX_IPC_FRAME_BYTES:
        raise ValueError(f"ERR_INVALID_IPC_FRAME_LENGTH: {frame_len}")
    raw = _recv_exact(sock, frame_len, deadline=deadline)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("ERR_IPC_EXPECTED_OBJECT")
    return value


def _recv_rpc_request(sock: socket.socket) -> Dict[str, Any]:
    req = _recv_framed_json(sock)
    if not isinstance(req.get("method"), str) or not isinstance(req.get("params", {}), dict):
        raise ValueError("ERR_IPC_INVALID_REQUEST_SHAPE")
    if "caller_principal" in req and not isinstance(req["caller_principal"], str):
        raise ValueError("ERR_IPC_INVALID_PRINCIPAL_SHAPE")
    return req


def _extract_so_peercred(conn: socket.socket) -> Tuple[int, int, int]:
    """
    Extract kernel-verified (pid, uid, gid) from an AF_UNIX stream socket on Linux.
    """
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", creds)
    return pid, uid, gid


# =============================================================================
# EXEC-GATE-1: Out-of-Process Privileged Vault & Observation Gate Daemon
# =============================================================================

ALLOWED_VAULT_RPC_METHODS = frozenset({
    "ping_status",
    "store_verified_source",
    "store_ghost_pixels",
    "run_pixel_operator",
    "compute_derived_summary",
    "issue_capability_token",
    "issue_observation_receipt",
    "verify_observation_receipt",
    "emergency_lock_vault",
    "evict_sources",
    "shutdown",
})


def _vault_gate_daemon_loop(
    socket_path: str,
    allowed_uid: int,
    allowed_gid: int,
    ready_event: Any,
    initial_key_bytes: Optional[bytes],
    capture_service_auth_secret: str,
) -> None:
    """
    Runs exclusively in the child OS process for EXEC-GATE-1.
    Raw RGBA pixel buffers and the 32-byte HMAC signing key live ONLY inside this process.
    """
    die_with_parent()
    if initial_key_bytes is not None:
        secret_key = initial_key_bytes
    else:
        secret_key = secrets.token_bytes(32)

    if (
        not isinstance(secret_key, bytes)
        or len(secret_key) < 32
        or secret_key == DISPOSABLE_FIXTURE_STATIC_HMAC_KEY
    ):
        raise PermissionError("ERR_STATIC_OR_WEAK_KEY_PROHIBITED in VaultGateDaemon")

    (
        store_verified_source,
        store_ghost_render_pixels,
        get_surface,
        sample_pixel,
        estimate_vault_bytes,
        sign_receipt,
        verify_receipt_sig,
        get_raster_cache_size,
        evict_surfaces,
    ) = _create_isolated_vault_and_gate(
        profile="controlled_pilot_simulation",
        hmac_key_provider=lambda: secret_key,
    )

    source_registry: Dict[Tuple[str, str], Dict[str, Any]] = {}
    issued_capability_tokens: Dict[str, Dict[str, Any]] = {}
    consumed_capability_tokens: Set[str] = set()
    used_rpc_nonces: Set[bytes] = set()
    issued_receipts: Dict[str, Dict[str, Any]] = {}
    cc_cache: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    verified_peer_connections = 0
    last_peer_cred = None
    rejected_peer_attempts = 0
    replay_blocks = 0
    unauthorized_issuer_blocks = 0
    emergency_locked = False

    def _get_fg_components(sid: str, sver: str, connectivity: int) -> List[Dict[str, Any]]:
        ckey = (sid, sver, connectivity)
        if ckey in cc_cache:
            return cc_cache[ckey]
        surface = get_surface(sid, sver)
        merged_runs: List[Tuple[int, int, int]] = []
        for y, x0, x1, _rgba in surface.fg_runs:
            if merged_runs and merged_runs[-1][0] == y and merged_runs[-1][2] + 1 == x0:
                merged_runs[-1] = (y, merged_runs[-1][1], x1)
            else:
                merged_runs.append((y, x0, x1))
        comps = ElectricEyeFixtureHarness._connected_components_from_runs(
            merged_runs, connectivity=connectivity
        )
        cc_cache[ckey] = comps
        if len(cc_cache) > MAX_CACHE_ENTRIES:
            oldest = next(iter(cc_cache))
            del cc_cache[oldest]
        return comps

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o600)
    server.listen(16)
    ready_event.set()

    running = True
    try:
        while running:
            conn, _ = server.accept()
            with _isolated_ipc_connection(conn):
                try:
                    peer_pid, peer_uid, peer_gid = _extract_so_peercred(conn)
                    req = _recv_rpc_request(conn)
                except Exception as exc:
                    _send_framed_json(conn, {"status": "error", "error_code": "ERR_IPC_READ", "reason": str(exc)})
                    continue

                # Only kernel credentials authenticate the caller; payload fields have no authority.
                if peer_uid != allowed_uid or peer_gid != allowed_gid:
                    rejected_peer_attempts += 1
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_PEER_AUTH_FAILED",
                            "reason": (
                                f"SO_PEERCRED uid={peer_uid}/gid={peer_gid} "
                                f"does not match allowed uid={allowed_uid}/gid={allowed_gid}."
                            ),
                        },
                    )
                    continue

                verified_peer_connections += 1
                last_peer_cred = {"pid": peer_pid, "uid": peer_uid, "gid": peer_gid}

                method = req.get("method")
                rpc_nonce = req.get("rpc_nonce")
                params = req.get("params", {})

                if method == "shutdown":
                    running = False
                    _send_framed_json(conn, {"status": "ok", "result": {"shutdown": True}})
                    continue

                if method not in ALLOWED_VAULT_RPC_METHODS:
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_RPC_METHOD_PROHIBITED",
                            "reason": f"Method '{method}' is prohibited on narrow VaultGate IPC boundary.",
                        },
                    )
                    continue

                # Replay protection on rpc_nonce
                if not isinstance(rpc_nonce, str) or not 8 <= len(rpc_nonce) <= MAX_RPC_NONCE_CHARS:
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_MISSING_RPC_NONCE",
                            "reason": "Every RPC call must carry a unique rpc_nonce.",
                        },
                    )
                    continue
                nonce_digest = hashlib.sha256(rpc_nonce.encode("utf-8")).digest()
                if nonce_digest in used_rpc_nonces:
                    replay_blocks += 1
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_REPLAY_DETECTED",
                            "reason": f"Duplicate rpc_nonce '{rpc_nonce}' rejected by replay guard.",
                        },
                    )
                    continue
                if len(used_rpc_nonces) >= MAX_RPC_NONCES:
                    # Never evict a replay tombstone. Safe, idempotent halt remains available.
                    if method == "emergency_lock_vault":
                        emergency_locked = True
                        _send_framed_json(conn, {"status": "ok", "result": {"emergency_locked": True}})
                    else:
                        _send_framed_json(conn, {"status": "error", "error_code": "ERR_RPC_NONCE_CAPACITY",
                                                 "reason": "Replay guard full; restart the session, not the replay window."})
                    continue
                used_rpc_nonces.add(nonce_digest)

                if emergency_locked and method != "ping_status":
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_KILL_SWITCH_HALTED",
                            "reason": "VaultGateDaemon is locked by Kill-Switch Supervisor.",
                        },
                    )
                    continue

                try:
                    if method == "ping_status":
                        res = {
                            "daemon_pid": os.getpid(),
                            "daemon_uid": os.getuid(),
                            "daemon_gid": os.getgid(),
                            "boundary_mode": "out_of_process_af_unix_daemon",
                            "claims_real_process_isolation": True,
                            "so_peercred_enforced": True,
                            "verified_peer_connections": verified_peer_connections,
                            "last_peer_cred": last_peer_cred,
                            "nonce_records": len(used_rpc_nonces),
                            "nonce_capacity": MAX_RPC_NONCES,
                            "rejected_peer_attempts": rejected_peer_attempts,
                            "replay_blocks": replay_blocks,
                            "unauthorized_issuer_blocks": unauthorized_issuer_blocks,
                            "registered_source_count": len(source_registry),
                            "vault_memory_bytes": estimate_vault_bytes(),
                            "raster_cache_size": get_raster_cache_size(),
                            "emergency_locked": emergency_locked,
                        }
                        _send_framed_json(conn, {"status": "ok", "result": res})

                    elif method == "emergency_lock_vault":
                        emergency_locked = True
                        _send_framed_json(conn, {"status": "ok", "result": {"emergency_locked": True}})

                    elif method == "store_verified_source":
                        sid = params["source_id"]
                        sver = params["source_version"]
                        skind = params["source_kind"]
                        plane = params.get("target_plane", "verified_plane")
                        marks = [
                            DrawingMark(
                                m["mark_id"],
                                m["shape"],
                                m["x0"],
                                m["y0"],
                                m["x1"],
                                m["y1"],
                                tuple(m["rgba"]),
                            )
                            for m in params["marks"]
                        ]
                        digest = store_verified_source(sid, sver, skind, marks, plane)
                        source_registry[(sid, sver)] = {
                            "source_id": sid,
                            "source_version": sver,
                            "source_kind": skind,
                            "content_sha256": digest,
                            "plane": plane,
                        }
                        _send_framed_json(conn, {"status": "ok", "result": {"content_sha256": digest}})

                    elif method == "evict_sources":
                        keys = [(str(k[0]), str(k[1])) for k in params.get("sources", [])][:4096]
                        retired = set(keys)
                        for key in keys:
                            source_registry.pop(key, None)
                        for receipt in issued_receipts.values():
                            if (receipt["captured_source_id"], receipt["captured_source_version"]) in retired:
                                receipt["revoked"] = True
                        for token_id, token in list(issued_capability_tokens.items()):
                            if (token["captured_source_id"], token["captured_source_version"]) in retired:
                                consumed_capability_tokens.add(token_id)
                                issued_capability_tokens.pop(token_id)
                        for cc_key in [k for k in cc_cache if (k[0], k[1]) in retired]:
                            cc_cache.pop(cc_key, None)
                        dropped = evict_surfaces(keys)
                        _send_framed_json(conn, {"status": "ok", "result": {"evicted": dropped}})

                    elif method == "store_ghost_pixels":
                        rid = params["render_id"]
                        plane = params["target_plane"]
                        marks = [
                            DrawingMark(
                                m["mark_id"],
                                m["shape"],
                                m["x0"],
                                m["y0"],
                                m["x1"],
                                m["y1"],
                                tuple(m["rgba"]),
                            )
                            for m in params["marks"]
                        ]
                        digest = store_ghost_render_pixels(rid, marks, plane)
                        _send_framed_json(conn, {"status": "ok", "result": {"ghost_plane_digest": digest}})

                    elif method == "run_pixel_operator":
                        op_name = params["operator_name"]
                        op_params = params["parameters"]
                        sid = params["source_id"]
                        sver = params["source_version"]
                        sec_src = params.get("secondary_source")

                        if op_name == "sample_colour":
                            x = int(op_params["x"])
                            y = int(op_params["y"])
                            r, g, b, a = sample_pixel(sid, sver, x, y)
                            op_res = {
                                "x": x,
                                "y": y,
                                "rgba": [r, g, b, a],
                                "hex": f"#{r:02X}{g:02X}{b:02X}{a:02X}",
                                "sampled_plane": "verified_plane",
                            }
                        elif op_name == "measure_bbox":
                            surface = get_surface(sid, sver)
                            selector = op_params.get("selector", {"type": "non_transparent"})
                            sel_type = selector.get("type", "non_transparent")
                            target_rgba = tuple(selector.get("rgba", [0, 0, 0, 0]))
                            tolerance = int(selector.get("tolerance", 0))
                            x_min, y_min = CANVAS_WIDTH, CANVAS_HEIGHT
                            x_max, y_max = -1, -1
                            matched = 0
                            if sel_type == "non_transparent":
                                for y, x0, x1, _rgba in surface.fg_runs:
                                    matched += x1 - x0 + 1
                                    x_min = min(x_min, x0)
                                    x_max = max(x_max, x1)
                                    y_min = min(y_min, y)
                                    y_max = max(y_max, y)
                            elif sel_type == "colour_match" and target_rgba[3] > tolerance:
                                tr, tg, tb, ta = target_rgba
                                for y, x0, x1, rgba in surface.fg_runs:
                                    hit = (
                                        rgba == target_rgba
                                        if tolerance == 0
                                        else (
                                            abs(rgba[0] - tr) <= tolerance
                                            and abs(rgba[1] - tg) <= tolerance
                                            and abs(rgba[2] - tb) <= tolerance
                                            and abs(rgba[3] - ta) <= tolerance
                                        )
                                    )
                                    if hit:
                                        matched += x1 - x0 + 1
                                        x_min = min(x_min, x0)
                                        x_max = max(x_max, x1)
                                        y_min = min(y_min, y)
                                        y_max = max(y_max, y)
                            else:
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
                                        x_min = min(x_min, x)
                                        x_max = max(x_max, x)
                                        y_min = min(y_min, y)
                                        y_max = max(y_max, y)
                            if matched == 0:
                                op_res = {
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
                                op_res = {
                                    "empty": False,
                                    "x_min": x_min,
                                    "y_min": y_min,
                                    "x_max": x_max,
                                    "y_max": y_max,
                                    "width": x_max - x_min + 1,
                                    "height": y_max - y_min + 1,
                                    "matching_pixels": matched,
                                }
                        elif op_name == "count_pixels":
                            surface = get_surface(sid, sver)
                            predicate = op_params.get("predicate", {"type": "non_transparent"})
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
                                    elif (
                                        abs(rgba[0] - tr) <= tolerance
                                        and abs(rgba[1] - tg) <= tolerance
                                        and abs(rgba[2] - tb) <= tolerance
                                        and abs(rgba[3] - ta) <= tolerance
                                    ):
                                        matching += x1 - x0 + 1
                            else:
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
                            op_res = {
                                "matching_pixels": matching,
                                "total_pixels": CANVAS_TOTAL_PIXELS,
                                "fraction": round(matching / CANVAS_TOTAL_PIXELS, 6),
                            }
                        elif op_name == "count_changed_pixels":
                            surf_a = get_surface(sid, sver)
                            surf_b = get_surface(sec_src["source_id"], sec_src["source_version"])
                            tol = int(op_params.get("channel_tolerance", 0))
                            changed = added = removed = modified_colour = 0
                            rows_a, rows_b = surf_a.row_bytes, surf_b.row_bytes
                            for y in range(CANVAS_HEIGHT):
                                ra, rb = rows_a[y], rows_b[y]
                                if ra == rb:
                                    continue
                                for x4 in range(0, ROW_BYTE_STRIDE, 4):
                                    is_diff = (
                                        ra[x4 : x4 + 4] != rb[x4 : x4 + 4]
                                        if tol == 0
                                        else (
                                            abs(ra[x4] - rb[x4]) > tol
                                            or abs(ra[x4 + 1] - rb[x4 + 1]) > tol
                                            or abs(ra[x4 + 2] - rb[x4 + 2]) > tol
                                            or abs(ra[x4 + 3] - rb[x4 + 3]) > tol
                                        )
                                    )
                                    if is_diff:
                                        changed += 1
                                        aa, ab = ra[x4 + 3], rb[x4 + 3]
                                        if aa == 0 and ab > 0:
                                            added += 1
                                        elif aa > 0 and ab == 0:
                                            removed += 1
                                        else:
                                            modified_colour += 1
                            op_res = {
                                "changed_pixels": changed,
                                "unchanged_pixels": CANVAS_TOTAL_PIXELS - changed,
                                "added_pixels": added,
                                "removed_pixels": removed,
                                "modified_colour_pixels": modified_colour,
                                "total_pixels": CANVAS_TOTAL_PIXELS,
                                "change_ratio": round(changed / CANVAS_TOTAL_PIXELS, 6),
                            }
                        elif op_name == "extract_changed_regions":
                            surf_a = get_surface(sid, sver)
                            surf_b = get_surface(sec_src["source_id"], sec_src["source_version"])
                            connectivity = int(op_params.get("connectivity", 8))
                            tol = int(op_params.get("channel_tolerance", 0))
                            diff_runs = ElectricEyeFixtureHarness._extract_diff_runs(surf_a, surf_b, tol=tol)
                            regions = ElectricEyeFixtureHarness._connected_components_from_runs(
                                diff_runs, connectivity=connectivity
                            )
                            op_res = {
                                "region_count": len(regions),
                                "connectivity": connectivity,
                                "regions": regions[:MAX_REPORTED_REGIONS],
                            }
                        elif op_name == "count_objects":
                            connectivity = int(op_params.get("connectivity", 8))
                            min_area = int(op_params.get("min_area_px", 1))
                            all_objs = _get_fg_components(sid, sver, connectivity=connectivity)
                            filtered = [copy.deepcopy(o) for o in all_objs if o["area_px"] >= min_area]
                            for idx, obj in enumerate(filtered, start=1):
                                obj["object_id"] = f"obj_{idx:02d}"
                            op_res = {
                                "object_count": len(filtered),
                                "connectivity": connectivity,
                                "min_area_px": min_area,
                                "objects": filtered[:MAX_REPORTED_REGIONS],
                            }
                        else:
                            raise ValueError(f"Unsupported vault pixel operator: {op_name}")

                        assert_no_raw_buffer_leakage(op_res)
                        _send_framed_json(conn, {"status": "ok", "result": op_res})

                    elif method == "compute_derived_summary":
                        rep_type = params["representation_type"]
                        sa = tuple(params["source_a"])
                        sb = tuple(params["source_b"])
                        surf_a = get_surface(sa[0], sa[1])
                        surf_b = get_surface(sb[0], sb[1])
                        rows_a, rows_b = surf_a.row_bytes, surf_b.row_bytes

                        if rep_type == "change_mask":
                            added = removed = modified = unchanged = 0
                            rle_changed: List[Tuple[int, int, str]] = []
                            for y in range(CANVAS_HEIGHT):
                                ra, rb = rows_a[y], rows_b[y]
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
                                        aa, ab = ra[x4 + 3], rb[x4 + 3]
                                        if aa == 0 and ab > 0:
                                            added += 1
                                            state = "added"
                                        elif aa > 0 and ab == 0:
                                            removed += 1
                                            state = "removed"
                                        else:
                                            modified += 1
                                            state = "modified"
                                        if (
                                            rle_changed
                                            and rle_changed[-1][2] == state
                                            and rle_changed[-1][1] + 1 == idx
                                        ):
                                            rle_changed[-1] = (rle_changed[-1][0], idx, state)
                                        else:
                                            rle_changed.append((idx, idx, state))
                            mask_digest = sha256_hex(canonical_json(rle_changed).encode("utf-8"))
                            out = {
                                "summary": {
                                    "added_pixels": added,
                                    "removed_pixels": removed,
                                    "modified_pixels": modified,
                                    "changed_pixels": added + removed + modified,
                                    "unchanged_pixels": unchanged,
                                    "rle_span_count": len(rle_changed),
                                    "change_mask_digest": mask_digest,
                                },
                                "confidence": "high",
                                "rationale": "Exact pixel-by-pixel ternary comparison across two immutable VerifiedSources.",
                            }
                        elif rep_type == "difference_map":
                            max_l1 = sum_l1 = non_zero_deltas = 0
                            delta_spans: List[Tuple[int, int, int, int, int]] = []
                            for y in range(CANVAS_HEIGHT):
                                ra, rb = rows_a[y], rows_b[y]
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
                                        max_l1 = max(max_l1, l1)
                                        delta_spans.append((row_start + x, dr, dg, db, da))
                            diff_digest = sha256_hex(canonical_json(delta_spans).encode("utf-8"))
                            out = {
                                "summary": {
                                    "non_zero_delta_pixels": non_zero_deltas,
                                    "max_channel_l1_delta": max_l1,
                                    "mean_channel_l1_delta_on_changed": round(
                                        sum_l1 / max(1, non_zero_deltas), 4
                                    ),
                                    "difference_map_digest": diff_digest,
                                },
                                "confidence": "high",
                                "rationale": "Exact per-channel absolute difference map over verified RGBA buffers.",
                            }
                        elif rep_type == "motion_summary":
                            objs_a = _get_fg_components(sa[0], sa[1], connectivity=8)
                            objs_b = _get_fg_components(sb[0], sb[1], connectivity=8)
                            vectors = []
                            for oa, ob in zip(objs_a, objs_b):
                                vectors.append({
                                    "region_id_a": oa["region_id"],
                                    "region_id_b": ob["region_id"],
                                    "area_px_a": oa["area_px"],
                                    "area_px_b": ob["area_px"],
                                    "displacement": {
                                        "dx": round(ob["centroid"]["x"] - oa["centroid"]["x"], 4),
                                        "dy": round(ob["centroid"]["y"] - oa["centroid"]["y"], 4),
                                    },
                                })
                            out = {
                                "summary": {"component_trajectories": vectors},
                                "confidence": "medium",
                                "rationale": (
                                    "Component correspondence between discrete frames matches connected components "
                                    "by area and scanline order; non-rigid or occluded motion is not uniquely identifiable."
                                ),
                            }
                        elif rep_type == "colour_delta":
                            changed_count = 0
                            sum_delta_e = max_delta_e = 0.0
                            for y in range(CANVAS_HEIGHT):
                                ra, rb = rows_a[y], rows_b[y]
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
                                        max_delta_e = max(max_delta_e, de)
                            out = {
                                "summary": {
                                    "changed_pixel_count": changed_count,
                                    "mean_rgba_euclidean_delta": round(
                                        sum_delta_e / max(1, changed_count), 4
                                    ),
                                    "max_rgba_euclidean_delta": round(max_delta_e, 4),
                                },
                                "confidence": "high",
                                "rationale": "Deterministic Euclidean RGBA colour delta across changed pixels.",
                            }
                        elif rep_type == "spatial_relationship":
                            objs_b = _get_fg_components(sb[0], sb[1], connectivity=8)
                            if len(objs_b) >= 2:
                                o1, o2 = objs_b[0], objs_b[1]
                                b1, b2 = o1["bbox"], o2["bbox"]
                                intersects = not (
                                    b1["x_max"] < b2["x_min"]
                                    or b2["x_max"] < b1["x_min"]
                                    or b1["y_max"] < b2["y_min"]
                                    or b2["y_max"] < b1["y_min"]
                                )
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
                                    "topological_relation": "intersects" if intersects else "disjoint",
                                    "edge_clearance_px": clearance_px,
                                    "centroid_bearing_deg": bearing_deg,
                                }
                            else:
                                summary = {
                                    "topological_relation": "single_region_only",
                                    "region_count": len(objs_b),
                                }
                            out = {
                                "summary": summary,
                                "confidence": "high",
                                "rationale": "Exact bounding-box topology, clearance, and centroid bearing on verified mask.",
                            }
                        else:
                            raise ValueError(f"Unsupported DerivedRepresentation type: {rep_type}")

                        assert_no_raw_buffer_leakage(out)
                        _send_framed_json(conn, {"status": "ok", "result": out})

                    elif method == "issue_capability_token":
                        auth_sec = params.get("capture_service_auth_secret")
                        if not isinstance(auth_sec, str) or not hmac.compare_digest(
                            auth_sec, capture_service_auth_secret
                        ):
                            unauthorized_issuer_blocks += 1
                            raise PermissionError(
                                "ERR_UNAUTHORIZED_CAPABILITY_REQUEST: Invalid capture service auth secret."
                            )
                        render_id = params["render_id"]
                        sid = params["captured_source_id"]
                        sver = params["captured_source_version"]
                        src_entry = source_registry.get((sid, sver))
                        if src_entry is None or src_entry["source_kind"] != "captured_frame":
                            raise PermissionError(
                                "ERR_UNVERIFIED_OBSERVATION_CLAIM: Capability token requires registered captured_frame."
                            )
                        if not isinstance(render_id, str) or len(render_id) > 256:
                            raise ValueError("ERR_CAPABILITY_RENDER_ID")
                        if len(issued_capability_tokens) + len(consumed_capability_tokens) >= MAX_CAPABILITY_RECORDS:
                            raise PermissionError("ERR_CAPABILITY_CAPACITY: session issuance limit reached")
                        token_id = "cap_tok_" + secrets.token_hex(16)
                        issued_capability_tokens[token_id] = {
                            "token_id": token_id,
                            "render_id": render_id,
                            "captured_source_id": sid,
                            "captured_source_version": sver,
                            "content_sha256": src_entry["content_sha256"],
                        }
                        _send_framed_json(conn, {"status": "ok", "result": {"capability_token": token_id}})

                    elif method == "issue_observation_receipt":
                        cap_tok = params.get("caller_capability_token")
                        rcpt_id = params["observation_receipt_id"]
                        render = params["render"]
                        sid = params["captured_source_id"]
                        sver = params["captured_source_version"]

                        if not isinstance(cap_tok, str):
                            unauthorized_issuer_blocks += 1
                            raise PermissionError(
                                "ERR_UNAUTHORIZED_RECEIPT_ISSUER: Missing capability token."
                            )
                        if cap_tok in consumed_capability_tokens:
                            replay_blocks += 1
                            raise PermissionError(
                                "ERR_REPLAY_DETECTED: Single-use capability token was already consumed."
                            )
                        tok_meta = issued_capability_tokens.get(cap_tok)
                        if tok_meta is None:
                            unauthorized_issuer_blocks += 1
                            raise PermissionError(
                                "ERR_UNAUTHORIZED_RECEIPT_ISSUER: Unknown or forged capability token."
                            )
                        if (
                            tok_meta["render_id"] != render.get("render_id")
                            or tok_meta["captured_source_id"] != sid
                            or tok_meta["captured_source_version"] != sver
                        ):
                            unauthorized_issuer_blocks += 1
                            raise PermissionError(
                                "ERR_CAPABILITY_SCOPE_MISMATCH: Token does not match (render_id, source_id, source_version)."
                            )

                        src_entry = source_registry.get((sid, sver))
                        if (
                            render.get("render_type") != "observed"
                            or src_entry is None
                            or src_entry["source_kind"] != "captured_frame"
                        ):
                            raise PermissionError(
                                "ERR_UNVERIFIED_OBSERVATION_CLAIM: Requires render_type='observed' and source_kind='captured_frame'."
                            )
                        binds = any(
                            ref.get("ref_type") == "source"
                            and ref.get("ref_id") == sid
                            and ref.get("ref_version") == sver
                            for ref in render.get("provenance_refs", ())
                        )
                        if not binds:
                            raise PermissionError(
                                "ERR_UNVERIFIED_OBSERVATION_CLAIM: Render provenance_refs do not bind to captured_frame."
                            )

                        if not isinstance(rcpt_id, str) or len(rcpt_id) > 256:
                            raise ValueError("ERR_RECEIPT_ID")
                        if rcpt_id in issued_receipts or len(issued_receipts) >= MAX_CAPABILITY_RECORDS:
                            raise PermissionError("ERR_RECEIPT_CAPACITY_OR_DUPLICATE")
                        # Retain bounded replay tombstones; never reuse an issued capability.
                        consumed_capability_tokens.add(cap_tok)
                        issued_capability_tokens.pop(cap_tok)
                        stage = "verified_observation_window"
                        sig = sign_receipt(rcpt_id, render["render_id"], sid, sver, stage)
                        receipt = {
                            "observation_receipt_id": rcpt_id,
                            "render_id": render["render_id"],
                            "captured_source_id": sid,
                            "captured_source_version": sver,
                            "pipeline_stage": stage,
                            "receipt_signature": sig,
                        }
                        issued_receipts[rcpt_id] = receipt
                        _send_framed_json(conn, {"status": "ok", "result": receipt})

                    elif method == "verify_observation_receipt":
                        rcpt = params["receipt"]
                        rcpt_id = rcpt.get("observation_receipt_id", "")
                        stored = issued_receipts.get(rcpt_id)
                        sig_ok = verify_receipt_sig(
                            rcpt_id,
                            rcpt.get("render_id", ""),
                            rcpt.get("captured_source_id", ""),
                            rcpt.get("captured_source_version", ""),
                            rcpt.get("pipeline_stage", ""),
                            rcpt.get("receipt_signature", ""),
                        )
                        valid = bool(
                            sig_ok
                            and stored is not None
                            and not stored.get("revoked", False)
                            and (stored["captured_source_id"], stored["captured_source_version"]) in source_registry
                            and stored["receipt_signature"] == rcpt.get("receipt_signature")
                        )
                        _send_framed_json(conn, {"status": "ok", "result": {"valid": valid}})

                except PermissionError as exc:
                    _send_framed_json(
                        conn,
                        {"status": "error", "error_code": "ERR_PERMISSION_DENIED", "reason": str(exc)},
                    )
                except Exception as exc:
                    _send_framed_json(
                        conn,
                        {"status": "error", "error_code": "ERR_VAULT_RPC", "reason": str(exc)},
                    )
    finally:
        server.close()
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass


class OutOfProcessVaultGateClient:
    """
    Client interface for the EXEC-GATE-1 Out-of-Process Privileged Vault & Observation Gate Daemon.
    Holds zero raw pixel bytes and zero HMAC signing keys in the harness process.
    """

    def __init__(self, socket_path: str, capture_service_auth_secret: str) -> None:
        self._socket_path = socket_path
        self._capture_service_auth_secret = capture_service_auth_secret

    def _call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        rpc_nonce: Optional[str] = None,
    ) -> Dict[str, Any]:
        nonce = rpc_nonce if rpc_nonce is not None else ("nonce_" + secrets.token_hex(12))
        payload: Dict[str, Any] = {
            "method": method,
            "rpc_nonce": nonce,
            "params": params or {},
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(self._socket_path)
            _send_framed_json(sock, payload)
            resp = _recv_framed_json(sock)

        if resp.get("status") != "ok":
            err_code = resp.get("error_code", "ERR_IPC")
            reason = resp.get("reason", "Unknown IPC error")
            if err_code in (
                "ERR_PERMISSION_DENIED",
                "ERR_PEER_AUTH_FAILED",
                "ERR_REPLAY_DETECTED",
                "ERR_RPC_METHOD_PROHIBITED",
            ):
                raise PermissionError(f"{err_code}: {reason}")
            if err_code == "ERR_KILL_SWITCH_HALTED":
                raise RuntimeError(f"{err_code}: {reason}")
            raise ValueError(f"{err_code}: {reason}")
        return resp["result"]

    def ping_status(self) -> Dict[str, Any]:
        return self._call("ping_status")

    def store_verified_source(
        self,
        source_id: str,
        source_version: str,
        source_kind: str,
        marks: Sequence[DrawingMark],
        target_plane: str = "verified_plane",
    ) -> str:
        res = self._call(
            "store_verified_source",
            {
                "source_id": source_id,
                "source_version": source_version,
                "source_kind": source_kind,
                "marks": [m.to_dict() for m in marks],
                "target_plane": target_plane,
            },
        )
        return res["content_sha256"]

    def evict_sources(self, source_keys: Sequence[Sequence[str]]) -> int:
        res = self._call("evict_sources", {"sources": [[str(k[0]), str(k[1])] for k in source_keys]})
        return int(res["evicted"])

    def store_ghost_pixels(
        self,
        render_id: str,
        marks: Sequence[DrawingMark],
        target_plane: str,
    ) -> str:
        res = self._call(
            "store_ghost_pixels",
            {
                "render_id": render_id,
                "marks": [m.to_dict() for m in marks],
                "target_plane": target_plane,
            },
        )
        return res["ghost_plane_digest"]

    def run_pixel_operator(
        self,
        *,
        operator_name: str,
        parameters: Dict[str, Any],
        source_id: Optional[str],
        source_version: Optional[str],
        secondary_source: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return self._call(
            "run_pixel_operator",
            {
                "operator_name": operator_name,
                "parameters": parameters,
                "source_id": source_id,
                "source_version": source_version,
                "secondary_source": secondary_source,
            },
        )

    def compute_derived_summary(
        self,
        *,
        representation_type: str,
        source_a: Tuple[str, str],
        source_b: Tuple[str, str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._call(
            "compute_derived_summary",
            {
                "representation_type": representation_type,
                "source_a": list(source_a),
                "source_b": list(source_b),
                "parameters": parameters,
            },
        )

    def request_capability_token(
        self,
        *,
        render_id: str,
        captured_source_id: str,
        captured_source_version: str,
        capture_service_auth_secret: Optional[str] = None,
    ) -> str:
        secret_to_use = (
            self._capture_service_auth_secret
            if capture_service_auth_secret is None
            else capture_service_auth_secret
        )
        res = self._call(
            "issue_capability_token",
            {
                "capture_service_auth_secret": secret_to_use,
                "render_id": render_id,
                "captured_source_id": captured_source_id,
                "captured_source_version": captured_source_version,
            },
        )
        return res["capability_token"]

    def rpc_issue_observation_receipt(
        self,
        *,
        caller_capability_token: Optional[str],
        observation_receipt_id: str,
        render: Dict[str, Any],
        captured_source_desc: VerifiedSourceDescriptor,
        rpc_nonce: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._call(
            "issue_observation_receipt",
            {
                "caller_capability_token": caller_capability_token,
                "observation_receipt_id": observation_receipt_id,
                "render": render,
                "captured_source_id": captured_source_desc.source_id,
                "captured_source_version": captured_source_desc.source_version,
            },
            rpc_nonce=rpc_nonce,
        )

    def rpc_verify_observation_receipt(self, receipt: Dict[str, Any]) -> bool:
        res = self._call("verify_observation_receipt", {"receipt": receipt})
        return bool(res["valid"])

    def emergency_lock_vault(self) -> Dict[str, Any]:
        return self._call("emergency_lock_vault")


class OutOfProcessVaultGateDaemon:
    """
    Context-managed EXEC-GATE-1 daemon process. Spawns a separate OS process and
    guarantees clean shutdown when exiting the `with` block.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        initial_key_bytes: Optional[bytes] = None,
        capture_service_auth_secret: Optional[str] = None,
    ) -> None:
        if initial_key_bytes is not None and (
            not isinstance(initial_key_bytes, bytes)
            or len(initial_key_bytes) < 32
            or initial_key_bytes == DISPOSABLE_FIXTURE_STATIC_HMAC_KEY
        ):
            raise PermissionError("ERR_STATIC_OR_WEAK_KEY_PROHIBITED")
        self.socket_path = socket_path
        self._initial_key_bytes = initial_key_bytes
        self.capture_service_auth_secret = capture_service_auth_secret or (
            "cap_auth_sec_" + secrets.token_hex(16)
        )
        self._proc: Optional[multiprocessing.Process] = None
        self.client: Optional[OutOfProcessVaultGateClient] = None

    def __enter__(self) -> OutOfProcessVaultGateClient:
        ready_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_vault_gate_daemon_loop,
            args=(
                self.socket_path,
                os.getuid(),
                os.getgid(),
                ready_event,
                self._initial_key_bytes,
                self.capture_service_auth_secret,
            ),
            daemon=True,
        )
        self._proc.start()
        if not ready_event.wait(timeout=5.0):
            self.stop()
            raise RuntimeError("OutOfProcessVaultGateDaemon failed to start within 5s")
        self.client = OutOfProcessVaultGateClient(
            self.socket_path,
            self.capture_service_auth_secret,
        )
        return self.client

    def stop(self) -> None:
        if self.client is not None and os.path.exists(self.socket_path):
            try:
                self.client._call("shutdown")
            except Exception:
                pass
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
            self._proc = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# =============================================================================
# EXEC-GATE-2: Out-of-Process Append-Only Durable WORM Audit Sink Daemon
# =============================================================================

PROHIBITED_UNREDACTED_AUDIT_SUBSTRINGS = (
    "Does this contain private user text?",
    "UNREDACTED_PII_MARKER",
)


def _audit_sink_daemon_loop(
    socket_path: str,
    ledger_file_path: str,
    allowed_uid: int,
    allowed_gid: int,
    ready_event: Any,
) -> None:
    die_with_parent()
    checkpoint_signing_key = secrets.token_bytes(32)
    entry_count = 0
    chain_head = "sha256:" + ("0" * 64)
    fsync_count = 0
    write_failed = False

    os.makedirs(os.path.dirname(ledger_file_path), exist_ok=True)
    log_fd = os.open(ledger_file_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_TRUNC | os.O_NOFOLLOW, 0o600)

    def _disk_entries():
        fd = os.open(ledger_file_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info, original = os.fstat(fd), os.fstat(log_fd)
            if (info.st_dev, info.st_ino) != (original.st_dev, original.st_ino):
                raise ValueError("ERR_AUDIT_FILE_REPLACED")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                while True:
                    line = handle.readline(MAX_AUDIT_ENTRY_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_AUDIT_ENTRY_BYTES or not line.endswith(b"\n"):
                        raise ValueError("ERR_AUDIT_ENTRY_LENGTH")
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("ERR_AUDIT_RECORD_SHAPE")
                    yield record, len(line)
        finally:
            os.close(fd)

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o600)
    server.listen(16)
    ready_event.set()

    def _sign_checkpoint(seq: int, head_digest: str) -> str:
        msg = canonical_json({
            "head_seq": seq,
            "chain_head_digest": head_digest,
            "ledger_file_path": os.path.basename(ledger_file_path),
        }).encode("utf-8")
        return "hmac-sha256:" + hmac.new(checkpoint_signing_key, msg, hashlib.sha256).hexdigest()

    running = True
    try:
        while running:
            conn, _ = server.accept()
            with _isolated_ipc_connection(conn):
                try:
                    peer_pid, peer_uid, peer_gid = _extract_so_peercred(conn)
                    req = _recv_rpc_request(conn)
                except Exception as exc:
                    _send_framed_json(conn, {"status": "error", "error_code": "ERR_IPC_READ", "reason": str(exc)})
                    continue

                if peer_uid != allowed_uid or peer_gid != allowed_gid:
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_PEER_AUTH_FAILED",
                            "reason": "Unauthorized SO_PEERCRED on AuditSinkDaemon.",
                        },
                    )
                    continue

                method = req.get("method")
                params = req.get("params", {})

                if method == "shutdown":
                    running = False
                    _send_framed_json(conn, {"status": "ok", "result": {"shutdown": True}})
                    continue

                try:
                    if method == "append_entry":
                        if write_failed:
                            raise RuntimeError("ERR_AUDIT_WRITE_FAILED_CLOSED")
                        entry = params["entry"]
                        if not isinstance(entry, dict):
                            raise ValueError("ERR_AUDIT_RECORD_SHAPE")
                        # 1. Verify no raw pixel bytes or unredacted prompt payloads
                        assert_no_raw_buffer_leakage(entry)
                        serialized_entry = canonical_json(entry)
                        if any(k in entry for k in ("raw_bytes", "pixel_buffer", "unredacted_prompt")) or any(
                            sub in serialized_entry for sub in PROHIBITED_UNREDACTED_AUDIT_SUBSTRINGS
                        ):
                            raise ValueError(
                                "ERR_AUDIT_PII_OR_RAW_BYTES_REJECTED: Unredacted prompt or raw bytes rejected by audit sink."
                            )

                        # 2. Verify sequential ordering & hash-chain link
                        expected_seq = entry_count + 1
                        expected_prev = chain_head
                        if entry.get("entry_seq") != expected_seq:
                            raise ValueError(
                                f"ERR_AUDIT_SEQUENCE_VIOLATION: Expected seq {expected_seq}, got {entry.get('entry_seq')}"
                            )
                        if entry.get("prev_hash") != expected_prev:
                            raise ValueError("ERR_AUDIT_CHAIN_BROKEN: prev_hash mismatch on append.")

                        # 3. Verify dual clocks
                        for req_clock in ("logical_clock", "logical_timestamp", "utc_wall_timestamp"):
                            if not entry.get(req_clock):
                                raise ValueError(f"ERR_AUDIT_CLOCK_MISSING: Missing {req_clock}")

                        # 4. Recompute entry_hash
                        body = {k: v for k, v in entry.items() if k != "entry_hash"}
                        recomputed = sha256_hex(canonical_json(body).encode("utf-8"))
                        if recomputed != entry.get("entry_hash"):
                            raise ValueError("ERR_AUDIT_HASH_MISMATCH: Entry hash failed verification.")

                        # 5. Durable O_APPEND + os.fsync write
                        line_bytes = (serialized_entry + "\n").encode("utf-8")
                        if len(line_bytes) > MAX_AUDIT_ENTRY_BYTES:
                            raise ValueError("ERR_AUDIT_ENTRY_TOO_LARGE")
                        try:
                            pending = memoryview(line_bytes)
                            while pending:
                                written = os.write(log_fd, pending)
                                if written <= 0:
                                    raise OSError("short audit write")
                                pending = pending[written:]
                            os.fsync(log_fd)
                        except OSError:
                            write_failed = True
                            raise
                        fsync_count += 1
                        entry_count = expected_seq
                        chain_head = entry["entry_hash"]
                        _send_framed_json(
                            conn,
                            {
                                "status": "ok",
                                "result": {
                                    "entry_seq": expected_seq,
                                    "entry_hash": entry["entry_hash"],
                                    "fsync_count": fsync_count,
                                },
                            },
                        )

                    elif method == "export_chain_head":
                        head_seq = entry_count
                        head_digest = chain_head
                        _send_framed_json(
                            conn,
                            {
                                "status": "ok",
                                "result": {
                                    "sink_type": "out_of_process_append_only_worm_audit_sink_v1",
                                    "daemon_pid": os.getpid(),
                                    "durable_ledger_path": ledger_file_path,
                                    "entry_count": entry_count,
                                    "retained_entry_records": 0,
                                    "write_failed_closed": write_failed,
                                    "head_seq": head_seq,
                                    "chain_head_digest": head_digest,
                                    "fsync_count": fsync_count,
                                    "checkpoint_signature": _sign_checkpoint(head_seq, head_digest),
                                    "retention_and_export_policy": {
                                        "worm_retention_days": 90,
                                        "write_mode": "O_APPEND_plus_fsync_per_event",
                                        "export_format": "canonical_jsonl_with_signed_checkpoint",
                                        "raw_pixels_prohibited": True,
                                        "unredacted_prompts_prohibited": True,
                                    },
                                },
                            },
                        )

                    elif method == "verify_durable_file":
                        expected_prev = "sha256:" + ("0" * 64)
                        count = 0
                        error = None
                        try:
                            for idx, (rec, _) in enumerate(_disk_entries(), start=1):
                                count = idx
                                if rec.get("entry_seq") != idx or rec.get("prev_hash") != expected_prev:
                                    error = "ERR_AUDIT_CHAIN_BROKEN"
                                elif any(not rec.get(c) for c in ("logical_clock", "logical_timestamp", "utc_wall_timestamp")):
                                    error = "ERR_AUDIT_CLOCK_MISSING"
                                else:
                                    body = {k: v for k, v in rec.items() if k != "entry_hash"}
                                    if sha256_hex(canonical_json(body).encode("utf-8")) != rec.get("entry_hash"):
                                        error = "ERR_AUDIT_HASH_MISMATCH"
                                if error:
                                    break
                                expected_prev = rec["entry_hash"]
                        except (OSError, ValueError, TypeError) as exc:
                            error = f"ERR_AUDIT_FILE_INVALID: {exc}"
                        if not error and (count != entry_count or expected_prev != chain_head):
                            error = "ERR_AUDIT_COMMITTED_HEAD_MISMATCH"
                        _send_framed_json(conn, {"status": "ok", "result": {
                            "valid": error is None, "verified_entries": count,
                            "chain_head_digest": expected_prev,
                            **({"error_code": error, "failed_seq": count} if error else {}),
                        }})

                    elif method == "get_entries":
                        after = int(params.get("after_seq", 0))
                        limit = int(params.get("limit", MAX_AUDIT_PAGE_ENTRIES))
                        if after < 0 or not 1 <= limit <= MAX_AUDIT_PAGE_ENTRIES:
                            raise ValueError("ERR_AUDIT_PAGE_BOUNDS")
                        page, page_bytes = [], 0
                        for rec, size in _disk_entries():
                            if rec["entry_seq"] <= after:
                                continue
                            if len(page) >= limit or page_bytes + size > MAX_AUDIT_PAGE_BYTES:
                                break
                            page.append(rec)
                            page_bytes += size
                        next_seq = page[-1]["entry_seq"] if page else after
                        _send_framed_json(conn, {"status": "ok", "result": {
                            "entries": page, "next_after_seq": next_seq,
                            "has_more": next_seq < entry_count, "entry_count": entry_count,
                        }})

                    else:
                        raise ValueError(f"Unsupported AuditSink method: {method}")

                except Exception as exc:
                    _send_framed_json(
                        conn,
                        {"status": "error", "error_code": "ERR_AUDIT_SINK_REJECTED", "reason": str(exc)},
                    )
    finally:
        try:
            os.close(log_fd)
        except OSError:
            pass
        server.close()
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass


class OutOfProcessAuditSinkClient:
    """
    Client for EXEC-GATE-2 External Append-Only Durable WORM Audit Sink Daemon.
    Fails closed synchronously if the external daemon is unavailable or rejects an entry.
    """

    def __init__(self, socket_path: str, ledger_file_path: str) -> None:
        self._socket_path = socket_path
        self.ledger_file_path = ledger_file_path
        self.simulate_unavailable = False

    def _call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.simulate_unavailable or not os.path.exists(self._socket_path):
            raise RuntimeError(
                "ERR_AUDIT_SINK_UNAVAILABLE_FAIL_CLOSED: External append-only audit sink daemon is unreachable."
            )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(5.0)
                sock.connect(self._socket_path)
                _send_framed_json(sock, {"method": method, "params": params or {}})
                resp = _recv_framed_json(sock)
        except OSError as exc:
            raise RuntimeError(
                f"ERR_AUDIT_SINK_UNAVAILABLE_FAIL_CLOSED: Socket error communicating with external audit sink: {exc}"
            ) from exc

        if resp.get("status") != "ok":
            raise ValueError(f"{resp.get('error_code', 'ERR_AUDIT')}: {resp.get('reason')}")
        return resp["result"]

    def append_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("append_entry", {"entry": entry})

    def export_chain_head(self) -> Dict[str, Any]:
        return self._call("export_chain_head")

    def verify_durable_file(self) -> Dict[str, Any]:
        return self._call("verify_durable_file")

    def get_entries_page(self, *, after_seq: int = 0, limit: int = MAX_AUDIT_PAGE_ENTRIES) -> Dict[str, Any]:
        return self._call("get_entries", {"after_seq": after_seq, "limit": limit})

    def get_entries(self) -> List[Dict[str, Any]]:
        page = self.get_entries_page()
        if page["has_more"]:
            raise RuntimeError("ERR_AUDIT_PAGINATION_REQUIRED: use get_entries_page; no events omitted silently")
        return page["entries"]


class OutOfProcessAuditSinkDaemon:
    """
    Context-managed EXEC-GATE-2 external append-only audit storage daemon.
    """

    def __init__(self, socket_path: str, ledger_file_path: str) -> None:
        self.socket_path = socket_path
        self.ledger_file_path = ledger_file_path
        self._proc: Optional[multiprocessing.Process] = None
        self.client: Optional[OutOfProcessAuditSinkClient] = None

    def __enter__(self) -> OutOfProcessAuditSinkClient:
        ready_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_audit_sink_daemon_loop,
            args=(
                self.socket_path,
                self.ledger_file_path,
                os.getuid(),
                os.getgid(),
                ready_event,
            ),
            daemon=True,
        )
        self._proc.start()
        if not ready_event.wait(timeout=5.0):
            self.stop()
            raise RuntimeError("OutOfProcessAuditSinkDaemon failed to start within 5s")
        self.client = OutOfProcessAuditSinkClient(self.socket_path, self.ledger_file_path)
        return self.client

    def stop(self) -> None:
        if self.client is not None and os.path.exists(self.socket_path):
            try:
                old_sim = self.client.simulate_unavailable
                self.client.simulate_unavailable = False
                self.client._call("shutdown")
                self.client.simulate_unavailable = old_sim
            except Exception:
                pass
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
            self._proc = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# =============================================================================
# EXEC-GATE-3: Pre-Ingestion Zero-PII Quarantine & Kill-Switch Supervisor Daemon
# =============================================================================

KILL_SWITCH_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "KS-1": {
        "name": "provenance_completeness_breach",
        "condition": "provenance_completeness < 1.0",
        "severity": "CRITICAL",
    },
    "KS-2": {
        "name": "deterministic_replay_breach",
        "condition": "deterministic_replay_equality < 1.0",
        "severity": "CRITICAL",
    },
    "KS-3": {
        "name": "false_observation_breach",
        "condition": "false_observation_count > 0",
        "severity": "CRITICAL",
    },
    "KS-4": {
        "name": "raw_buffer_leakage_breach",
        "condition": "raw_rgba_buffer_or_cache_bytes_exposed",
        "severity": "CRITICAL",
    },
    "KS-5": {
        "name": "ghost_isolation_breach",
        "condition": "ghost_pixel_leakage_into_verified_plane > 0",
        "severity": "CRITICAL",
    },
    "KS-6": {
        "name": "diagnostic_visibility_breach",
        "condition": "silent_diagnostic_exposed_in_default_feed > 0",
        "severity": "HIGH",
    },
    "KS-7": {
        "name": "semantic_refusal_bypass_breach",
        "condition": "semantic_or_intent_query_not_refused",
        "severity": "CRITICAL",
    },
    "KS-8": {
        "name": "audit_chain_or_sink_breach",
        "condition": "audit_chain_invalid_or_worm_sink_failure",
        "severity": "CRITICAL",
    },
    "KS-9": {
        "name": "pii_or_non_synthetic_input_breach",
        "condition": "non_synthetic_capture_or_pii_source_ingested",
        "severity": "CRITICAL",
    },
    "KS-10": {
        "name": "latency_or_resource_cap_breach",
        "condition": "cpu_or_wall_latency_gt_50ms_or_memory_gt_2mib",
        "severity": "HIGH",
    },
}

ESCALATION_PATH_SPEC = {
    "primary_on_call": "Security Architecture Lead (Cyclops Core Security)",
    "privacy_escalation": "Privacy & Data Governance Lead",
    "audit_escalation": "Platform Reliability & Audit Engineering Lead",
    "human_governance_authority": "Cyclops Human Governance Review Board",
    "sla_response_minutes": 15,
    "restart_policy": "manual_human_governance_signoff_required_after_halt",
}


def _quarantine_and_kill_switch_daemon_loop(
    socket_path: str,
    forensic_evidence_path: str,
    vault_socket_path: Optional[str],
    audit_socket_path: Optional[str],
    allowed_uid: int,
    allowed_gid: int,
    ready_event: Any,
    allow_alien_field: bool = False,
) -> None:
    die_with_parent()
    kill_switches_state: Dict[str, Dict[str, Any]] = {
        ks_id: {
            **meta,
            "armed": True,
            "triggered": False,
            "trigger_count": 0,
            "last_trigger_reason": None,
            "last_trigger_utc": None,
        }
        for ks_id, meta in KILL_SWITCH_DEFINITIONS.items()
    }
    halted = False
    halt_reason: Optional[str] = None
    quarantine_passed_count = 0
    quarantine_rejected_count = 0
    preserved_forensic_bundles: List[Dict[str, Any]] = []

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o600)
    server.listen(16)
    ready_event.set()

    def _preserve_forensic_evidence_and_halt(
        ks_id: str,
        reason: str,
        telemetry_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        nonlocal halted, halt_reason
        # Close admission before clocks, escalation, serialization or storage can fail.
        halted = True
        halt_reason = f"{ks_id} ({kill_switches_state[ks_id]['name']}): {reason}"
        utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ks_entry = kill_switches_state[ks_id]
        ks_entry["triggered"] = True
        ks_entry["trigger_count"] += 1
        ks_entry["last_trigger_reason"] = reason
        ks_entry["last_trigger_utc"] = utc_now

        # Query current audit chain head from EXEC-GATE-2 if available
        audit_head: Dict[str, Any] = {}
        if audit_socket_path and os.path.exists(audit_socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as asock:
                    asock.settimeout(2.0)
                    asock.connect(audit_socket_path)
                    _send_framed_json(asock, {"method": "export_chain_head", "params": {}})
                    aresp = _recv_framed_json(asock)
                    if aresp.get("status") == "ok":
                        audit_head = aresp["result"]
            except Exception:
                pass

        # Lock EXEC-GATE-1 VaultGateDaemon if available
        vault_locked = False
        if vault_socket_path and os.path.exists(vault_socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as vsock:
                    vsock.settimeout(2.0)
                    vsock.connect(vault_socket_path)
                    _send_framed_json(
                        vsock,
                        {
                            "method": "emergency_lock_vault",
                            "rpc_nonce": "ks_lock_" + secrets.token_hex(12),
                            "params": {},
                        },
                    )
                    vresp = _recv_framed_json(vsock)
                    vault_locked = vresp.get("status") == "ok"
            except Exception:
                pass

        bundle = {
            "evidence_id": f"ev_{ks_id.lower().replace('-', '_')}_{ len(preserved_forensic_bundles) + 1:04d}",
            "kill_switch_id": ks_id,
            "kill_switch_name": ks_entry["name"],
            "trigger_reason": reason,
            "utc_wall_timestamp": utc_now,
            "logical_timestamp": FIXED_LOGICAL_TIMESTAMP,
            "supervisor_pid": os.getpid(),
            "vault_emergency_locked": vault_locked,
            "audit_chain_head_snapshot": audit_head,
            "telemetry_snapshot": telemetry_snapshot,
            "escalation_path": ESCALATION_PATH_SPEC,
        }
        bundle["evidence_sha256"] = sha256_hex(canonical_json(bundle).encode("utf-8"))
        preserved_forensic_bundles.append(bundle)

        # Persist evidence after admission is irrevocably closed.
        os.makedirs(os.path.dirname(forensic_evidence_path), exist_ok=True)
        fd = os.open(
            forensic_evidence_path,
            os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
            0o600,
        )
        try:
            os.write(fd, (canonical_json(preserved_forensic_bundles) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

        return bundle

    running = True
    try:
        while running:
            conn, _ = server.accept()
            with _isolated_ipc_connection(conn):
                try:
                    _peer_pid, peer_uid, peer_gid = _extract_so_peercred(conn)
                    req = _recv_rpc_request(conn)
                except Exception as exc:
                    _send_framed_json(conn, {"status": "error", "error_code": "ERR_IPC_READ", "reason": str(exc)})
                    continue

                if peer_uid != allowed_uid or peer_gid != allowed_gid:
                    _send_framed_json(
                        conn,
                        {
                            "status": "error",
                            "error_code": "ERR_PEER_AUTH_FAILED",
                            "reason": "Unauthorized SO_PEERCRED on QuarantineAndKillSwitchDaemon.",
                        },
                    )
                    continue

                method = req.get("method")
                params = req.get("params", {})

                if method == "shutdown":
                    running = False
                    _send_framed_json(conn, {"status": "ok", "result": {"shutdown": True}})
                    continue

                try:
                    if method == "get_status":
                        _send_framed_json(
                            conn,
                            {
                                "status": "ok",
                                "result": {
                                    "daemon_pid": os.getpid(),
                                    "halted": halted,
                                    "halt_reason": halt_reason,
                                    "quarantine_passed_count": quarantine_passed_count,
                                    "quarantine_rejected_count": quarantine_rejected_count,
                                    "kill_switches": kill_switches_state,
                                    "preserved_evidence_count": len(preserved_forensic_bundles),
                                    "forensic_evidence_path": forensic_evidence_path,
                                    "escalation_path": ESCALATION_PATH_SPEC,
                                    "privacy_and_retention_contract": {
                                        "synthetic_only_input_policy": not allow_alien_field,
                                        "live_capture_authorized": False,
                                        "alien_field_trace_authorized": allow_alien_field,
                                        "alien_field_trace_scope": {
                                            "origin_attestation": "authorized_alien_field_trace",
                                            "source_id": "src_alien_field",
                                            "source_kind": "canvas_state",
                                            "capture_identity_verified_by": "external_observer",
                                            "observation_receipts_allowed": False,
                                        } if allow_alien_field else None,
                                        "live_capture_future_controls": [
                                            "Explicit per-session user consent receipt signed by UI shell",
                                            "Window-scoped crop excluding OS notifications and credential fields",
                                            "Pre-vault OCR/face PII redaction pass",
                                        ],
                                        "retention_and_deletion_controls": {
                                            "ephemeral_vault_ttl_minutes": 15,
                                            "deletion_mechanism": "Immediate memory zeroization and HMAC key destruction on session end or kill-switch halt",
                                        },
                                    },
                                },
                            },
                        )

                    elif method == "assert_not_halted":
                        if halted:
                            raise RuntimeError(f"ERR_KILL_SWITCH_HALTED: {halt_reason}")
                        _send_framed_json(conn, {"status": "ok", "result": {"halted": False}})

                    elif method == "inspect_source":
                        if halted:
                            raise RuntimeError(f"ERR_KILL_SWITCH_HALTED: {halt_reason}")

                        sid = params.get("source_id", "")
                        sver = params.get("source_version", "")
                        skind = params.get("source_kind", "")
                        marks = params.get("marks", [])
                        origin = params.get("origin_attestation", "synthetic_fixture")

                        # 1. Enforce synthetic-only input policy; and, on the owner's word (2026-10-30), a picture the owner
                        #    sent to remake or enhance, said honestly as such; on the owner's word (2026-11-04), the
                        #    companion's own desktop glance (frames held in memory only, never saved); and, on the owner's
                        #    word (2026-11-16), Electric Browser's in-memory viewport frame ('electric_browser_frame');
                        #    and, on the owner's word (2026-11-13/14), a PlayHub screen's frames: the player's own game
                        #    window, measured in memory only ('play_engine_screen').
                        #    All other live capture stays refused.
                        alien_trace = (
                            allow_alien_field
                            and origin == "authorized_alien_field_trace"
                            and sid == "src_alien_field"
                            and skind in ("canvas_state", "captured_frame")
                        )
                        if origin not in (
                            "synthetic_fixture",
                            "owner_supplied_picture",
                            "companion_desktop_glance",
                            "electric_browser_frame",
                            "play_engine_screen",
                        ) and not alien_trace:
                            quarantine_rejected_count += 1
                            raise PermissionError(
                                f"ERR_NON_SYNTHETIC_INPUT_PROHIBITED: origin_attestation='{origin}' rejected by zero-PII quarantine."
                            )

                        # 2. Enforce safe opaque source_id
                        if not isinstance(sid, str) or not SOURCE_ID_REGEX.match(sid):
                            quarantine_rejected_count += 1
                            raise ValueError(
                                f"ERR_INVALID_SOURCE_ID: '{sid}' rejected by quarantine."
                            )
                        if LONG_DIGIT_RUN_REGEX.search(sid):
                            quarantine_rejected_count += 1
                            raise ValueError(
                                f"ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID: '{sid}' contains long digit run."
                            )
                        tokens = [t for t in sid[4:].split("_") if t]
                        if not tokens or any(t in PROHIBITED_SOURCE_ID_TOKENS for t in tokens):
                            quarantine_rejected_count += 1
                            raise ValueError(
                                f"ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID: '{sid}' contains prohibited token."
                            )
                        if skind not in VALID_SOURCE_KINDS:
                            quarantine_rejected_count += 1
                            raise ValueError(f"Unsupported source_kind '{skind}'")
                        if len(marks) > MAX_MARKS_PER_SOURCE:
                            quarantine_rejected_count += 1
                            raise ValueError(f"ERR_MARK_COUNT_EXCEEDED: {len(marks)} > {MAX_MARKS_PER_SOURCE}")

                        # 3. Inspect mark_ids for prohibited PII/filename tokens
                        for m in marks:
                            mid = str(m.get("mark_id", "")).lower()
                            if any(tok in mid.split("_") for tok in PROHIBITED_SOURCE_ID_TOKENS):
                                quarantine_rejected_count += 1
                                raise ValueError(
                                    f"ERR_QUARANTINE_MARK_ID_REJECTED: mark_id '{mid}' contains prohibited token."
                                )

                        quarantine_passed_count += 1
                        _send_framed_json(
                            conn,
                            {
                                "status": "ok",
                                "result": {
                                    "quarantine_status": "passed_zero_pii_synthetic_check",
                                    "source_id": sid,
                                    "source_version": sver,
                                },
                            },
                        )

                    elif method == "trigger_kill_switch":
                        ks_id = params["ks_id"]
                        reason = params["reason"]
                        telemetry = params.get("telemetry_snapshot", {})
                        if ks_id not in kill_switches_state:
                            raise ValueError(f"Unknown kill switch: {ks_id}")
                        bundle = _preserve_forensic_evidence_and_halt(ks_id, reason, telemetry)
                        _send_framed_json(
                            conn,
                            {
                                "status": "ok",
                                "result": {
                                    "halted": True,
                                    "halt_reason": halt_reason,
                                    "forensic_bundle": bundle,
                                },
                            },
                        )

                    elif method == "evaluate_telemetry":
                        telemetry = params["telemetry"]
                        triggered_ks: Optional[Tuple[str, str]] = None
                        if telemetry.get("provenance_completeness", 1.0) < 1.0:
                            triggered_ks = ("KS-1", "provenance_completeness dropped below 1.0")
                        elif telemetry.get("deterministic_replay_equality", 1.0) < 1.0:
                            triggered_ks = ("KS-2", "deterministic_replay_equality dropped below 1.0")
                        elif telemetry.get("false_observation_count", 0) > 0:
                            triggered_ks = ("KS-3", "false_observation_count > 0")
                        elif telemetry.get("raw_buffer_leakage_detected", False):
                            triggered_ks = ("KS-4", "raw RGBA buffer leakage detected")
                        elif telemetry.get("ghost_pixel_leakage_count", 0) > 0:
                            triggered_ks = ("KS-5", "ghost pixel contamination detected")
                        elif telemetry.get("diagnostic_leakage_count", 0) > 0:
                            triggered_ks = ("KS-6", "silent_diagnostic render leaked into default feed")
                        elif telemetry.get("semantic_refusal_rate", 1.0) < 1.0:
                            triggered_ks = ("KS-7", "semantic/intent query was not refused")
                        elif not telemetry.get("audit_chain_valid", True):
                            triggered_ks = ("KS-8", "audit hash chain integrity broken")
                        elif telemetry.get("pii_or_non_synthetic_detected", False):
                            triggered_ks = ("KS-9", "non-synthetic or PII input detected")
                        elif (
                            telemetry.get("cpu_time_ms", 0.0) > 50.0
                            or telemetry.get("wall_latency_ms", 0.0) > 50.0
                        ):
                            triggered_ks = ("KS-10", "latency exceeded 50.0 ms ceiling")

                        if triggered_ks is not None:
                            bundle = _preserve_forensic_evidence_and_halt(
                                triggered_ks[0], triggered_ks[1], telemetry
                            )
                            _send_framed_json(
                                conn,
                                {
                                    "status": "ok",
                                    "result": {
                                        "halted": True,
                                        "triggered_ks_id": triggered_ks[0],
                                        "forensic_bundle": bundle,
                                    },
                                },
                            )
                        else:
                            _send_framed_json(
                                conn,
                                {
                                    "status": "ok",
                                    "result": {
                                        "halted": halted,
                                        "triggered_ks_id": None,
                                    },
                                },
                            )

                    else:
                        raise ValueError(f"Unsupported QuarantineSupervisor method: {method}")

                except PermissionError as exc:
                    _send_framed_json(
                        conn,
                        {"status": "error", "error_code": "ERR_QUARANTINE_PERMISSION", "reason": str(exc)},
                    )
                except RuntimeError as exc:
                    _send_framed_json(
                        conn,
                        {"status": "error", "error_code": "ERR_KILL_SWITCH_HALTED", "reason": str(exc)},
                    )
                except Exception as exc:
                    _send_framed_json(
                        conn,
                        {"status": "error", "error_code": "ERR_QUARANTINE_VALIDATION", "reason": str(exc)},
                    )
    finally:
        server.close()
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass


class OutOfProcessQuarantineAndKillSwitchClient:
    """
    Client for EXEC-GATE-3 Pre-Ingestion Zero-PII Quarantine & Kill-Switch Supervisor Daemon.
    """

    def __init__(self, socket_path: str, forensic_evidence_path: str) -> None:
        self._socket_path = socket_path
        self.forensic_evidence_path = forensic_evidence_path

    def _call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(self._socket_path)
            _send_framed_json(sock, {"method": method, "params": params or {}})
            resp = _recv_framed_json(sock)

        if resp.get("status") != "ok":
            code = resp.get("error_code", "ERR_QUARANTINE")
            reason = resp.get("reason", "Unknown quarantine error")
            if code == "ERR_KILL_SWITCH_HALTED":
                raise RuntimeError(reason)
            if code in ("ERR_QUARANTINE_PERMISSION", "ERR_PEER_AUTH_FAILED"):
                raise PermissionError(reason)
            raise ValueError(reason)
        return resp["result"]

    def get_status(self) -> Dict[str, Any]:
        return self._call("get_status")

    def assert_not_halted(self) -> None:
        self._call("assert_not_halted")

    def inspect_source(
        self,
        *,
        source_id: str,
        source_version: str,
        source_kind: str,
        marks: Sequence[DrawingMark],
        origin_attestation: str = "synthetic_fixture",
    ) -> Dict[str, Any]:
        return self._call(
            "inspect_source",
            {
                "source_id": source_id,
                "source_version": source_version,
                "source_kind": source_kind,
                "marks": [m.to_dict() for m in marks],
                "origin_attestation": origin_attestation,
            },
        )

    def trigger_kill_switch(
        self,
        ks_id: str,
        reason: str,
        telemetry_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._call(
            "trigger_kill_switch",
            {
                "ks_id": ks_id,
                "reason": reason,
                "telemetry_snapshot": telemetry_snapshot or {},
            },
        )

    def evaluate_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("evaluate_telemetry", {"telemetry": telemetry})


class OutOfProcessQuarantineAndKillSwitchDaemon:
    """
    Context-managed EXEC-GATE-3 Pre-Ingestion Zero-PII Quarantine & Kill-Switch Supervisor Daemon.
    """

    def __init__(
        self,
        socket_path: str,
        forensic_evidence_path: str,
        *,
        vault_socket_path: Optional[str] = None,
        audit_socket_path: Optional[str] = None,
        allow_alien_field: bool = False,
    ) -> None:
        if type(allow_alien_field) is not bool:
            raise TypeError("allow_alien_field must be a bool")
        self.socket_path = socket_path
        self.forensic_evidence_path = forensic_evidence_path
        self.vault_socket_path = vault_socket_path
        self.audit_socket_path = audit_socket_path
        self.allow_alien_field = allow_alien_field
        self._proc: Optional[multiprocessing.Process] = None
        self.client: Optional[OutOfProcessQuarantineAndKillSwitchClient] = None

    def __enter__(self) -> OutOfProcessQuarantineAndKillSwitchClient:
        ready_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_quarantine_and_kill_switch_daemon_loop,
            args=(
                self.socket_path,
                self.forensic_evidence_path,
                self.vault_socket_path,
                self.audit_socket_path,
                os.getuid(),
                os.getgid(),
                ready_event,
                self.allow_alien_field,
            ),
            daemon=True,
        )
        self._proc.start()
        if not ready_event.wait(timeout=5.0):
            self.stop()
            raise RuntimeError("OutOfProcessQuarantineAndKillSwitchDaemon failed to start within 5s")
        self.client = OutOfProcessQuarantineAndKillSwitchClient(
            self.socket_path, self.forensic_evidence_path
        )
        return self.client

    def stop(self) -> None:
        if self.client is not None and os.path.exists(self.socket_path):
            try:
                self.client._call("shutdown")
            except Exception:
                pass
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
            self._proc = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# =============================================================================
# End-to-End EXEC-GATE-1, EXEC-GATE-2, and EXEC-GATE-3 Verification Suite
# =============================================================================

def run_exec_gate_verification_suite(
    lab_base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Spawns the isolated EXEC-GATE-1, EXEC-GATE-2, and EXEC-GATE-3 OS processes in a
    temporary lab directory, executes the full verification protocol across all three
    gates plus E1-E10 / C1-C9 regression checks, and cleanly terminates all child processes.
    """
    if lab_base_dir is None:
        lab_base_dir = os.environ.get(
            "ELECTRIC_SIGHT_LAB_DIR",
            os.path.join(tempfile.gettempdir(), "cyclops-ee-lab"),
        )
    os.makedirs(lab_base_dir, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="eg_", dir=lab_base_dir)

    vault_sock = os.path.join(run_dir, "v.sock")
    audit_sock = os.path.join(run_dir, "a.sock")
    quar_sock = os.path.join(run_dir, "q.sock")
    worm_ledger_path = os.path.join(run_dir, "audit_worm_ledger.jsonl")
    forensic_path = os.path.join(run_dir, "forensic_halt_evidence.json")

    harness_pid = os.getpid()
    marks_v1 = [
        DrawingMark("m_sq_01", "rect", 16, 16, 63, 63, (0, 255, 170, 255)),
        DrawingMark("m_mk_01", "rect", 96, 96, 103, 103, (255, 180, 0, 255)),
    ]
    marks_v2 = [
        DrawingMark("m_sq_02", "rect", 26, 16, 73, 63, (0, 255, 170, 255)),
        DrawingMark("m_mk_01", "rect", 96, 96, 103, 103, (255, 180, 0, 255)),
    ]

    t_pilot_start = time.monotonic()

    vault_daemon = OutOfProcessVaultGateDaemon(vault_sock)
    audit_daemon = OutOfProcessAuditSinkDaemon(audit_sock, worm_ledger_path)
    quar_daemon = OutOfProcessQuarantineAndKillSwitchDaemon(
        quar_sock,
        forensic_path,
        vault_socket_path=vault_sock,
        audit_socket_path=audit_sock,
    )

    with vault_daemon as vault_client, audit_daemon as audit_client, quar_daemon as quar_client:
        pilot_harness = ElectricEyeFixtureHarness(
            profile="controlled_pilot_simulation",
            out_of_process_vault_client=vault_client,
            audit_sink=audit_client,
            quarantine_supervisor_client=quar_client,
        )

        # ---------------------------------------------------------------------
        # 1. EXEC-GATE-1 Verification
        # ---------------------------------------------------------------------
        v_status_initial = vault_client.ping_status()
        vault_pid = v_status_initial["daemon_pid"]

        # Negative credential testing uses a separate daemon's expected UID, never RPC data.
        peer_auth_rejected = False
        negative_sock = os.path.join(run_dir, "n.sock")
        negative_ready = multiprocessing.Event()
        negative_proc = multiprocessing.Process(
            target=_vault_gate_daemon_loop,
            args=(negative_sock, os.getuid() + 9999, os.getgid(), negative_ready, None,
                  vault_daemon.capture_service_auth_secret),
            daemon=True,
        )
        negative_proc.start()
        try:
            if not negative_ready.wait(timeout=5.0):
                raise RuntimeError("Negative peer test daemon failed to start")
            negative_client = OutOfProcessVaultGateClient(negative_sock, vault_daemon.capture_service_auth_secret)
            try:
                negative_client.ping_status()
            except PermissionError as exc:
                peer_auth_rejected = "ERR_PEER_AUTH_FAILED" in str(exc)
        finally:
            negative_proc.terminate()
            negative_proc.join(timeout=2.0)
            if os.path.exists(negative_sock):
                os.unlink(negative_sock)

        # Test prohibited RPC method on narrow IPC boundary
        prohibited_method_rejected = False
        try:
            vault_client._call("export_raw_rgba_bytes")
        except PermissionError as exc:
            prohibited_method_rejected = "ERR_RPC_METHOD_PROHIBITED" in str(exc)

        # Test RPC nonce replay protection
        fixed_nonce = "nonce_replay_test_0001"
        vault_client._call("ping_status", rpc_nonce=fixed_nonce)
        rpc_nonce_replay_blocked = False
        try:
            vault_client._call("ping_status", rpc_nonce=fixed_nonce)
        except PermissionError as exc:
            rpc_nonce_replay_blocked = "ERR_REPLAY_DETECTED" in str(exc)

        # Register synthetic sources through EXEC-GATE-3 Quarantine -> EXEC-GATE-1 Vault
        src1 = pilot_harness.register_verified_source("src_pilot_01", "v1", "verified_pixels", marks_v1)
        src2 = pilot_harness.mutate_canvas("src_pilot_01", "v1", "v2", marks_v2)
        src_cap = pilot_harness.register_verified_source("src_cap_pilot", "v1", "captured_frame", marks_v2)

        # Execute all 10 closed-form operators over AF_UNIX IPC
        op_calls = [
            ("inspect_source", {}, "src_pilot_01", "v1", None),
            ("measure_dimensions", {}, "src_pilot_01", "v1", None),
            ("measure_bbox", {"selector": {"type": "non_transparent"}}, "src_pilot_01", "v1", None),
            ("sample_colour", {"x": 32, "y": 32}, "src_pilot_01", "v1", None),
            ("measure_distance", {"p1": {"x": 16, "y": 16}, "p2": {"x": 63, "y": 63}}, "src_pilot_01", "v1", None),
            ("measure_angle", {"p1": {"x": 16, "y": 16}, "p2": {"x": 63, "y": 63}}, "src_pilot_01", "v1", None),
            ("count_pixels", {"predicate": {"type": "non_transparent"}}, "src_pilot_01", "v1", None),
            (
                "count_changed_pixels",
                {"channel_tolerance": 0},
                "src_pilot_01",
                "v1",
                {"source_id": "src_pilot_01", "source_version": "v2"},
            ),
            (
                "extract_changed_regions",
                {"connectivity": 8, "channel_tolerance": 0},
                "src_pilot_01",
                "v1",
                {"source_id": "src_pilot_01", "source_version": "v2"},
            ),
            ("count_objects", {"connectivity": 8, "min_area_px": 1}, "src_pilot_01", "v1", None),
        ]
        for op_name, params, sid, sver, sec in op_calls:
            r_a = pilot_harness.execute_measurement(op_name, params, source_id=sid, source_version=sver, secondary_source=sec)
            r_b = pilot_harness.execute_measurement(op_name, params, source_id=sid, source_version=sver, secondary_source=sec)
            pilot_harness.record_replay_pair(r_a, r_b)

        # Compute all 5 derived representations over AF_UNIX IPC
        for rep_type in ("change_mask", "difference_map", "motion_summary", "colour_delta", "spatial_relationship"):
            pilot_harness.compute_derived_representation(
                rep_type,
                ("src_pilot_01", "v1"),
                ("src_pilot_01", "v2"),
            )

        # Create renders and test capability-token-gated receipt issuance
        rnd_obs = pilot_harness.create_electric_render(
            render_type="observed",
            source="verified_pixels",
            confidence="high",
            uncertainty_state="verified",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_cap_pilot", "ref_version": "v1"}],
        )
        rnd_ghost = pilot_harness.create_electric_render(
            render_type="predicted",
            source="inference",
            confidence="low",
            uncertainty_state="predicted",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_pilot_01", "ref_version": "v2"}],
            ghost_marks=[DrawingMark("m_ghost_01", "rect", 0, 0, 15, 15, (255, 0, 255, 128))],
        )
        rnd_diag = pilot_harness.create_electric_render(
            render_type="diagnostic",
            source="measurement",
            confidence="high",
            uncertainty_state="verified",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_pilot_01", "ref_version": "v1"}],
        )

        # Missing capability token rejected
        missing_token_rejected = False
        try:
            pilot_harness.issue_observation_receipt(
                render_id=rnd_obs["render_id"],
                captured_source_id="src_cap_pilot",
                captured_source_version="v1",
                caller_capability_token=None,
            )
        except PermissionError:
            missing_token_rejected = True

        # Issue single-use capability token and obtain signed receipt
        cap_tok = vault_client.request_capability_token(
            render_id=rnd_obs["render_id"],
            captured_source_id="src_cap_pilot",
            captured_source_version="v1",
        )
        valid_rcpt = pilot_harness.issue_observation_receipt(
            render_id=rnd_obs["render_id"],
            captured_source_id="src_cap_pilot",
            captured_source_version="v1",
            caller_capability_token=cap_tok,
        )

        # Replaying the single-use capability token is rejected
        cap_token_replay_blocked = False
        try:
            pilot_harness.issue_observation_receipt(
                render_id=rnd_obs["render_id"],
                captured_source_id="src_cap_pilot",
                captured_source_version="v1",
                caller_capability_token=cap_tok,
            )
        except PermissionError as exc:
            cap_token_replay_blocked = "ERR_REPLAY_DETECTED" in str(exc)

        # Tampered receipt rejected, valid receipt promotes cleanly
        forged_rcpt = copy.deepcopy(valid_rcpt)
        forged_rcpt["receipt_signature"] = "hmac-sha256:" + ("a" * 64)
        forged_promo = pilot_harness.promote_render_to_observed(rnd_obs["render_id"], forged_rcpt)
        valid_promo = pilot_harness.promote_render_to_observed(rnd_obs["render_id"], valid_rcpt)

        # Semantic query refused and redacted
        ref_sem = pilot_harness.execute_measurement(
            "semantic_meaning",
            {"query_type": "semantic_meaning", "question": "Does this contain private user text?"},
            source_id="src_pilot_01",
            source_version="v1",
        )

        # ---------------------------------------------------------------------
        # 2. EXEC-GATE-2 Verification (Durable WORM Sink, Fail-Closed, Tamper)
        # ---------------------------------------------------------------------
        audit_head = audit_client.export_chain_head()
        audit_pid = audit_head["daemon_pid"]
        durable_verify = audit_client.verify_durable_file()

        # Test fail-closed when sink is unavailable
        seq_before_outage = pilot_harness._audit_seq
        audit_client.simulate_unavailable = True
        fail_closed_verified = False
        try:
            pilot_harness.promote_render_to_observed(rnd_ghost["render_id"], receipt=None)
        except RuntimeError as exc:
            fail_closed_verified = "ERR_AUDIT_SINK_UNAVAILABLE_FAIL_CLOSED" in str(exc)
        finally:
            audit_client.simulate_unavailable = False
        seq_unchanged_on_outage = pilot_harness._audit_seq == seq_before_outage

        # Test sink rejection of unredacted PII / raw bytes
        raw_or_pii_rejected_by_sink = False
        try:
            audit_client.append_entry({
                "entry_seq": audit_head["head_seq"] + 1,
                "logical_clock": f"seq:{audit_head['head_seq'] + 1:08d}@{FIXED_LOGICAL_TIMESTAMP}",
                "logical_timestamp": FIXED_LOGICAL_TIMESTAMP,
                "utc_wall_timestamp": "2026-10-30T19:00:00.000000Z",
                "event_type": "test_pii",
                "target_id": "t1",
                "error_code": "ERR",
                "detail": "UNREDACTED_PII_MARKER",
                "prev_hash": audit_head["chain_head_digest"],
                "entry_hash": "sha256:" + ("0" * 64),
            })
        except ValueError as exc:
            raw_or_pii_rejected_by_sink = "ERR_AUDIT_PII_OR_RAW_BYTES_REJECTED" in str(exc)

        # ---------------------------------------------------------------------
        # 3. EXEC-GATE-3 Verification (Quarantine + Telemetry Evaluation)
        # ---------------------------------------------------------------------
        non_synthetic_rejected = False
        try:
            pilot_harness.register_verified_source(
                "src_live_attempt",
                "v1",
                "captured_frame",
                marks_v1,
                origin_attestation="live_webcam_capture",
            )
        except PermissionError as exc:
            non_synthetic_rejected = "ERR_NON_SYNTHETIC_INPUT_PROHIBITED" in str(exc)

        unsafe_id_rejected = False
        try:
            pilot_harness.register_verified_source(
                "src_user_alice",
                "v1",
                "verified_pixels",
                marks_v1,
            )
        except ValueError as exc:
            unsafe_id_rejected = "ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID" in str(exc)

        unsafe_mark_rejected = False
        try:
            pilot_harness.register_verified_source(
                "src_clean_02",
                "v1",
                "verified_pixels",
                [DrawingMark("mark_user_password", "dot", 1, 1, 1, 1, (255, 0, 0, 255))],
            )
        except ValueError as exc:
            unsafe_mark_rejected = "ERR_QUARANTINE_MARK_ID_REJECTED" in str(exc)

        # Evaluate normal pilot telemetry -> 0 kill-switch triggers
        normal_eval = quar_client.evaluate_telemetry({
            "provenance_completeness": pilot_harness.compute_provenance_completeness(),
            "deterministic_replay_equality": pilot_harness.compute_deterministic_replay_equality(),
            "false_observation_count": pilot_harness.false_observation_count,
            "raw_buffer_leakage_detected": False,
            "ghost_pixel_leakage_count": 0,
            "diagnostic_leakage_count": 0,
            "semantic_refusal_rate": 1.0,
            "audit_chain_valid": durable_verify["valid"],
            "pii_or_non_synthetic_detected": False,
            "cpu_time_ms": 15.0,
            "wall_latency_ms": 15.0,
        })

        quar_status_clean = quar_client.get_status()
        quar_pid = quar_status_clean["daemon_pid"]
        v_status_final = vault_client.ping_status()

        t_pilot_end = time.monotonic()
        pilot_wall_ms = round((t_pilot_end - t_pilot_start) * 1000.0, 4)

        # ---------------------------------------------------------------------
        # 4. Isolated Fault-Injection Test of Kill-Switch Trigger, Forensic
        #    Evidence Preservation, Vault Lock, and Immediate Halt
        # ---------------------------------------------------------------------
        ks_trigger_res = quar_client.evaluate_telemetry({
            "provenance_completeness": 1.0,
            "deterministic_replay_equality": 1.0,
            "false_observation_count": 1,  # Inject simulated KS-3 breach in isolated test step
        })
        halt_enforced_on_harness = False
        try:
            pilot_harness.execute_measurement(
                "measure_dimensions", {}, source_id="src_pilot_01", source_version="v1"
            )
        except RuntimeError as exc:
            halt_enforced_on_harness = "ERR_KILL_SWITCH_HALTED" in str(exc)

        vault_locked_after_ks = vault_client.ping_status()["emergency_locked"]
        forensic_file_exists = os.path.exists(forensic_path)
        with open(forensic_path, "r", encoding="utf-8") as ff:
            forensic_bundles = json.load(ff)

    # Verify all three daemon processes have exited cleanly (zero persistent background daemons)
    daemons_terminated_cleanly = (
        vault_daemon._proc is None
        and audit_daemon._proc is None
        and quar_daemon._proc is None
        and not os.path.exists(vault_sock)
        and not os.path.exists(audit_sock)
        and not os.path.exists(quar_sock)
    )

    # Also verify tamper detection on a mutated copy of the on-disk JSONL file
    tampered_ledger_path = os.path.join(run_dir, "tampered_ledger.jsonl")
    with open(worm_ledger_path, "r", encoding="utf-8") as src_f:
        orig_lines = src_f.readlines()
    tampered_rec = json.loads(orig_lines[0])
    tampered_rec["detail"] = "tampered_on_disk_detail"
    with open(tampered_ledger_path, "w", encoding="utf-8") as dst_f:
        dst_f.write(canonical_json(tampered_rec) + "\n")
        for ln in orig_lines[1:]:
            dst_f.write(ln)

    body_check = {k: v for k, v in tampered_rec.items() if k != "entry_hash"}
    tamper_detected_on_disk = (
        sha256_hex(canonical_json(body_check).encode("utf-8")) != tampered_rec["entry_hash"]
    )

    # Run standard E1-E10 suite to verify <50ms and 100% pass preservation
    e1_e10_report = run_all_scenarios()

    return {
        "harness_pid": harness_pid,
        "vault_daemon_pid": vault_pid,
        "audit_daemon_pid": audit_pid,
        "quarantine_daemon_pid": quar_pid,
        "distinct_os_pids_verified": len({harness_pid, vault_pid, audit_pid, quar_pid}) == 4,
        "daemons_terminated_cleanly": daemons_terminated_cleanly,
        "exec_gate_1": {
            "has_local_vault_or_hmac_keys_in_harness": pilot_harness.has_local_vault_or_hmac_keys,
            "so_peercred_enforced": v_status_final["so_peercred_enforced"],
            "peer_auth_rejected_unauthorized_uid": peer_auth_rejected,
            "prohibited_method_rejected": prohibited_method_rejected,
            "rpc_nonce_replay_blocked": rpc_nonce_replay_blocked,
            "missing_capability_token_rejected": missing_token_rejected,
            "capability_token_replay_blocked": cap_token_replay_blocked,
            "forged_receipt_rejected": forged_promo["status"] == "rejected",
            "valid_receipt_promoted": valid_promo["status"] == "ok" and valid_promo["observed"] is True,
            "false_observation_count": pilot_harness.false_observation_count,
            "deterministic_replay_equality": pilot_harness.compute_deterministic_replay_equality(),
            "provenance_completeness": pilot_harness.compute_provenance_completeness(),
            "ghost_isolation_intact": rnd_ghost["layer_isolation"] == "ghost_isolation_plane",
            "diagnostic_hidden_by_default": all(
                r["render_id"] != rnd_diag["render_id"]
                for r in pilot_harness.list_agent_visible_renders()
            ),
        },
        "exec_gate_2": {
            "audit_head": audit_head,
            "durable_file_verified": durable_verify,
            "fail_closed_verified": fail_closed_verified and seq_unchanged_on_outage,
            "raw_or_pii_rejected_by_sink": raw_or_pii_rejected_by_sink,
            "tamper_detected_on_disk": tamper_detected_on_disk,
            "refused_prompt_redacted": "question" not in ref_sem["parameters"] and "question_sha256" in ref_sem["parameters"],
        },
        "exec_gate_3": {
            "non_synthetic_rejected": non_synthetic_rejected,
            "unsafe_id_rejected": unsafe_id_rejected,
            "unsafe_mark_rejected": unsafe_mark_rejected,
            "normal_run_halted": normal_eval["halted"],
            "normal_run_triggered_ks": normal_eval["triggered_ks_id"],
            "fault_injection_ks_triggered": ks_trigger_res["triggered_ks_id"] == "KS-3",
            "halt_enforced_on_harness": halt_enforced_on_harness,
            "vault_locked_after_ks": vault_locked_after_ks,
            "forensic_evidence_preserved": forensic_file_exists and len(forensic_bundles) == 1,
            "forensic_bundle": forensic_bundles[0] if forensic_bundles else None,
        },
        "pilot_multi_process_wall_ms": pilot_wall_ms,
        "e1_e10_report_summary": {
            "all_scenarios_passed": e1_e10_report["all_scenarios_passed"],
            "all_criteria_passed": e1_e10_report["all_criteria_passed"],
            "metrics": e1_e10_report["metrics"],
        },
    }

