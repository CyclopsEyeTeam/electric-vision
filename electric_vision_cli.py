#!/usr/bin/env python3
"""
electric_vision_cli.py (cyclops-ee-ctl)

Standalone CLI and Terminal Companion Formatter for the Electric Vision for
Agents stack.

Adheres strictly to Prism Terminal Companion & Electric Eye Governance Conventions:
- Supports `--once` for lightweight snapshotting and `--mode focus` for full telemetry.
- Supports `NO_COLOR` environment variable, `--no-color`, and `--plain` ASCII
  rendering without emitting broken ANSI escape sequences.
- Zero raw pixel buffer, cache byte, or cryptographic key leakage.
- Enforces explicit epistemic badges (`[OBS:VERIFIED]`, `[RENDER:DERIVED · UNOBSERVED]`,
  `[GHOST:PREDICTED · UNOBSERVED]`, `[REFUSED:EPISTEMIC_BOUNDARY]`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from electric_eye_harness import (
    UNCERTAINTY_STYLE_SPEC,
    assert_no_raw_buffer_leakage,
    run_all_scenarios,
)
from electric_vision_agent_bridge import (
    SUPPORTED_RESOLUTIONS,
    ElectricVisionAgentSession,
)

PLAIN_ASCII_BORDERS: Dict[str, str] = {
    "verified": "========",
    "derived": "--------",
    "predicted": "........",
    "stale": "^^^^^^^^",
    "unknown": "????????",
}

PLAIN_ASCII_GLYPHS: Dict[str, str] = {
    "verified": "[V]",
    "derived": "[D]",
    "predicted": "[P]",
    "stale": "[S]",
    "unknown": "[?]",
}

ANSI_COLOURS: Dict[str, str] = {
    "verified": "\033[38;2;0;229;255m",
    "derived": "\033[38;2;255;179;0m",
    "predicted": "\033[38;2;213;0;249m",
    "stale": "\033[38;2;255;82;82m",
    "unknown": "\033[38;2;158;158;158m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def _resolve_color_and_plain(
    no_color_flag: bool,
    plain_flag: bool,
) -> tuple[bool, bool]:
    """
    Returns (use_color, use_plain_ascii).
    Honors NO_COLOR env var (any non-empty value disables ANSI color) and --plain.
    """
    env_no_color = bool(os.environ.get("NO_COLOR", "").strip())
    use_plain = bool(plain_flag)
    use_color = not (no_color_flag or env_no_color or use_plain)
    return use_color, use_plain


def _extract_marks_list_from_parsed(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        for key in ("marks", "operations", "objects"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        raise ValueError(
            "Marks JSON dict must contain a 'marks', 'operations' (CYCLOPS_DRAW_WORLD_V3), or 'objects' (drawing.json) array"
        )
    if isinstance(parsed, list):
        return parsed
    raise ValueError("Marks JSON must be a list of mark objects or a dict with 'marks'/'operations'/'objects'")


def _load_marks_arg(spec: str) -> List[Dict[str, Any]]:
    """Loads a JSON array of marks from an inline JSON string or a JSON file path."""
    s = spec.strip()
    if s.startswith("[") or s.startswith("{"):
        parsed = json.loads(s)
    elif os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    else:
        raise ValueError(f"Expected inline JSON or existing file path for marks, got: {spec!r}")
    return _extract_marks_list_from_parsed(parsed)


def _load_stream_frames_arg(spec: str) -> List[Dict[str, Any]]:
    """
    Loads a multi-frame stream specification from inline JSON, a JSON file, or a JSONL file.
    Each frame entry is normalized to `{"version": "vN", "marks": [...]}`.
    """
    s = spec.strip()
    raw_frames: List[Any] = []
    if s.startswith("[") or s.startswith("{"):
        parsed = json.loads(s)
        raw_frames = parsed.get("frames", []) if isinstance(parsed, dict) else parsed
    elif os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("[") or (content.startswith("{") and "\n" not in content):
            parsed = json.loads(content)
            raw_frames = parsed.get("frames", []) if isinstance(parsed, dict) else parsed
        else:
            for line in content.splitlines():
                ln = line.strip()
                if ln:
                    raw_frames.append(json.loads(ln))
    else:
        raise ValueError(f"Expected inline JSON or existing JSON/JSONL file path for stream frames, got: {spec!r}")

    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("Stream frames must be a non-empty list of frame mark sets")

    normalized_frames: List[Dict[str, Any]] = []
    for idx, frm in enumerate(raw_frames):
        ver = f"v{idx + 1}"
        if isinstance(frm, dict) and ("marks" in frm or "operations" in frm or "objects" in frm):
            if "version" in frm:
                ver = str(frm["version"])
            marks_list = _extract_marks_list_from_parsed(frm)
        elif isinstance(frm, list):
            marks_list = frm
        else:
            raise ValueError(f"Invalid frame specification at index {idx}")
        normalized_frames.append({"version": ver, "marks": marks_list})
    return normalized_frames


def format_terminal_banner(
    state: str,
    title: str,
    *,
    observed: bool,
    use_color: bool,
    use_plain: bool,
) -> str:
    """Formats a terminal header banner respecting NO_COLOR and --plain ASCII."""
    st = state if state in UNCERTAINTY_STYLE_SPEC else "unknown"
    obs_tag = "OBS:VERIFIED_RECEIPT" if observed else "UNOBSERVED_RENDER"
    if use_plain:
        border = PLAIN_ASCII_BORDERS[st]
        glyph = PLAIN_ASCII_GLYPHS[st]
        return f"{border} {glyph} [{st.upper()} | {obs_tag}] {title} {border}"

    spec = UNCERTAINTY_STYLE_SPEC[st]
    border = spec["border_pattern"]
    glyph = spec["glyph"]
    line = f"{border} {glyph} [{st.upper()} · {obs_tag}] {title} {border}"
    if use_color:
        col = ANSI_COLOURS.get(st, "")
        rst = ANSI_COLOURS["reset"]
        bld = ANSI_COLOURS["bold"]
        return f"{col}{bld}{line}{rst}"
    return line


def render_inspect_terminal_output(
    report: Dict[str, Any],
    *,
    mode: str = "once",
    use_color: bool = True,
    use_plain: bool = False,
) -> str:
    """Renders an inspect_canvas report for terminal output without any ANSI leakage when disabled."""
    rnd = report["render"]
    state = rnd["uncertainty_state"]
    observed = bool(rnd["observed"])
    res = report["resolution"]
    simd = report["native_simd_metrics"]
    bbox = simd["op_connected_components_4c"]["foreground_bbox"]

    lines: List[str] = []
    lines.append(
        format_terminal_banner(
            state,
            f"Electric Vision Inspect {report['source_id']}@{report['source_version']} ({res['width']}x{res['height']})",
            observed=observed,
            use_color=use_color,
            use_plain=use_plain,
        )
    )
    lines.append(f"  Render ID          : {rnd['render_id']}")
    lines.append(f"  Layer Isolation    : {rnd['layer_isolation']}")
    lines.append(f"  Observed Status    : {observed} (Receipt: {rnd.get('observation_receipt_id') or 'NONE'})")
    lines.append(f"  SIMD Backend       : {report['native_simd_backend']} ({report['elapsed_ms']} ms)")
    lines.append(f"  Foreground BBox    : {bbox}")
    lines.append(f"  Connected Objects  : {simd['op_connected_components_4c']['component_count']} (4-connected)")
    lines.append(f"  WCAG Contrast Q16  : {simd['op_contrast_ratio_wcag_q16']} ({round(simd['op_contrast_ratio_wcag_q16'] / 65536.0, 3)}:1)")
    lines.append(f"  Sobel Edge Q16     : {simd['op_sobel_edge_density_q16']}")
    lines.append(f"  Perceptual dHash64 : {simd['op_perceptual_hash_dhash64']}")
    lines.append(f"  Replay Digest      : {report['native_replay_digest']}")

    if mode == "focus":
        lines.append(f"  Luma 16-Bin Hist   : {simd['op_histogram_luma_16bin']}")
        lines.append(f"  Mean RGBA Q16      : {simd['op_mean_rgba_u8']['mean_rgba_q16']}")
        aria_str = str(report["accessibility"]["aria_label"])
        if use_plain:
            aria_str = aria_str.encode("ascii", errors="replace").decode("ascii")
        lines.append(f"  ARIA Label         : {aria_str}")
        if report.get("hsm_receipt_countersignature"):
            hsm_sig = report["hsm_receipt_countersignature"]
            lines.append(f"  HSM Key ID         : {hsm_sig['key_id']}")
    return "\n".join(lines)


def render_diff_terminal_output(
    report: Dict[str, Any],
    *,
    mode: str = "once",
    use_color: bool = True,
    use_plain: bool = False,
) -> str:
    """Renders a compare_canvases structural diff report for terminal output."""
    diff = report["native_diff_metrics"]
    res = report["resolution"]
    sa = report["source_a"]
    sb = report["source_b"]
    lines: List[str] = []
    lines.append(
        format_terminal_banner(
            "derived",
            f"Electric Vision Diff {sa['source_id']}@{sa['source_version']} -> {sb['source_id']}@{sb['source_version']} ({res['width']}x{res['height']})",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
    )
    lines.append(f"  Epistemic Label    : {report['epistemic_label']} (observed={report['observed']})")
    lines.append(f"  Changed Pixels     : {diff['changed_pixels']} / {diff['total_pixels']} (ratio={diff['change_ratio']})")
    lines.append(f"  Added / Removed    : +{diff['added_pixels']} / -{diff['removed_pixels']} (modified={diff['modified_colour_pixels']})")
    lines.append(f"  Diff BBox          : {diff['diff_bbox']}")
    lines.append(f"  Diff Replay Digest : {report['diff_replay_digest']}")
    if mode == "focus":
        motion = report["derived_representations"]["motion_summary"]["summary"]
        cdelta = report["derived_representations"]["colour_delta"]["summary"]
        spatial = report["derived_representations"]["spatial_relationship"]["summary"]
        lines.append(f"  Motion Trajectories: {len(motion.get('component_trajectories', []))} tracked components")
        lines.append(f"  Colour Delta (Euc) : mean={cdelta.get('mean_rgba_euclidean_delta')}, max={cdelta.get('max_rgba_euclidean_delta')}")
        lines.append(f"  Spatial Topology   : {spatial}")
    return "\n".join(lines)


def render_stream_terminal_output(
    stream_report: Dict[str, Any],
    *,
    mode: str = "once",
    use_color: bool = True,
    use_plain: bool = False,
) -> str:
    """Renders a ContinuousElectricEyeStream report for terminal output."""
    summary = stream_report["final_window_summary"]
    res = stream_report["resolution"]
    det = summary["settling_and_oscillation_detector"]
    dhash_tl = summary["dhash64_hamming_timeline"]
    watchdog = summary["epistemic_watchdog"]
    kin = summary["kinematic_trajectory_chain"]
    dirty_prof = summary["dirty_tile_16x16_profile"]

    banner_state = "verified" if watchdog["fresh_verified_versions"] else "derived"
    observed = bool(watchdog["fresh_verified_versions"])
    lines: List[str] = [
        format_terminal_banner(
            banner_state,
            f"Continuous Electric Eye Stream {stream_report['source_id']} ({res['width']}x{res['height']})",
            observed=observed,
            use_color=use_color,
            use_plain=use_plain,
        ),
        f"  Frames Ingested    : {stream_report['total_frames_ingested']} (Window Active: {summary['active_frame_count']}/{summary['window_capacity']})",
        f"  Evicted & Zeroized : {summary['evicted_frame_count']} frames ({summary['cumulative_zeroized_bytes']} bytes explicit_bzero)",
        f"  Stream Regime      : {det['regime']} (settled={det['settled']}, cycle={det['oscillating_cycle_detected']})",
        f"  dHash64 Hamming    : steps={dhash_tl['step_hamming_distances']} (max={dhash_tl['max_step_hamming']}, checkpoints={dhash_tl['checkpoints_triggered']})",
        f"  Dirty 16x16 Tiles  : mean_ratio={dirty_prof['mean_dirty_tile_ratio']}",
        f"  Fresh Verified     : {watchdog['fresh_verified_versions']}",
        f"  Decayed [STALE]    : {watchdog['decayed_stale_versions']}",
    ]
    if mode == "focus":
        if watchdog["decayed_stale_versions"]:
            stale_banner = format_terminal_banner(
                "stale",
                f"Epistemic Watchdog: {len(watchdog['decayed_stale_versions'])} Prior Observation(s) Superseded -> STALE",
                observed=False,
                use_color=use_color,
                use_plain=use_plain,
            )
            lines.append(stale_banner)
        lines.append(f"  Kinematic Vel      : {kin['velocities']}")
        lines.append(f"  Kinematic Accel    : {kin['accelerations']}")
        lines.append(f"  Next Ghost Centroid: {kin['predicted_next_centroid']}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyclops-ee-ctl",
        description="Electric Vision for Agents CLI & Terminal Companion (Zero-PII Synthetic Enclave)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color codes")
    parser.add_argument("--plain", action="store_true", help="Use plain 7-bit ASCII borders and glyphs")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON output")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single lightweight snapshot and exit immediately (default behavior)",
    )
    parser.add_argument(
        "--mode",
        choices=("once", "focus"),
        default="once",
        help="Telemetry verbosity mode ('once' for compact snapshot, 'focus' for full telemetry)",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # 1. status
    subparsers.add_parser("status", help="Snapshot Electric Vision enclave & SIMD engine status")

    # 2. selftest
    subparsers.add_parser("selftest", help="Run E1-E10, C1-C9, and multi-resolution agent bridge self-test")

    # 3. inspect-marks
    p_inspect = subparsers.add_parser(
        "inspect-marks",
        help="Ingest synthetic marks (cyclopsdraw / Prism / DrawingMark) and run closed-form + SIMD inspection",
    )
    p_inspect.add_argument("--marks", required=True, help="Inline JSON array of marks or path to JSON file")
    p_inspect.add_argument("--width", type=int, default=256, help="Canvas width (128, 256, 512, or 1920)")
    p_inspect.add_argument("--height", type=int, default=256, help="Canvas height (128, 256, 512, or 1080)")
    p_inspect.add_argument("--mirror", choices=("none", "x", "y", "xy"), default="none", help="Prism symmetry reflection")
    p_inspect.add_argument(
        "--source-kind",
        choices=("drawing_state", "canvas_state", "verified_pixels", "captured_frame"),
        default="captured_frame",
        help="Source kind registration",
    )
    p_inspect.add_argument(
        "--issue-receipt",
        action="store_true",
        help="Mint capability token and issue HSM-countersigned ObservationReceipt",
    )

    # 4. diff-marks
    p_diff = subparsers.add_parser(
        "diff-marks",
        help="Compute multi-resolution structural diff and Layer-3 derived representations between two mark sets",
    )
    p_diff.add_argument("--marks-a", required=True, help="Before marks (inline JSON or file path)")
    p_diff.add_argument("--marks-b", required=True, help="After marks (inline JSON or file path)")
    p_diff.add_argument("--width", type=int, default=256, help="Canvas width (128, 256, 512, or 1920)")
    p_diff.add_argument("--height", type=int, default=256, help="Canvas height (128, 256, 512, or 1080)")

    # 5. preview-ghost
    p_ghost = subparsers.add_parser(
        "preview-ghost",
        help="Project proposed marks onto ghost_isolation_plane without mutating verified_plane",
    )
    p_ghost.add_argument("--base-marks", required=True, help="Verified base marks (inline JSON or file path)")
    p_ghost.add_argument("--ghost-marks", required=True, help="Proposed ghost marks (inline JSON or file path)")
    p_ghost.add_argument("--width", type=int, default=256, help="Canvas width")
    p_ghost.add_argument("--height", type=int, default=256, help="Canvas height")

    # 6. refusal-check
    p_refuse = subparsers.add_parser(
        "refusal-check",
        help="Verify deterministic epistemic refusal on semantic, aesthetic, or intent queries",
    )
    p_refuse.add_argument("--query", required=True, help="Semantic/aesthetic question to test")
    p_refuse.add_argument(
        "--query-type",
        default="aesthetic_judgment",
        help="Operator or semantic query type",
    )

    # 7. watch-stream (Continuous Electric Eye)
    p_stream = subparsers.add_parser(
        "watch-stream",
        help="Run Continuous Electric Eye over a multi-frame stream (ring buffer, dirty tiles, stale decay, kinematics)",
    )
    p_stream.add_argument(
        "--frames",
        required=True,
        help="Inline JSON array of frames or path to JSON/JSONL stream file",
    )
    p_stream.add_argument("--width", type=int, default=256, help="Canvas width (128, 256, 512, or 1920)")
    p_stream.add_argument("--height", type=int, default=256, help="Canvas height (128, 256, 512, or 1080)")
    p_stream.add_argument("--window-size", type=int, default=8, help="Bounded ring-buffer frame capacity (2..64)")
    p_stream.add_argument("--settle-ticks", type=int, default=2, help="Consecutive zero-delta ticks to mark settled")
    p_stream.add_argument(
        "--issue-receipts",
        action="store_true",
        help="Issue ObservationReceipts on ingested frames (demonstrating automatic verified -> stale decay)",
    )

    return parser


def run_cli(argv: Optional[Sequence[str]] = None) -> tuple[int, str]:
    """
    Executes the CLI with `argv` and returns `(exit_code, formatted_output)`.
    Safe to call directly from unit tests or `__main__`.
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    use_color, use_plain = _resolve_color_and_plain(args.no_color, args.plain)
    subcmd = args.subcommand or "status"

    if subcmd == "status":
        with ElectricVisionAgentSession(default_resolution=(256, 256)) as session:
            health = session.get_enclave_health_and_audit_summary()
        payload = {
            "command": "status",
            "mode": args.mode,
            "supported_resolutions": [f"{w}x{h}" for w, h in SUPPORTED_RESOLUTIONS],
            "enclave_health": health,
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return 0, json.dumps(payload, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "verified",
            "Electric Vision Enclave Status (Non-Live Synthetic Mode)",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  SIMD Backend       : {health['native_simd_backend']}",
            f"  HSM Key ID         : {health['hsm_enclave'].get('active_key_id')}",
            f"  Vault Sources      : {health['vault_enclave'].get('registered_source_count', 0)}",
            f"  Kill Switches Trig : {health['quarantine_supervisor']['triggered_kill_switches']}",
            f"  Resolutions        : {', '.join(payload['supported_resolutions'])}",
        ]
        if args.mode == "focus":
            lines.append(f"  Audit Chain Head   : {health['audit_worm_ledger'].get('chain_head_digest')}")
            lines.append(f"  HSM OS Hardening   : {health['hsm_enclave'].get('os_hardening')}")
        return 0, "\n".join(lines)

    if subcmd == "selftest":
        e_rep = run_all_scenarios()
        with ElectricVisionAgentSession(default_resolution=(256, 256)) as session:
            session.register_synthetic_canvas(
                "src_cli_selftest",
                "v1",
                [{"k": "dot", "x": 0.5, "y": 0.5, "r": 0.1, "c": [0, 229, 255], "a": 1.0}],
                width=256,
                height=256,
                source_kind="captured_frame",
            )
            insp = session.inspect_canvas("src_cli_selftest", "v1", issue_observation_receipt=True)
            health = session.get_enclave_health_and_audit_summary()

        all_ok = bool(
            e_rep["all_scenarios_passed"]
            and e_rep["all_criteria_passed"]
            and insp["render"]["observed"] is True
            and health["quarantine_supervisor"]["triggered_kill_switches"] == 0
        )
        payload = {
            "command": "selftest",
            "all_passed": all_ok,
            "e1_e10_passed": e_rep["all_scenarios_passed"],
            "c1_c9_passed": e_rep["all_criteria_passed"],
            "agent_bridge_256x256_verified": insp["render"]["observed"],
            "replay_digest": insp["native_replay_digest"],
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return (0 if all_ok else 1), json.dumps(payload, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "verified" if all_ok else "stale",
            f"Electric Vision Self-Test: {'PASS' if all_ok else 'FAIL'}",
            observed=True,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  E1-E10 Scenarios   : {'PASS' if e_rep['all_scenarios_passed'] else 'FAIL'}",
            f"  C1-C9 Criteria     : {'PASS' if e_rep['all_criteria_passed'] else 'FAIL'}",
            f"  256x256 SIMD Bridge: {'PASS' if insp['render']['observed'] else 'FAIL'}",
            f"  Replay Digest      : {insp['native_replay_digest']}",
        ]
        return (0 if all_ok else 1), "\n".join(lines)

    if subcmd == "inspect-marks":
        marks = _load_marks_arg(args.marks)
        with ElectricVisionAgentSession(default_resolution=(args.width, args.height)) as session:
            session.register_synthetic_canvas(
                "src_cli_inspect",
                "v1",
                marks,
                width=args.width,
                height=args.height,
                source_kind=args.source_kind,
                mirror=args.mirror,
            )
            rep = session.inspect_canvas(
                "src_cli_inspect",
                "v1",
                issue_observation_receipt=args.issue_receipt,
            )
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        return 0, render_inspect_terminal_output(
            rep, mode=args.mode, use_color=use_color, use_plain=use_plain
        )

    if subcmd == "diff-marks":
        marks_a = _load_marks_arg(args.marks_a)
        marks_b = _load_marks_arg(args.marks_b)
        with ElectricVisionAgentSession(default_resolution=(args.width, args.height)) as session:
            session.register_synthetic_canvas(
                "src_cli_diff", "v1", marks_a, width=args.width, height=args.height
            )
            session.register_synthetic_canvas(
                "src_cli_diff", "v2", marks_b, width=args.width, height=args.height
            )
            rep = session.compare_canvases(("src_cli_diff", "v1"), ("src_cli_diff", "v2"))
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        return 0, render_diff_terminal_output(
            rep, mode=args.mode, use_color=use_color, use_plain=use_plain
        )

    if subcmd == "preview-ghost":
        base_marks = _load_marks_arg(args.base_marks)
        ghost_marks = _load_marks_arg(args.ghost_marks)
        with ElectricVisionAgentSession(default_resolution=(args.width, args.height)) as session:
            session.register_synthetic_canvas(
                "src_cli_ghost", "v1", base_marks, width=args.width, height=args.height
            )
            rep = session.preview_ghost_marks("src_cli_ghost", "v1", ghost_marks)
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "predicted",
            f"GhostOverlay Projection on src_cli_ghost@v1 ({args.width}x{args.height})",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        hm = rep["hypothetical_metrics"]
        lines = [
            banner,
            f"  Layer Isolation    : {rep['layer_isolation']}",
            f"  Observed Status    : {rep['observed']} (Ghost Leakage: {rep['ghost_pixel_leakage_bytes']} bytes)",
            f"  Base SHA Preserved : {rep['base_pixel_sha256_preserved']}",
            f"  Hypothetical BBox  : {hm['op_connected_components_4c']['foreground_bbox']}",
            f"  Hypothetical Diff  : {hm['op_bbox_aligned_diff_u8']['changed_pixels']} px changed vs base",
        ]
        return 0, "\n".join(lines)

    if subcmd == "refusal-check":
        with ElectricVisionAgentSession(default_resolution=(128, 128)) as session:
            session.register_synthetic_canvas(
                "src_cli_refuse",
                "v1",
                [{"shape": "rect", "x": 10, "y": 10, "width": 20, "height": 20, "color": "#00e5ff"}],
                width=128,
                height=128,
            )
            rep = session.evaluate_agent_query(
                args.query_type,
                {"query_type": args.query_type, "question": args.query},
                source_id="src_cli_refuse",
                source_version="v1",
            )
        assert_no_raw_buffer_leakage(rep)
        if args.json:
            return 0, json.dumps(rep, indent=2, sort_keys=True)
        banner = format_terminal_banner(
            "unknown",
            f"Semantic Refusal Gate: {rep['status'].upper()}",
            observed=False,
            use_color=use_color,
            use_plain=use_plain,
        )
        lines = [
            banner,
            f"  Refusal Code       : {rep.get('refusal_code')}",
            f"  Refusal Reason     : {rep.get('refusal_reason')}",
            f"  Result             : {rep.get('result')}",
        ]
        return 0, "\n".join(lines)

    if subcmd == "watch-stream":
        frames = _load_stream_frames_arg(args.frames)
        tick_reports: List[Dict[str, Any]] = []
        with ElectricVisionAgentSession(default_resolution=(args.width, args.height)) as session:
            stream = session.create_continuous_stream(
                "src_cli_stream",
                width=args.width,
                height=args.height,
                max_window_frames=args.window_size,
                settle_ticks=args.settle_ticks,
            )
            for frm in frames:
                tick_rep = stream.ingest_frame(
                    frm["version"],
                    frm["marks"],
                    issue_observation_receipt=args.issue_receipts,
                )
                tick_reports.append(tick_rep)
            final_summary = stream.compute_temporal_window_summary()

        payload = {
            "command": "watch-stream",
            "source_id": "src_cli_stream",
            "resolution": {"width": args.width, "height": args.height},
            "total_frames_ingested": len(frames),
            "ticks": tick_reports,
            "final_window_summary": final_summary,
        }
        assert_no_raw_buffer_leakage(payload)
        if args.json:
            return 0, json.dumps(payload, indent=2, sort_keys=True)
        return 0, render_stream_terminal_output(
            payload, mode=args.mode, use_color=use_color, use_plain=use_plain
        )

    return 1, "Unknown subcommand"


if __name__ == "__main__":
    code, output = run_cli()
    print(output)
    sys.exit(code)
