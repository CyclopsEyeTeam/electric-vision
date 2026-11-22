"""Deterministic static-review regressions; all execution boundaries use fakes."""
from contextlib import ExitStack
import copy
import ctypes
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Importing these modules must not initialize a native engine or launch a helper.
with patch.object(ctypes, "CDLL", side_effect=AssertionError("native execution prohibited")), \
     patch.object(socket, "socket", side_effect=AssertionError("socket execution prohibited")), \
     patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess execution prohibited")):
    import exec_gate_daemons as gates
    import electric_vision_agent_bridge as bridge
    import electric_vision_mcp as mcp
    import visual_sources as sources


class FakeOnlyTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module, name in ((ctypes, "CDLL"), (socket, "socket"), (subprocess, "Popen"),
                             (sources, "_chrome_screenshot"), (sources, "render_svg"),
                             (mcp, "_render_specs_to_png"), (mcp, "_render_temporal_contact_sheet_png"),
                             (mcp, "_rasterize_normalized_specs"), (bridge, "_rasterize_normalized_specs"),
                             (bridge.multiprocessing, "get_context")):
            self.stack.enter_context(patch.object(module, name, side_effect=AssertionError(f"real {name} prohibited")))


class Connection:
    def __init__(self, request, uid=42, gid=43):
        self.request = request
        self.credentials = (101, uid, gid)
        self.responses = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def settimeout(self, timeout):
        pass

    def getsockopt(self, level, option, size):
        assert (level, option, size) == (socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.pack("3i", *self.credentials)


class GateTests(FakeOnlyTest):
    def fake_server(self, connections):
        server = Mock()
        server.accept.side_effect = [(c, None) for c in connections]
        self.stack.enter_context(patch.object(gates.socket, "socket", return_value=server))
        self.stack.enter_context(patch.object(gates, "die_with_parent"))
        self.stack.enter_context(patch.object(gates.os.path, "exists", return_value=False))
        self.stack.enter_context(patch.object(gates.os, "chmod"))
        self.stack.enter_context(patch.object(gates, "_recv_rpc_request", side_effect=lambda c: c.request))
        self.stack.enter_context(patch.object(gates, "_send_framed_json", side_effect=lambda c, p: c.responses.append(p)))
        return server

    def test_vault_auth_uses_only_kernel_credentials(self):
        authorized = Connection({"method": "ping_status", "rpc_nonce": "nonce-001", "params": {},
                                 "_simulate_peer_uid": "malformed", "_simulate_peer_gid": -1})
        bad_uid = Connection({"method": "shutdown", "_simulate_peer_uid": 42, "_simulate_peer_gid": 43}, uid=99)
        bad_gid = Connection({"method": "shutdown", "_simulate_peer_uid": 42, "_simulate_peer_gid": 43}, gid=99)
        stop = Connection({"method": "shutdown"})
        server = self.fake_server([authorized, bad_uid, bad_gid, stop])
        self.stack.enter_context(patch.object(gates, "_create_isolated_vault_and_gate", return_value=tuple(Mock(return_value=0) for _ in range(9))))
        gates._vault_gate_daemon_loop("/fake/v", 42, 43, Mock(), b"k" * 32, "secret")
        status = authorized.responses[0]
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["result"]["last_peer_cred"], {"pid": 101, "uid": 42, "gid": 43})
        for c in (bad_uid, bad_gid):
            self.assertEqual(c.responses[0]["error_code"], "ERR_PEER_AUTH_FAILED")
        self.assertEqual(stop.responses[0]["status"], "ok")
        server.close.assert_called_once()

    def test_eviction_revokes_receipts_tokens_and_keeps_replay_tombstones(self):
        for fail_cleanup in (False, True):
            with self.subTest(fail_cleanup=fail_cleanup), ExitStack() as local:
                sid, ver, rid = "src_review", "v1", "render-one"
                render = {"render_id": rid, "render_type": "observed",
                          "provenance_refs": [{"ref_type": "source", "ref_id": sid, "ref_version": ver}]}
                receipt = {"observation_receipt_id": "receipt-one", "render_id": rid,
                           "captured_source_id": sid, "captured_source_version": ver,
                           "pipeline_stage": "verified_observation_window", "receipt_signature": "signature"}
                requests = []
                def rpc(method, params=None, nonce=None):
                    c = Connection({"method": method, "params": params or {},
                                    "rpc_nonce": nonce or f"nonce-{len(requests):04d}"})
                    requests.append(c)
                    return c
                source = {"source_id": sid, "source_version": ver, "source_kind": "captured_frame", "marks": []}
                cap = {"render_id": rid, "captured_source_id": sid, "captured_source_version": ver,
                       "capture_service_auth_secret": "secret"}
                def issue(token):
                    return {"caller_capability_token": token, "observation_receipt_id": "receipt-one",
                            "render": render, "captured_source_id": sid, "captured_source_version": ver}
                rpc("store_verified_source", source, "nonce-original")
                rpc("issue_capability_token", cap)
                rpc("issue_observation_receipt", issue("cap_tok_a"))
                rpc("issue_capability_token", cap)
                before = rpc("verify_observation_receipt", {"receipt": receipt})
                eviction = rpc("evict_sources", {"sources": [[sid, ver]]})
                after = rpc("verify_observation_receipt", {"receipt": receipt})
                blocked_cap = rpc("issue_capability_token", cap)
                blocked_unused = rpc("issue_observation_receipt", issue("cap_tok_b"))
                blocked_consumed = rpc("issue_observation_receipt", issue("cap_tok_a"))
                blocked_nonce = rpc("store_verified_source", source, "nonce-original")
                rpc("store_verified_source", source)
                rpc("issue_capability_token", cap)
                blocked_receipt_id = rpc("issue_observation_receipt", issue("cap_tok_c"))
                restored = rpc("verify_observation_receipt", {"receipt": receipt})
                rpc("shutdown")
                self.fake_server(requests)
                functions = [Mock(return_value="digest"), Mock(), Mock(), Mock(), Mock(return_value=0),
                             Mock(return_value="signature"), Mock(return_value=True), Mock(return_value=0),
                             Mock(return_value=1, side_effect=RuntimeError("cleanup failed") if fail_cleanup else None)]
                local.enter_context(patch.object(gates, "_create_isolated_vault_and_gate", return_value=tuple(functions)))
                local.enter_context(patch.object(gates.secrets, "token_hex", side_effect=["a", "b", "c"]))
                gates._vault_gate_daemon_loop("/fake/v", 42, 43, Mock(), b"k" * 32, "secret")
                self.assertTrue(before.responses[0]["result"]["valid"])
                self.assertEqual(eviction.responses[0]["status"], "error" if fail_cleanup else "ok")
                self.assertFalse(after.responses[0]["result"]["valid"])
                self.assertFalse(restored.responses[0]["result"]["valid"])
                for blocked in (blocked_cap, blocked_unused, blocked_consumed, blocked_nonce, blocked_receipt_id):
                    self.assertEqual(blocked.responses[0]["status"], "error")
                self.assertIn("REPLAY", blocked_unused.responses[0]["reason"])
                self.assertIn("DUPLICATE", blocked_receipt_id.responses[0]["reason"])

    def test_client_has_no_payload_credential_injection(self):
        with self.assertRaises(TypeError):
            gates.OutOfProcessVaultGateClient("/fake/v", "secret")._call("ping_status", simulate_peer_uid=42)

    def test_halt_latches_before_every_storage_failure(self):
        for operation in ("makedirs", "open", "write", "fsync"):
            for method in ("trigger_kill_switch", "evaluate_telemetry"):
                with self.subTest(operation=operation, method=method), ExitStack() as local:
                    params = ({"ks_id": "KS-8", "reason": "storage failure"} if method == "trigger_kill_switch"
                              else {"telemetry": {"audit_chain_valid": False}})
                    requests = [Connection({"method": method, "params": params}),
                                Connection({"method": "get_status"}),
                                Connection({"method": "inspect_source", "params": {}}),
                                Connection({"method": "assert_not_halted"}),
                                Connection({"method": "shutdown"})]
                    self.fake_server(requests)
                    for name, value in (("makedirs", None), ("open", 999), ("write", 1), ("fsync", None), ("close", None)):
                        local.enter_context(patch.object(gates.os, name, return_value=value,
                                                        side_effect=OSError("synthetic storage failure") if name == operation else None))
                    gates._quarantine_and_kill_switch_daemon_loop("/fake/q", "/fake/evidence", None, None, 42, 43, Mock())
                    self.assertEqual(requests[0].responses[0]["status"], "error")
                    status = requests[1].responses[0]["result"]
                    self.assertTrue(status["halted"])
                    self.assertIn("KS-8", status["halt_reason"])
                    self.assertEqual(status["quarantine_passed_count"], 0)
                    for c in requests[2:4]:
                        self.assertEqual(c.responses[0]["error_code"], "ERR_KILL_SWITCH_HALTED")

    def test_halt_latches_before_clock_or_serialization_failure(self):
        for stage in ("clock", "serialization"):
            with self.subTest(stage=stage), ExitStack() as local:
                requests = [Connection({"method": "trigger_kill_switch", "params": {"ks_id": "KS-8", "reason": "failure"}}),
                            Connection({"method": "get_status"}), Connection({"method": "shutdown"})]
                self.fake_server(requests)
                if stage == "clock":
                    local.enter_context(patch.object(gates, "datetime", Mock(now=Mock(side_effect=RuntimeError("clock failure")))))
                else:
                    local.enter_context(patch.object(gates, "canonical_json", side_effect=ValueError("serialization failure")))
                gates._quarantine_and_kill_switch_daemon_loop("/fake/q", "/fake/evidence", None, None, 42, 43, Mock())
                self.assertEqual(requests[0].responses[0]["status"], "error")
                status = requests[1].responses[0]["result"]
                self.assertTrue(status["halted"])
                self.assertIn("KS-8", status["halt_reason"])

    def test_quarantine_allowlist_accepts_electric_browser_frame_and_preserves_eviction_and_refusals(self):
        allowed = (
            "synthetic_fixture",
            "owner_supplied_picture",
            "companion_desktop_glance",
            "electric_browser_frame",
        )
        refused_live = (
            "live_webcam_capture",
            "live_screen_capture",
            "screen_capture",
            "anything_else",
        )
        bad_ids = (
            ("bad-id", "ERR_INVALID_SOURCE_ID"),
            ("src_UpperCase", "ERR_INVALID_SOURCE_ID"),
            ("src_eb_123456", "ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID"),
            ("src_user_frame", "ERR_UNSAFE_SEMANTIC_OR_PII_SOURCE_ID"),
        )
        requests = []
        for idx, origin in enumerate(allowed, 1):
            requests.append(Connection({
                "method": "inspect_source",
                "params": {
                    "source_id": "src_eb_frame",
                    "source_version": f"v{idx}",
                    "source_kind": "captured_frame",
                    "marks": [{"mark_id": "m1"}],
                    "origin_attestation": origin,
                },
            }))
        for idx, origin in enumerate(refused_live, 10):
            requests.append(Connection({
                "method": "inspect_source",
                "params": {
                    "source_id": "src_eb_frame",
                    "source_version": f"v{idx}",
                    "source_kind": "captured_frame",
                    "marks": [{"mark_id": "m1"}],
                    "origin_attestation": origin,
                },
            }))
        for idx, (sid, _expected_err) in enumerate(bad_ids, 20):
            requests.append(Connection({
                "method": "inspect_source",
                "params": {
                    "source_id": sid,
                    "source_version": f"v{idx}",
                    "source_kind": "captured_frame",
                    "marks": [{"mark_id": "m1"}],
                    "origin_attestation": "electric_browser_frame",
                },
            }))
        requests.append(Connection({
            "method": "inspect_source",
            "params": {
                "source_id": "src_eb_frame",
                "source_version": "v30",
                "source_kind": "captured_frame",
                "marks": [{"mark_id": "user_avatar"}],
                "origin_attestation": "electric_browser_frame",
            },
        }))
        requests.append(Connection({"method": "get_status"}))
        requests.append(Connection({"method": "shutdown"}))
        self.fake_server(requests)
        gates._quarantine_and_kill_switch_daemon_loop("/fake/q", "/fake/evidence", None, None, 42, 43, Mock())

        for idx, origin in enumerate(allowed):
            resp = requests[idx].responses[0]
            self.assertEqual(resp["status"], "ok", origin)
            self.assertEqual(resp["result"]["quarantine_status"], "passed_zero_pii_synthetic_check")
        offset = len(allowed)
        for idx, origin in enumerate(refused_live):
            resp = requests[offset + idx].responses[0]
            self.assertEqual(resp["status"], "error", origin)
            self.assertIn("ERR_NON_SYNTHETIC_INPUT_PROHIBITED", resp["reason"])
        offset += len(refused_live)
        for idx, (_sid, expected_err) in enumerate(bad_ids):
            resp = requests[offset + idx].responses[0]
            self.assertEqual(resp["status"], "error", expected_err)
            self.assertIn(expected_err, resp["reason"])
        pii_mark_resp = requests[offset + len(bad_ids)].responses[0]
        self.assertEqual(pii_mark_resp["status"], "error")
        self.assertIn("ERR_QUARANTINE_MARK_ID_REJECTED", pii_mark_resp["reason"])

        status = requests[-2].responses[0]["result"]
        self.assertEqual(status["quarantine_passed_count"], len(allowed))
        self.assertEqual(status["quarantine_rejected_count"], len(refused_live) + len(bad_ids) + 1)

        vault = bridge._create_multi_res_closure_vault(Mock())
        vault["register_rgba"]("src_eb_frame", "v1", 128, 128, bytes([12, 34, 56, 255]) * (128 * 128))
        evicted = vault["evict_source"](("src_eb_frame", "v1"))
        self.assertTrue(evicted["evicted"])
        self.assertEqual(evicted["zeroized_bytes"], 128 * 128 * 4)
        sample_buf = bytearray(b"electric-browser-frame-bytes")
        wiped = bridge.explicit_zeroize_bytearray(sample_buf)
        self.assertEqual(wiped, len(sample_buf))
        self.assertEqual(sample_buf, bytearray(len(sample_buf)))


class FakeSession:
    def __init__(self):
        self._source_dimensions = {}
        self.native = set()
        self.proxy = set()
        self._harness = SimpleNamespace(_sources={})
        self.failure = None
        self.changed = True
        self.events = []
        self._multi_res_vault = {"compute_dirty_tiles_16x16": self.dirty}

    def fail(self, stage):
        if self.failure == stage:
            raise RuntimeError(f"synthetic {stage} failure")

    def register_synthetic_canvas(self, sid, version, marks, *, width, height, **kwargs):
        self.fail("register_before")
        key = (sid, version)
        self.native.add(key)
        self.fail("register_native")
        self.proxy.add(key)
        self._harness._sources[key] = True
        self._source_dimensions[key] = (width, height)
        self.fail("register_proxy")
        return {"native_pixel_sha256": version if self.changed else "same"}

    def register_pixel_canvas(self, sid, version, rgba, width, height, **kwargs):
        return self.register_synthetic_canvas(sid, version, [], width=width, height=height, **kwargs)

    def inspect_canvas(self, sid, version, issue_observation_receipt=False):
        self.fail("inspect")
        return {"render": {"observed": issue_observation_receipt, "observation_receipt_id": "receipt-" + version},
                "native_simd_metrics": {"op_perceptual_hash_dhash64": version[1:] if self.changed else "0",
                                        "op_connected_components_4c": {"components": []}}}

    def compare_canvases(self, a, b):
        self.fail("compare")
        out = {"native_diff_metrics": {"changed_pixels": int(self.changed), "change_ratio": float(self.changed), "diff_bbox": {}},
               "derived_representations": {"motion_summary": {"summary": {}}}, "diff_replay_digest": "digest"}
        if self.failure == "transition":
            del out["diff_replay_digest"]
        return out

    def dirty(self, a, b):
        self.fail("dirty")
        return {"dirty_tile_ratio": float(self.changed)}

    def _require_active(self):
        return self

    def _append_audit_event(self, **event):
        self.fail("audit")
        self.events.append(event)

    def evict_synthetic_canvas(self, sid, version):
        key = (sid, version)
        evicted = key in self.native
        self.native.discard(key)
        self._source_dimensions.pop(key, None)
        self.fail("evict")
        self.proxy.discard(key)
        return {"evicted": evicted}


class ReceiptSession(FakeSession):
    def __init__(self):
        super().__init__()
        harness = object.__new__(bridge.ElectricEyeFixtureHarness)
        harness._sources = {}
        harness._renders = {}
        harness._observation_receipts = {}
        harness._assert_supervisor_not_halted = Mock()
        harness._append_audit_event = self._append_audit_event
        harness.blocked_observation_attempts = 0
        self.gate_sources = set()
        self.gate_receipts = {}
        harness._out_of_process_vault_client = SimpleNamespace(
            rpc_verify_observation_receipt=lambda r: (
                (r["captured_source_id"], r["captured_source_version"]) in self.gate_sources
                and self.gate_receipts.get(r["observation_receipt_id"]) == r
            )
        )
        harness.release_source_pixels = Mock(side_effect=self.release_proxy)
        self._harness = harness
        self._multi_res_vault["evict_source"] = self.release_native
        self.evict_synthetic_canvas = bridge.ElectricVisionAgentSession.evict_synthetic_canvas.__get__(self)

    def _require_active(self):
        return self._harness

    def register_synthetic_canvas(self, sid, version, marks, *, width, height, **kwargs):
        result = super().register_synthetic_canvas(sid, version, marks, width=width, height=height, **kwargs)
        self._harness._sources[(sid, version)] = SimpleNamespace(source_kind="captured_frame")
        self.gate_sources.add((sid, version))
        return result

    def inspect_canvas(self, sid, version, issue_observation_receipt=False):
        result = super().inspect_canvas(sid, version, issue_observation_receipt)
        rid, rcpt_id = "render-" + version, "receipt-" + version
        render = {"render_id": rid, "render_type": "observed", "observed": issue_observation_receipt,
                  "uncertainty_state": "verified", "observation_receipt_id": rcpt_id,
                  "provenance_refs": [{"ref_type": "source", "ref_id": sid, "ref_version": version}]}
        self._harness._renders[rid] = render
        if issue_observation_receipt:
            receipt = {"observation_receipt_id": rcpt_id, "render_id": rid,
                       "captured_source_id": sid, "captured_source_version": version,
                       "pipeline_stage": "verified_observation_window", "receipt_signature": "signature-" + version}
            self._harness._observation_receipts[rcpt_id] = copy.deepcopy(receipt)
            self.gate_receipts[rcpt_id] = copy.deepcopy(receipt)
        result["render"] = copy.deepcopy(render)
        return result

    def dirty(self, a, b):
        return {**super().dirty(a, b), "dirty_tile_count": int(self.changed)}

    def release_proxy(self, keys):
        for key in keys:
            self.gate_sources.discard(key)
            for rid, receipt in list(self.gate_receipts.items()):
                if (receipt["captured_source_id"], receipt["captured_source_version"]) == key:
                    self.gate_receipts.pop(rid)
        self.fail("proxy_cleanup")
        before = len(self.proxy)
        self.proxy.difference_update(keys)
        return before - len(self.proxy)

    def release_native(self, key):
        evicted = key in self.native
        self.native.discard(key)
        self.fail("native_cleanup")
        return {"evicted": evicted}


class StreamTests(FakeOnlyTest):
    def stream(self, window=8):
        session = FakeSession()
        return session, bridge.ContinuousElectricEyeStream(session, "src_review", width=128, height=128, max_window_frames=window)

    def push(self, stream, version, pixels=False):
        return stream.ingest_frame(version, [], issue_observation_receipt=True, rgba=b"fake" if pixels else None)

    def test_complete_transaction_failures_invalidate_scope_and_retire_frame(self):
        for stage in ("register_before", "register_native", "register_proxy", "inspect", "compare", "dirty", "transition", "audit", "ghost", "summary", "output"):
            for pixels in (False, True):
                with self.subTest(stage=stage, pixels=pixels), ExitStack() as local:
                    session, stream = self.stream()
                    self.push(stream, "v1", pixels)
                    self.push(stream, "v2", pixels)
                    # Also retain an older fresh receipt on identical frames to verify whole-scope invalidation.
                    stream._observation_ledger["v1"]["fresh"] = True
                    session.failure = stage
                    kwargs = {}
                    if stage == "ghost":
                        kwargs["verify_staged_ghost_id"] = "missing"
                    if stage == "summary":
                        local.enter_context(patch.object(stream, "compute_temporal_window_summary", side_effect=RuntimeError("synthetic summary failure")))
                    if stage == "output":
                        original = bridge.assert_no_raw_buffer_leakage
                        def reject_output(value):
                            if value.get("status") == "ok" and "temporal_window_summary" in value:
                                raise RuntimeError("synthetic output failure")
                            return original(value)
                        local.enter_context(patch.object(bridge, "assert_no_raw_buffer_leakage", side_effect=reject_output))
                    with self.assertRaises((RuntimeError, KeyError)):
                        stream.ingest_frame("v3", [], rgba=b"fake" if pixels else None, issue_observation_receipt=True, **kwargs)
                    for version in ("v1", "v2", "v3"):
                        receipt = stream.check_receipt_freshness(version)
                        self.assertFalse(receipt["fresh"])
                        self.assertFalse(receipt["observed"])
                    key = (stream.source_id, "v3")
                    self.assertNotIn(key, session.native)
                    self.assertNotIn(key, session.proxy)
                    self.assertNotIn(key, session._source_dimensions)
                    self.assertNotIn("v3", stream._window_versions)
                    self.assertNotIn("v3", stream._frame_records)
                    self.assertFalse([t for t in stream._transition_records if t["to_version"] == "v3"])
                    self.assertEqual(stream._tick_counter, 2)
                    with self.assertRaisesRegex(RuntimeError, "FAILED_CLOSED"):
                        self.push(stream, "v4", pixels)
                    session.failure = None
                    self.assertFalse(session.native | session.proxy)
                    self.assertEqual(stream.release(), 0)
                    self.assertFalse(session.native | session.proxy)

    def test_eviction_failure_retirement_is_retried_by_release(self):
        session, stream = self.stream(window=2)
        self.push(stream, "v1")
        self.push(stream, "v2")
        session.failure = "evict"
        with self.assertRaisesRegex(RuntimeError, "evict"):
            self.push(stream, "v3")
        self.assertFalse(stream.check_receipt_freshness("v2")["fresh"])
        self.assertEqual(stream._pending_retirements, {"v1", "v2", "v3"})
        session.failure = None
        stream.release()
        self.assertFalse(session.native | session.proxy | stream._pending_retirements)

    def test_success_preserves_decay_unchanged_freshness_and_bounded_window(self):
        session, stream = self.stream(window=2)
        session.changed = False
        self.assertEqual(self.push(stream, "v1")["tick"], 1)
        self.assertEqual(self.push(stream, "v2")["decayed_to_stale_versions"], [])
        self.assertTrue(stream.check_receipt_freshness("v1")["fresh"])
        session.changed = True
        out = self.push(stream, "v3")
        self.assertEqual(out["decayed_to_stale_versions"], ["v1", "v2"])
        self.assertEqual(len(session.events), 2)
        self.assertEqual(stream._window_versions, ["v2", "v3"])
        self.assertEqual(len(session.native), 2)
        self.assertTrue(stream.check_receipt_freshness("v3")["fresh"])
        with self.assertRaisesRegex(ValueError, "IMMUTABLE"):
            self.push(stream, "v3")
        self.assertTrue(stream.check_receipt_freshness("v3")["fresh"])
        stream.release()
        self.assertFalse(stream.check_receipt_freshness("v3")["fresh"])


class ReceiptRevocationTests(FakeOnlyTest):
    def setup_stream(self):
        session = ReceiptSession()
        stream = bridge.ContinuousElectricEyeStream(session, "src_review", width=128, height=128)
        for version in ("v1", "v2"):
            session.changed = False
            stream.ingest_frame(version, [], issue_observation_receipt=True)
        receipts = copy.deepcopy(session._harness._observation_receipts)
        return session, stream, receipts

    def assert_revoked(self, session, stream, receipts):
        session.failure = None
        for rid, receipt in receipts.items():
            self.assertFalse(stream.check_receipt_freshness(receipt["captured_source_version"])["fresh"])
            self.assertFalse(session._harness._out_of_process_vault_client.rpc_verify_observation_receipt(receipt))
            result = session._harness.promote_render_to_observed(receipt["render_id"], receipt)
            self.assertEqual(result["status"], "rejected")
            render = session._harness._renders[receipt["render_id"]]
            self.assertFalse(render["observed"])
            self.assertEqual(render["uncertainty_state"], "unknown")
            self.assertIsNone(session._harness._observation_receipts[rid]["receipt_signature"])
        self.assertTrue(session._harness._sources)
        with self.assertRaisesRegex(RuntimeError, "FAILED_CLOSED"):
            stream.ingest_frame("v9", [])

    def test_failed_ingest_revokes_retained_session_harness_and_gate_authority(self):
        for stage in ("inspect", "compare", "audit"):
            with self.subTest(stage=stage):
                session, stream, receipts = self.setup_stream()
                session.changed = True
                session.failure = stage
                with self.assertRaises(RuntimeError):
                    stream.ingest_frame("v3", [], issue_observation_receipt=True)
                self.assert_revoked(session, stream, receipts)
                self.assertFalse(session.native | session.proxy | session.gate_sources)
                self.assertFalse(session._source_dimensions)
                self.assertFalse(stream._window_versions)

    def test_release_owns_all_candidates_before_each_fallible_cleanup(self):
        for stage in ("proxy_cleanup", "native_cleanup", "cleanup_reply"):
            with self.subTest(stage=stage), ExitStack() as local:
                session, stream, receipts = self.setup_stream()
                session.failure = stage
                if stage == "cleanup_reply":
                    local.enter_context(patch.object(bridge, "assert_no_raw_buffer_leakage", side_effect=RuntimeError("cleanup reply")))
                with self.assertRaises(RuntimeError):
                    stream.release()
                self.assertEqual(stream._pending_retirements, {"v1", "v2"})
                self.assertFalse(session._source_dimensions)
                if stage == "cleanup_reply":
                    local.close()
                self.assert_revoked(session, stream, receipts)
                stream.release()
                self.assertFalse(session.native | session.proxy | session.gate_sources | stream._pending_retirements)
                self.assertEqual(session._harness.release_source_pixels.call_count, 4)


class PictureTests(FakeOnlyTest):
    def test_lazy_png_survives_eight_and_retires_with_count_and_byte_budget(self):
        for byte_budget in (False, True):
            with self.subTest(byte_budget=byte_budget), tempfile.TemporaryDirectory() as tmp:
                server = mcp.ElectricVisionServer(Path(tmp), "fake-test")
                self.addCleanup(server.close)
                session = FakeSession()
                self.stack.enter_context(patch.object(server, "ensure_session", return_value=session))
                self.stack.enter_context(patch.object(sources, "load_image", return_value=object()))
                self.stack.enter_context(patch.object(mcp, "_fitted", return_value=(b"rgba", b"", {})))
                def reply(sess, sid, ver, reg, **kwargs):
                    return server._result({"source_id": sid, "source_version": ver},
                                          [server._picture_png((sid, ver))] if kwargs["want_image"] else None)
                self.stack.enter_context(patch.object(server, "_inspect_reply", side_effect=reply))
                canvas = Mock()
                canvas.size = (128, 128)
                self.stack.enter_context(patch.object(mcp.Image, "frombytes", return_value=canvas))
                self.stack.enter_context(patch.object(mcp.Image, "new", return_value=canvas))
                encode = self.stack.enter_context(patch.object(mcp, "_encode_png_bounded", return_value=(b"fake-png", "sha")))
                self.stack.enter_context(patch.object(mcp, "MAX_RETAINED_SOURCES", 100 if byte_budget else 10))
                self.stack.enter_context(patch.object(mcp, "NATIVE_BYTES_BUDGET", 10 * 128 * 128 * 4 if byte_budget else 10**9))
                for _ in range(10):
                    server._handle_look({"file": "/fake/picture.png", "source_id": "src_picture", "width": 128, "height": 128, "image": False})
                encode.assert_not_called()
                self.assertEqual(len(server.picture_rgba_cache), 10)
                self.assertEqual(server._picture_png(("src_picture", "v1")), b"fake-png")
                encode.assert_called_once()
                self.assertEqual(server._picture_png(("src_picture", "v1")), b"fake-png")
                encode.assert_called_once()
                server._handle_look({"file": "/fake/picture.png", "source_id": "src_picture", "width": 128, "height": 128})
                key = ("src_picture", "v1")
                for cache in (session._source_dimensions, server.picture_rgba_cache, server.picture_png_cache):
                    self.assertNotIn(key, cache)
                self.assertIsNone(server._picture_png(key))
                self.assertEqual(len(server.picture_rgba_cache), 10)
                self.assertEqual(server.retired_source_count, 1)


class ServerFailureTests(FakeOnlyTest):
    def test_budget_retains_failed_victim_and_caches_until_retry_finishes(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as local:
            server = mcp.ElectricVisionServer(Path(tmp), "fake-test")
            local.callback(server.close)
            session = FakeSession()
            keys = [("src_review", f"v{i}") for i in range(1, 4)]
            for key in keys:
                session.register_synthetic_canvas(*key, [], width=128, height=128)
                server._source_order.append(key)
                server.compiled_specs_cache[key] = ([], 128, 128)
                server.picture_png_cache[key] = b"png"
                server.picture_rgba_cache[key] = (b"rgba", 128, 128)
            local.enter_context(patch.object(mcp, "MAX_RETAINED_SOURCES", 2))
            session.failure = "evict"
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "evict"):
                    server._enforce_source_budget(session, protect=keys[1:])
                self.assertEqual(server._pending_source_retirements, {keys[0]})
                self.assertIn(keys[0], server._source_order)
                self.assertNotIn(keys[0], session._source_dimensions)
                self.assertIn(keys[0], session.proxy)
                for cache in (server.compiled_specs_cache, server.picture_png_cache, server.picture_rgba_cache):
                    self.assertIn(keys[0], cache)
            session.failure = None
            server._enforce_source_budget(session, protect=keys[1:])
            self.assertFalse(server._pending_source_retirements)
            self.assertNotIn(keys[0], session.proxy)
            for cache in (server.compiled_specs_cache, server.picture_png_cache, server.picture_rgba_cache):
                self.assertNotIn(keys[0], cache)
            self.assertEqual(server.retired_source_count, 1)

    def test_startup_close_and_reset_clean_only_owned_frame_files_in_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            outside = root / "keep.rgba"
            outside.write_bytes(b"keep")
            unrelated = frames / "picture.png"
            unrelated.write_bytes(b"keep")
            nested = frames / "nested"
            nested.mkdir()
            (nested / "frame-000001.rgba").write_bytes(b"keep")
            names = ["frame-000001.png", "frame-000001.rgba", "frame-000002.rgba.partial", "frame-000002.png.tmp"]
            for name in names:
                (frames / name).write_bytes(b"stale")
            (frames / "frame-000003.rgba").symlink_to(outside)
            server = mcp.ElectricVisionServer(root, "fake-test")
            try:
                self.assertFalse([p for p in frames.iterdir() if p.name.startswith("frame-")])
                for operation in (server._reset_stream, server.close):
                    for name in names:
                        (frames / name).write_bytes(b"orphan")
                    operation()
                    self.assertFalse([p for p in frames.iterdir() if p.name.startswith("frame-")])
                    self.assertEqual(outside.read_bytes(), b"keep")
                    self.assertEqual(unrelated.read_bytes(), b"keep")
                    self.assertEqual((nested / "frame-000001.rgba").read_bytes(), b"keep")
            finally:
                server.close()

    def test_close_keeps_failed_file_cleanup_owned_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = mcp.ElectricVisionServer(Path(tmp), "fake-test")
            path = server.frames_dir / "frame-000001.rgba.partial"
            path.write_bytes(b"partial")
            original = Path.unlink
            def unlink(p, **kwargs):
                if p == path:
                    raise OSError("unlink failure")
                return original(p, **kwargs)
            with patch.object(Path, "unlink", unlink), self.assertRaisesRegex(OSError, "unlink failure"):
                server.close()
            self.assertEqual(server._pending_frame_retirements, {path})
            self.assertTrue(path.exists())
            server.close()
            self.assertFalse(path.exists())
            self.assertFalse(server._pending_frame_retirements)

    def test_lazy_png_failure_closes_stream_and_reset_retries_failed_release(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as local:
            server = mcp.ElectricVisionServer(Path(tmp), "fake-test")
            local.callback(server.close)
            session = ReceiptSession()
            session.changed = False
            stream = bridge.ContinuousElectricEyeStream(session, mcp.STREAM_SOURCE_ID, width=128, height=128)
            server.stream = stream
            local.enter_context(patch.object(server, "ensure_session", return_value=session))
            args = {"action": "push", "frames": [mcp._PixelFrame(b"rgba", b"", {})], "issue_receipts": True}
            self.assertFalse(server._handle_stream(args)["isError"])
            self.assertFalse(list(server.frames_dir.glob("*.png")))
            receipts = copy.deepcopy(session._harness._observation_receipts)
            canvas = Mock(size=(128, 128))
            local.enter_context(patch.object(mcp.Image, "frombytes", return_value=canvas))
            local.enter_context(patch.object(mcp.Image, "new", return_value=canvas))
            local.enter_context(patch.object(mcp, "_encode_png_bounded", side_effect=RuntimeError("lazy PNG failure")))
            session.failure = "proxy_cleanup"
            with self.assertRaisesRegex(RuntimeError, "lazy PNG failure"):
                server._handle_stream({**args, "image": True})
            self.assertEqual(server.stream_sequence, 1)
            self.assertEqual(stream._pending_retirements, {"v1", "v2"})
            self.assertFalse(list(server.frames_dir.iterdir()))
            with self.assertRaisesRegex(RuntimeError, "proxy_cleanup"):
                server._reset_stream()
            self.assertIs(server.stream, stream)
            self.assertEqual(stream._pending_retirements, {"v1", "v2"})
            ReceiptRevocationTests.assert_revoked(self, session, stream, receipts)
            server._reset_stream()
            self.assertIsNone(server.stream)
            self.assertFalse(session.native | session.proxy | stream._pending_retirements)

    def test_whole_push_storage_publication_and_reply_failures_revoke_and_block(self):
        stages = ("rgba_write", "png_write", "chmod", "metadata", "summary", "contact_sheet", "detail_read",
                  "output", "reply", "status", "second_ingest", "retire_files")
        for stage in stages:
            for pixels in (False, True):
                with self.subTest(stage=stage, pixels=pixels), tempfile.TemporaryDirectory() as tmp, ExitStack() as local:
                    server = mcp.ElectricVisionServer(Path(tmp), "fake-test")
                    local.callback(server.close)
                    session = ReceiptSession()
                    session.changed = False
                    stream = bridge.ContinuousElectricEyeStream(session, mcp.STREAM_SOURCE_ID, width=128, height=128)
                    server.stream = stream
                    local.enter_context(patch.object(server, "ensure_session", return_value=session))
                    local.enter_context(patch.object(mcp, "_rasterize_normalized_specs", return_value=b"rgba"))
                    local.enter_context(patch.object(mcp, "_render_specs_to_png", return_value=(b"png", "sha")))
                    panel = local.enter_context(patch.object(mcp, "_render_temporal_contact_sheet_png", return_value=(b"panel", "sha")))
                    frame = mcp._PixelFrame(b"rgba", b"png", {}) if pixels else []
                    args = {"action": "push", "frames": [frame, frame], "issue_receipts": True, "image": True}
                    result = server._handle_stream(args)
                    self.assertFalse(result["isError"])
                    self.assertEqual(server.stream_sequence, 2)
                    self.assertEqual(len(server.stream_frames_meta), 2)
                    receipts = copy.deepcopy(session._harness._observation_receipts)
                    original_publish = mcp._atomic_json
                    def publish(path, value):
                        self.assertEqual(server.stream_sequence, 2)
                        if value["phase"] == "streaming":
                            self.assertEqual(len(server.stream_frames_meta), 2)
                            if stage == "status":
                                raise OSError("status failure")
                        return original_publish(path, value)
                    publication = local.enter_context(patch.object(mcp, "_atomic_json", side_effect=publish))
                    original_write, original_read = Path.write_bytes, Path.read_bytes
                    if stage in ("rgba_write", "png_write"):
                        def write(path, data):
                            if path.suffix == (".rgba" if stage == "rgba_write" else ".png"):
                                original_write(path, b"partial")
                                raise OSError("write failure")
                            return original_write(path, data)
                        local.enter_context(patch.object(Path, "write_bytes", write))
                    elif stage == "chmod":
                        local.enter_context(patch.object(mcp.os, "chmod", side_effect=OSError("chmod failure")))
                    elif stage == "metadata":
                        local.enter_context(patch.object(mcp.uuid, "uuid4", side_effect=RuntimeError("metadata failure")))
                    elif stage == "summary":
                        original_ingest = stream.ingest_frame
                        def ingest(*args, **kwargs):
                            out = original_ingest(*args, **kwargs)
                            out["temporal_window_summary"] = {}
                            return out
                        local.enter_context(patch.object(stream, "ingest_frame", side_effect=ingest))
                    elif stage == "contact_sheet":
                        panel.side_effect = RuntimeError("panel failure")
                    elif stage == "detail_read":
                        def read(path):
                            if path.suffix == ".png":
                                raise OSError("detail failure")
                            return original_read(path)
                        local.enter_context(patch.object(Path, "read_bytes", read))
                    elif stage == "output":
                        local.enter_context(patch.object(mcp, "assert_no_raw_buffer_leakage", side_effect=RuntimeError("output failure")))
                    elif stage == "reply":
                        local.enter_context(patch.object(server, "_result", side_effect=RuntimeError("reply failure")))
                    elif stage == "second_ingest":
                        original_inspect = session.inspect_canvas
                        def inspect(sid, ver, **kwargs):
                            if ver == "v4":
                                raise RuntimeError("second frame failure")
                            return original_inspect(sid, ver, **kwargs)
                        local.enter_context(patch.object(session, "inspect_canvas", side_effect=inspect))
                    elif stage == "retire_files":
                        local.enter_context(patch.object(mcp, "MAX_RETAINED_FRAMES", 2))
                        local.enter_context(patch.object(Path, "unlink", side_effect=OSError("retirement failure")))
                    invalidate = local.enter_context(patch.object(stream, "invalidate_failure", wraps=stream.invalidate_failure))
                    with self.assertRaises((RuntimeError, OSError, KeyError)):
                        server._handle_stream(args)
                    self.assertTrue(invalidate.called)
                    self.assertEqual(server.stream_sequence, 2)
                    self.assertEqual(stream._tick_counter, 2)
                    self.assertFalse(server.stream_frames_meta)
                    self.assertEqual(server.last_summary["phase"], "failed")
                    self.assertFalse(session.native | session.proxy | session.gate_sources)
                    ReceiptRevocationTests.assert_revoked(self, session, stream, receipts)
                    for call in publication.call_args_list:
                        if stage != "status":
                            self.assertNotEqual(call.args[1]["phase"], "streaming")
                    if stage == "retire_files":
                        self.assertTrue(server._pending_frame_retirements)
                        local.close()
                        self.assertFalse(list(server.frames_dir.iterdir()))
                        self.assertFalse(server._pending_frame_retirements)
                    else:
                        self.assertFalse(list(server.frames_dir.iterdir()))
                        self.assertFalse(server._pending_frame_retirements)


class VisualTests(FakeOnlyTest):
    def test_image_decoder_receives_checked_snapshot_despite_path_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"original")
            decoder = Mock()
            decoder.__enter__ = Mock(return_value=SimpleNamespace(width=1, height=1, convert=lambda mode: "decoded"))
            decoder.__exit__ = Mock(return_value=False)
            def decode(source):
                self.assertIsInstance(source, io.BytesIO)
                path.unlink()
                path.write_bytes(b"replacement")
                self.assertEqual(source.read(), b"original")
                return decoder
            with patch.object(sources.Image, "open", side_effect=decode):
                self.assertEqual(sources.load_image(str(path)), "decoded")

    def test_checked_descriptor_survives_final_and_parent_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            path = root / "asset"
            path.write_bytes(b"safe")
            with sources._checked_file(str(path), 10) as handle:
                root.rename(Path(tmp) / "old")
                root.mkdir()
                path.write_bytes(b"replacement")
                self.assertEqual(sources._read_bounded(handle, 10), b"safe")

    def test_fifo_symlinks_traversal_and_growth_are_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "safe").write_bytes(b"safe")
            (root / "link").symlink_to(root / "safe")
            (root / "dir").symlink_to(root, target_is_directory=True)
            os.mkfifo(root / "fifo")
            original_open = sources.os.open
            flags_seen = []
            def record_open(path, flags, *args, **kwargs):
                flags_seen.append(flags)
                # This assertion makes a regression deterministic instead of hanging on FIFO open.
                self.assertTrue(flags & os.O_NONBLOCK)
                self.assertTrue(flags & os.O_NOFOLLOW)
                return original_open(path, flags, *args, **kwargs)
            with patch.object(sources.os, "open", side_effect=record_open), patch.object(sources.Image, "open") as decoder:
                for path in (root / "link", root / "dir" / "safe", root / "fifo", str(root) + "/../safe"):
                    with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                        sources.load_image(str(path))
                decoder.assert_not_called()
            self.assertTrue(flags_seen)
            with sources._checked_file(str(root / "safe"), 4) as handle:
                (root / "safe").write_bytes(b"grew-beyond-limit")
                with self.assertRaisesRegex(ValueError, "TOO_LARGE"):
                    sources._read_bounded(handle, 4)

    def test_svg_reads_bounded_snapshot_with_stub_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.svg"
            path.write_bytes(b"<svg>safe</svg>")
            with patch.object(sources, "render_svg", return_value="fake-svg") as render:
                self.assertEqual(sources.load_svg(str(path)), "fake-svg")
                render.assert_called_once_with("<svg>safe</svg>")
            with patch.object(sources, "MAX_SVG_TEXT_BYTES", 4), patch.object(sources, "render_svg") as render:
                with self.assertRaisesRegex(ValueError, "TOO_LARGE"):
                    sources.load_svg(str(path))
                render.assert_not_called()

    def test_video_probe_and_all_decodes_use_same_private_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(b"original-video")
            snapshots = []
            def run(argv, **kwargs):
                is_probe = argv[0] == sources.FFPROBE
                snapshot = Path(argv[-1] if is_probe else argv[argv.index("-i") + 1])
                snapshots.append(snapshot)
                self.assertNotEqual(snapshot, path)
                self.assertEqual(snapshot.read_bytes(), b"original-video")
                self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
                if is_probe:
                    path.write_bytes(b"replacement-video")
                    return json.dumps({"streams": [{"width": 32, "height": 16, "avg_frame_rate": "1/1"}], "format": {"duration": "2"}}).encode()
                return b"stub-png"
            decoder = Mock()
            decoder.__enter__ = Mock(return_value=SimpleNamespace(width=32, height=16, convert=lambda mode: "frame"))
            decoder.__exit__ = Mock(return_value=False)
            with patch.object(sources, "_run_owned", side_effect=run), patch.object(sources.Image, "open", return_value=decoder):
                self.assertEqual(sources.video_frames(str(path), [0, 1]), ["frame", "frame"])
            self.assertEqual(len(snapshots), 3)
            self.assertEqual(len(set(snapshots)), 1)
            self.assertFalse(snapshots[0].exists())

    def test_mcp_video_look_pins_one_snapshot_for_probe_and_all_frames(self):
        for multiple in (False, True):
            for fail_decode in (False, True):
                with self.subTest(multiple=multiple, fail_decode=fail_decode), tempfile.TemporaryDirectory() as tmp, ExitStack() as local:
                    path = Path(tmp) / "clip.mp4"
                    path.write_bytes(b"original-video")
                    server = mcp.ElectricVisionServer(Path(tmp) / "server", "fake-test")
                    local.callback(server.close)
                    snapshots = []
                    def run(argv, **kwargs):
                        probe = argv[0] == sources.FFPROBE
                        snapshot = Path(argv[-1] if probe else argv[argv.index("-i") + 1])
                        snapshots.append(snapshot)
                        self.assertNotEqual(snapshot, path)
                        self.assertEqual(snapshot.read_bytes(), b"original-video")
                        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
                        if probe:
                            path.write_bytes(b"replacement-video")
                            return json.dumps({"streams": [{"width": 32, "height": 16}], "format": {"duration": "3"}}).encode()
                        if fail_decode:
                            raise RuntimeError("decode failure")
                        return b"stub-png"
                    decoder = Mock()
                    decoder.__enter__ = Mock(return_value=SimpleNamespace(width=32, height=16, convert=lambda mode: "frame"))
                    decoder.__exit__ = Mock(return_value=False)
                    local.enter_context(patch.object(sources, "_run_owned", side_effect=run))
                    local.enter_context(patch.object(sources.Image, "open", return_value=decoder))
                    local.enter_context(patch.object(mcp, "_fitted", return_value=(b"rgba", b"", {})))
                    local.enter_context(patch.object(server, "ensure_session", return_value=FakeSession()))
                    stream_push = local.enter_context(patch.object(server, "_handle_stream", side_effect=lambda args: server._result({"status": "ok"})))
                    local.enter_context(patch.object(server, "_inspect_reply", side_effect=lambda *args, **kwargs: server._result({"status": "ok"})))
                    args = {"file": str(path), "width": 128, "height": 128}
                    if multiple:
                        args["frames"] = {"start": 0, "every": 1, "count": 3}
                    if fail_decode:
                        with self.assertRaisesRegex(RuntimeError, "decode failure"):
                            server._handle_look(args)
                        stream_push.assert_not_called()
                    else:
                        self.assertFalse(server._handle_look(args)["isError"])
                        self.assertEqual(len(snapshots), 4 if multiple else 2)
                        if multiple:
                            self.assertEqual(len(stream_push.call_args.args[0]["frames"]), 3)
                    self.assertEqual(len(set(snapshots)), 1)
                    self.assertFalse(snapshots[0].exists())

    def test_hostile_video_dimensions_prevent_decode(self):
        with tempfile.NamedTemporaryFile() as path:
            output = json.dumps({"streams": [{"width": 100000, "height": 100000}]}).encode()
            with patch.object(sources, "_run_owned", return_value=output) as run, patch.object(sources.Image, "open") as decode:
                with self.assertRaises(ValueError):
                    sources.video_frames(path.name, [0])
                self.assertEqual(run.call_count, 1)
                decode.assert_not_called()

    def page_server(self, root, *, _implementation=None, **kwargs):
        import http.server
        # Extract the real handler without creating a socket or starting its thread.
        with patch.object(http.server, "ThreadingHTTPServer", return_value=Mock()) as factory:
            server = (_implementation or sources._PageServer)(root, **kwargs)
        self.addCleanup(lambda: os.close(server.root_fd))
        handler_class = factory.call_args.args[1].func
        def request(path):
            handler = object.__new__(handler_class)
            handler.path = path
            handler.send_response = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler.send_error = Mock()
            result = handler.send_head()
            if result is None:
                return None, handler
            with result:
                return result.read(), handler
        return server, request

    def test_nested_page_assets_are_confined_bounded_and_pinned_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "page"
            root.mkdir()
            (root / "safe.html").write_bytes(b"safe")
            (root / "big.mp4").write_bytes(b"x" * 9)
            (Path(tmp) / "secret").write_bytes(b"secret")
            (root / "link").symlink_to(Path(tmp) / "secret")
            (root / "dir").symlink_to(Path(tmp), target_is_directory=True)
            (root / "index.html").symlink_to(Path(tmp) / "secret")
            os.mkfifo(root / "fifo")
            _, request = self.page_server(root)
            with patch.object(sources, "MAX_FILE_BYTES", 8):
                self.assertEqual(request("/safe.html?query=yes")[0], b"safe")
                for path in ("/link", "/dir/secret", "/", "/%2e%2e/secret", "/%2e/safe.html", "/big.mp4", "/fifo"):
                    with self.subTest(path=path):
                        data, handler = request(path)
                        self.assertIsNone(data)
                        handler.send_error.assert_called_once()
                root.rename(Path(tmp) / "old")
                root.mkdir()
                (root / "safe.html").write_bytes(b"swapped")
                self.assertEqual(request("/safe.html")[0], b"safe")

    def test_capture_stub_uses_original_page_and_pinned_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "page"
            root.mkdir()
            page = root / "a #.html"
            page.write_bytes(b"original")
            (root / "media").write_bytes(b"safe-media")
            def fake_server(folder, **kwargs):
                self.assertEqual(kwargs["page_bytes"], b"original")
                root.rename(Path(tmp) / "old")
                root.mkdir()
                page.write_bytes(b"replacement")
                _, request = self.page_server(folder, _implementation=original_server, **kwargs)
                self.assertEqual(request("/a%20%23.html")[0], b"original")
                self.assertEqual(request("/media")[0], b"safe-media")
                context = Mock()
                context.__enter__ = Mock(return_value="fake-origin")
                context.__exit__ = Mock(return_value=False)
                return context
            original_server = sources._PageServer
            # page_server must instantiate the original class even while capture uses the stub.
            with patch.object(sources, "_PageServer", side_effect=fake_server), \
                 patch.object(sources, "_chrome_screenshot", return_value="fake-frame") as capture:
                self.assertEqual(sources.capture_page(str(page), width=128, height=128, at_ms=[1], root=str(root)), ["fake-frame"])
                self.assertIn("a%20%23.html", capture.call_args.args[0])

    def test_page_outside_root_and_symlink_root_fail_before_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "page"
            root.mkdir()
            outside = Path(tmp) / "outside.html"
            outside.write_bytes(b"outside")
            link = Path(tmp) / "link"
            link.symlink_to(root, target_is_directory=True)
            with patch.object(sources, "_PageServer") as server:
                with self.assertRaisesRegex(ValueError, "OUTSIDE_ROOT"):
                    sources.capture_page(str(outside), width=128, height=128, at_ms=[1], root=str(root))
                with self.assertRaises(OSError):
                    sources.capture_page(str(link / "index.html"), width=128, height=128, at_ms=[1], root=str(link))
                server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
