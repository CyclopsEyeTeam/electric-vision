"""Synthetic/mocked regressions: no Chrome, desktop, camera, microphone or model calls."""
import io
import json
import multiprocessing
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from PIL import Image
import visual_sources as sources
import exec_gate_daemons as gates
import prod_gate_enclave as enclave
from electric_vision_agent_bridge import ElectricVisionAgentSession
from electric_vision_mcp import ElectricVisionServer, _fitted


def dot(x):
    return [{"k": "dot", "x": x, "y": 0.3, "r": 0.05, "c": [255, 0, 0]}]


def rpc(path, payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        sock.connect(str(path))
        gates._send_framed_json(sock, payload)
        return gates._recv_framed_json(sock)


class SourceTests(unittest.TestCase):
    def test_server_refuses_symlink_files_directories_and_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "served"
            root.mkdir()
            (root / "ok.txt").write_text("safe")
            secret = Path(tmp) / "secret.txt"
            secret.write_text("outside-secret")
            (root / "link.txt").symlink_to(secret)
            (root / "dir").symlink_to(Path(tmp), target_is_directory=True)
            (root / "index.html").symlink_to(secret)
            with sources._PageServer(root) as origin:
                self.assertEqual(urllib.request.urlopen(f"http://{origin}/ok.txt").read(), b"safe")
                for path in ("/link.txt", "/dir/secret.txt", "/", "/%2e%2e/secret.txt"):
                    with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError):
                        urllib.request.urlopen(f"http://{origin}{path}")

    def test_server_pins_root_when_directory_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "served"
            root.mkdir()
            (root / "ok.txt").write_text("safe")
            with sources._PageServer(root) as origin:
                root.rename(Path(tmp) / "original")
                root.mkdir()
                (root / "ok.txt").write_text("replacement")
                self.assertEqual(urllib.request.urlopen(f"http://{origin}/ok.txt").read(), b"safe")

    def test_owned_group_kills_descendant_after_success_and_timeout(self):
        # Grandchild is a synthetic wait process, not Chrome. Its inherited pipe
        # deliberately tests the case where the leader has already exited.
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],stdout=subprocess.DEVNULL); print(p.pid,flush=True); "
        for suffix, timeout in (("", 2), ("time.sleep(60)", 0.1)):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                pidfile = Path(tmp) / "pid"
                script = code.replace("print(p.pid,flush=True)", f"open({str(pidfile)!r},'w').write(str(p.pid))") + suffix
                if suffix:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        sources._run_owned([sys.executable, "-c", script], env=dict(os.environ), timeout=timeout)
                else:
                    sources._run_owned([sys.executable, "-c", script], env=dict(os.environ), timeout=timeout)
                pid = int(pidfile.read_text())
                # A killed orphan may briefly be an init-owned zombie; it cannot execute.
                status = Path(f"/proc/{pid}/stat")
                for _ in range(1000):
                    try:
                        state = status.read_text().split()[2]
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if state == "Z":
                        break
                    os.sched_yield()
                else:
                    self.fail("task-owned descendant still running")

    def test_owned_output_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "OUTPUT_TOO_LARGE"):
            sources._run_owned([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*1000000)"],
                               env=dict(os.environ), timeout=3, max_output=100)

    def test_video_rejects_coded_dimensions_before_decode(self):
        with tempfile.NamedTemporaryFile() as video:
            bad = json.dumps({"streams": [{"width": 100000, "height": 100000}]}).encode()
            with patch.object(sources, "_run_owned", return_value=bad) as run:
                with self.assertRaises(ValueError):
                    sources.video_frames(video.name, [0])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][0], sources.FFPROBE)

    def test_video_scales_and_checks_png_header_before_convert(self):
        for dimensions, rejected in (((32, 16), False), ((2000, 16), True)):
            with self.subTest(dimensions=dimensions), tempfile.NamedTemporaryFile() as video:
                buffer = io.BytesIO()
                Image.new("RGB", dimensions).save(buffer, format="PNG")
                with patch.object(sources, "_video_info_snapshot", return_value={}), patch.object(sources, "_run_owned", return_value=buffer.getvalue()) as run:
                    if rejected:
                        with self.assertRaisesRegex(ValueError, "DECODE_DIMENSIONS"):
                            sources.video_frames(video.name, [0])
                    else:
                        self.assertEqual(sources.video_frames(video.name, [0])[0].size, dimensions)
                    self.assertIn("-vf", run.call_args.args[0])
                    self.assertEqual(run.call_args.kwargs["max_output"], 16 * 1024 * 1024)

    def test_native_colour_boxes_preserve_coordinates_and_alpha(self):
        image = Image.new("RGBA", (1920, 1080), (0, 255, 0, 0))
        image.paste((255, 0, 0, 1), (1700, 900, 1705, 905))
        comps = [{"region_id": 1, "bbox": {"x_min": 1699, "y_min": 899, "x_max": 1705, "y_max": 905}}]
        result = sources.component_colours(image, comps)
        self.assertEqual(result[0]["colours"], [{"hex": "#f80808", "name": "red", "share": 1.0}])
        self.assertEqual(sources.component_colours(image, [{"bbox": {"x_min": 10**9, "x_max": 10**12, "y_min": 0, "y_max": 10**12}}]), [])

    def test_fit_does_not_encode_without_image(self):
        import electric_vision_mcp as mcp
        with patch.object(mcp, "_encode_png_bounded", side_effect=AssertionError("unrequested PNG")):
            self.assertEqual(_fitted(Image.new("RGBA", (8, 8)), 128, 128, want_image=False)[1], b"")


class IpcTests(unittest.TestCase):
    def test_absolute_deadline_includes_header_and_body(self):
        reader, writer = socket.socketpair()
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        writer.sendall(struct.pack(">I", 100) + b"{")
        stop = threading.Event()
        self.addCleanup(stop.set)
        def drip():
            while not stop.wait(0.025):
                try:
                    writer.sendall(b" ")
                except OSError:
                    return
        thread = threading.Thread(target=drip, daemon=True)
        thread.start()
        start = time.monotonic()
        with patch.object(gates, "IPC_REQUEST_TIMEOUT_S", 0.15), self.assertRaises(TimeoutError):
            gates._recv_framed_json(reader)
        self.assertLess(time.monotonic() - start, 0.5)
        stop.set()
        thread.join(1)

    def exercise_bad_clients(self, path, valid):
        def ping():
            return rpc(path, {**valid, "rpc_nonce": "nonce-" + str(time.monotonic_ns())})
        malformed = [b"[]", b"null", b"1", b"{", b"\xff", b'{"method":[],"params":{}}', b'{"method":"x","params":[]}']
        for raw in malformed:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(str(path))
                sock.sendall(struct.pack(">I", len(raw)) + raw)
                sock.shutdown(socket.SHUT_WR)
            self.assertEqual(ping()["status"], "ok")
        for wire in (struct.pack(">I", gates.MAX_IPC_FRAME_BYTES + 1), b"\x00\x00", struct.pack(">I", 50) + b"{"):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(str(path))
                sock.sendall(wire)
                sock.shutdown(socket.SHUT_WR)
            self.assertEqual(ping()["status"], "ok")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stalled:
            stalled.connect(str(path))
            stalled.sendall(b"\x00")
            started = time.monotonic()
            self.assertEqual(ping()["status"], "ok")
            self.assertLess(time.monotonic() - started, 2.8)

    def test_all_helpers_survive_malformed_disconnected_and_stalled_clients(self):
        with tempfile.TemporaryDirectory(prefix="ev-ipc-") as tmp:
            root = Path(tmp)
            managers = [gates.OutOfProcessVaultGateDaemon(str(root / "v")),
                        gates.OutOfProcessAuditSinkDaemon(str(root / "a"), str(root / "ledger")),
                        gates.OutOfProcessQuarantineAndKillSwitchDaemon(str(root / "q"), str(root / "forensic"))]
            methods = ["ping_status", "export_chain_head", "get_status"]
            for manager, method in zip(managers, methods):
                with self.subTest(helper=method), manager:
                    self.exercise_bad_clients(manager.socket_path, {"method": method, "params": {}, "rpc_nonce": "unique-" + str(time.monotonic_ns())})
            ctx = multiprocessing.get_context("fork")
            ready = ctx.Event()
            path = root / "h"
            proc = ctx.Process(target=enclave._hsm_kms_enclave_loop, args=(str(path), os.getuid(), os.getgid(), ready))
            proc.start()
            try:
                self.assertTrue(ready.wait(5))
                self.exercise_bad_clients(path, {"method": "ping_hsm", "params": {}})
                self.assertEqual(rpc(path, {"method": "shutdown"})["status"], "ok")
            finally:
                proc.join(3)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(1)
            self.assertEqual(proc.exitcode, 0)

    def test_nonce_capacity_fails_closed_without_eviction_and_halt_works(self):
        with tempfile.TemporaryDirectory(prefix="ev-nonce-") as tmp, patch.object(gates, "MAX_RPC_NONCES", 3):
            daemon = gates.OutOfProcessVaultGateDaemon(str(Path(tmp) / "v"))
            with daemon as client:
                for nonce in ("nonce-000", "nonce-001", "nonce-002"):
                    status = client._call("ping_status", rpc_nonce=nonce)
                self.assertEqual(status["nonce_records"], 3)
                self.assertEqual(status["verified_peer_connections"], 3)
                with self.assertRaisesRegex(Exception, "CAPACITY"):
                    client._call("ping_status", rpc_nonce="nonce-003")
                with self.assertRaisesRegex(Exception, "REPLAY"):
                    client._call("ping_status", rpc_nonce="nonce-000")
                self.assertTrue(client.emergency_lock_vault()["emergency_locked"])

    def test_capability_capacity_retains_consumed_replay_tombstones(self):
        with tempfile.TemporaryDirectory(prefix="ev-cap-") as tmp, patch.object(gates, "MAX_CAPABILITY_RECORDS", 2):
            with gates.OutOfProcessVaultGateDaemon(str(Path(tmp) / "v")) as client:
                client.store_verified_source("src_cap_test", "v1", "captured_frame",
                                             [gates.DrawingMark("m_test", "rect", 0, 0, 3, 3, (255, 0, 0, 255))])
                tokens = [client.request_capability_token(render_id=f"render_{i}", captured_source_id="src_cap_test",
                                                         captured_source_version="v1") for i in range(2)]
                params = {"caller_capability_token": tokens[0], "observation_receipt_id": "receipt_0",
                          "captured_source_id": "src_cap_test", "captured_source_version": "v1",
                          "render": {"render_id": "render_0", "render_type": "observed", "provenance_refs": [
                              {"ref_type": "source", "ref_id": "src_cap_test", "ref_version": "v1"}]}}
                self.assertIn("receipt_signature", client._call("issue_observation_receipt", params))
                with self.assertRaisesRegex(PermissionError, "REPLAY"):
                    client._call("issue_observation_receipt", params)
                with self.assertRaisesRegex(PermissionError, "CAPACITY"):
                    client.request_capability_token(render_id="render_2", captured_source_id="src_cap_test", captured_source_version="v1")

    def test_audit_pagination_constant_metadata_and_truncation_detection(self):
        with tempfile.TemporaryDirectory(prefix="ev-audit-") as tmp:
            root = Path(tmp)
            with gates.OutOfProcessAuditSinkDaemon(str(root / "a"), str(root / "ledger")) as client:
                previous = "sha256:" + "0" * 64
                for seq in range(1, 301):
                    body = {"entry_seq": seq, "prev_hash": previous, "logical_clock": "clock", "logical_timestamp": "logical", "utc_wall_timestamp": "utc", "event_type": "SYNTHETIC_TEST"}
                    previous = gates.sha256_hex(gates.canonical_json(body).encode())
                    client.append_entry({**body, "entry_hash": previous})
                head = client.export_chain_head()
                self.assertEqual(head["entry_count"], 300)
                self.assertEqual(head["retained_entry_records"], 0)
                self.assertTrue(client.verify_durable_file()["valid"])
                with self.assertRaisesRegex(RuntimeError, "PAGINATION_REQUIRED"):
                    client.get_entries()
                first = client.get_entries_page(limit=100)
                self.assertEqual(len(first["entries"]), 100)
                self.assertTrue(first["has_more"])
                last = client.get_entries_page(after_seq=first["next_after_seq"])
                self.assertEqual(len(last["entries"]), 200)
                self.assertFalse(last["has_more"])
                data = (root / "ledger").read_bytes()
                (root / "ledger").write_bytes(data[:data.find(b"\n") + 1])
                self.assertEqual(client.verify_durable_file()["error_code"], "ERR_AUDIT_COMMITTED_HEAD_MISMATCH")

    def test_audit_missing_clock_produces_one_failure(self):
        with tempfile.TemporaryDirectory(prefix="ev-audit-") as tmp:
            root = Path(tmp)
            with gates.OutOfProcessAuditSinkDaemon(str(root / "a"), str(root / "ledger")) as client:
                body = {"entry_seq": 1, "prev_hash": "sha256:" + "0" * 64, "logical_clock": "clock", "logical_timestamp": "logical", "utc_wall_timestamp": "utc"}
                client.append_entry({**body, "entry_hash": gates.sha256_hex(gates.canonical_json(body).encode())})
                body.pop("logical_clock")
                (root / "ledger").write_text(gates.canonical_json({**body, "entry_hash": gates.sha256_hex(gates.canonical_json(body).encode())}) + "\n")
                self.assertEqual(client.verify_durable_file()["error_code"], "ERR_AUDIT_CLOCK_MISSING")
                self.assertEqual(client.export_chain_head()["entry_count"], 1)


class NativeTests(unittest.TestCase):
    def test_wcag_known_pairs_vector_and_scalar_paths(self):
        engine = enclave.NativeSimdOperatorEngine()
        for colour, expected in (((0, 0, 0), 21), ((255, 0, 0), 3.998476), ((128, 128, 128), 3.94944), ((0, 0, 255), 8.592471)):
            for count in (2, 9, 17):
                with self.subTest(colour=colour, count=count):
                    rgba = bytes((*colour, 255)) * (count - 1) + b"\xff" * 4
                    native = engine.run_full_10_operator_suite_native(rgba, rgba, count, 1)
                    reference = engine.run_reference_python_bbox_and_diff(rgba, rgba, count, 1)
                    actual = native["op_contrast_ratio_wcag_q16"]
                    self.assertEqual(actual, reference["wcag_q16"])
                    self.assertAlmostEqual(actual / 65536, expected, delta=0.0001)

    def test_10000_differential_cases(self):
        result = enclave.run_10000_differential_fuzz_suite(enclave.NativeSimdOperatorEngine())
        self.assertEqual(result["passed_cases"], 10000)
        self.assertEqual(result["divergences"], 0)


class SessionTests(unittest.TestCase):
    def server(self):
        tmp = tempfile.TemporaryDirectory(prefix="ev-reg-")
        self.addCleanup(tmp.cleanup)
        server = ElectricVisionServer(Path(tmp.name), "synthetic-test")
        self.addCleanup(server.close)
        return server

    def test_inspect_native_component_colours_and_png_opt_in(self):
        server = self.server()
        import electric_vision_mcp as mcp
        marks = [{"shape": "rect", "x": 1700, "y": 900, "width": 8, "height": 8, "fill": {"color": "#ff0000"}}]
        with patch.object(mcp, "_encode_png_bounded", wraps=mcp._encode_png_bounded) as encode:
            reply = server.handle_tool("eye.electric.inspect", {"marks": marks, "width": 1920, "height": 1080, "image": False})
            self.assertFalse(reply["isError"], reply)
            self.assertEqual(encode.call_count, 0)
            self.assertEqual(reply["structuredContent"]["component_colours"][0]["colours"][0]["hex"], "#f80808")
            reply = server.handle_tool("eye.electric.inspect", {"source_id": reply["structuredContent"]["source_id"], "source_version": reply["structuredContent"]["source_version"], "image": True})
            self.assertFalse(reply["isError"], reply)
            self.assertEqual(encode.call_count, 1)

    def test_stream_and_packet_no_png_until_requested(self):
        server = self.server()
        import electric_vision_mcp as mcp
        with patch.object(mcp, "_encode_png_bounded", wraps=mcp._encode_png_bounded) as encode:
            pushed = server.handle_tool("eye.electric.stream", {"action": "push", "frames": [dot(.2), dot(.3)], "image": False})
            self.assertFalse(pushed["isError"], pushed)
            packet = server.handle_tool("eye.electric.stream", {"action": "packet", "image": False})
            self.assertFalse(packet["isError"], packet)
            self.assertEqual(encode.call_count, 0)
            self.assertFalse(list(server.frames_dir.glob("*.png")))
            self.assertNotIn("rgba_path", packet["structuredContent"]["packet"]["frames"][0])
            packet = server.handle_tool("eye.electric.stream", {"action": "packet", "image": True})
            self.assertFalse(packet["isError"], packet)
            self.assertEqual(len([p for p in packet["content"] if p["type"] == "image"]), 2)
            self.assertEqual(len(list(server.frames_dir.glob("*.png"))), 2)
            server._reset_stream()
            self.assertFalse(list(server.frames_dir.iterdir()))

    def test_stale_decay_is_durable_and_sink_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory(prefix="ev-stale-") as tmp, ElectricVisionAgentSession(run_dir=tmp) as session:
            stream = session.create_continuous_stream(source_id="src_decay", width=128, height=128)
            stream.ingest_frame("v1", dot(.2), issue_observation_receipt=True)
            stream.ingest_frame("v2", dot(.3), issue_observation_receipt=True)
            events = session._audit_client.get_entries()
            stale = [e for e in events if e["event_type"] == "CONTINUOUS_OBSERVATION_DECAYED_TO_STALE"]
            self.assertEqual(len(stale), 1)
            self.assertEqual(json.loads(stale[0]["detail"])["decayed_version"], "v1")
            self.assertTrue(session._audit_client.verify_durable_file()["valid"])
            append = session._audit_client.append_entry
            def reject_decay(entry):
                if entry["event_type"] == "CONTINUOUS_OBSERVATION_DECAYED_TO_STALE":
                    raise RuntimeError("synthetic sink failure")
                return append(entry)
            with patch.object(session._audit_client, "append_entry", side_effect=reject_decay), self.assertRaisesRegex(RuntimeError, "sink failure"):
                stream.ingest_frame("v3", dot(.4), issue_observation_receipt=True)
            self.assertFalse(stream.check_receipt_freshness("v2")["fresh"])
            self.assertFalse(stream.check_receipt_freshness("v3")["fresh"])
            with self.assertRaisesRegex(RuntimeError, "AUDIT_FAILED_CLOSED"):
                stream.ingest_frame("v4", dot(.5))
            self.assertFalse([k for k in session._source_dimensions if k[0] == "src_decay"])
            receipts = [r for r in session._harness._observation_receipts.values() if r["captured_source_id"] == "src_decay"]
            self.assertGreaterEqual(len(receipts), 2)
            self.assertTrue(all(r.get("revoked") and r["receipt_signature"] is None for r in receipts))
            self.assertEqual(stream.release(), 0)
            self.assertFalse([k for k in session._source_dimensions if k[0] == "src_decay"])


if __name__ == "__main__":
    unittest.main()
