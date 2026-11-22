#!/usr/bin/env python3
"""
Unit, Scenario & Remediation Gate Verification Suite for the Disposable Electric Eye
Fixture Harness.
Verifies all hard boundaries, operators, contracts, E1-E10 scenarios, C1-C9 pass
criteria, <50 ms acceptance thresholds, and Remediation Gates 1-5.
"""

from __future__ import annotations

import copy
import json
import os
import random
import unittest

from electric_eye_harness import (
    CLOSED_FORM_OPERATORS,
    DISPOSABLE_FIXTURE_STATIC_HMAC_KEY,
    MAX_CACHE_ENTRIES,
    MAX_MARKS_PER_SOURCE,
    MAX_REPORTED_REGIONS,
    REFUSAL_CODE_SEMANTIC,
    REFUSAL_REASON_SEMANTIC,
    VALID_RENDER_TYPES,
    VALID_SOURCE_KINDS,
    VALID_UNCERTAINTY_STATES,
    DrawingMark,
    ElectricEyeFixtureHarness,
    InMemoryAppendOnlyAuditSink,
    SimulatedPrivilegedGateBoundary,
    assert_no_raw_buffer_leakage,
    canonical_json,
    compute_replay_digest,
    run_all_scenarios,
)


class TestElectricEyeFixtureHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = ElectricEyeFixtureHarness()
        self.marks_v1 = [
            DrawingMark("sq1", "rect", 16, 16, 63, 63, (0, 255, 170, 255)),
            DrawingMark("mk1", "rect", 96, 96, 103, 103, (255, 180, 0, 255)),
        ]
        self.marks_v2 = [
            DrawingMark("sq1_shifted", "rect", 26, 16, 73, 63, (0, 255, 170, 255)),
            DrawingMark("mk1", "rect", 96, 96, 103, 103, (255, 180, 0, 255)),
        ]
        self.src_v1 = self.harness.register_verified_source(
            "src_alpha", "v1", "verified_pixels", self.marks_v1
        )
        self.src_v2 = self.harness.mutate_canvas(
            "src_alpha", "v1", "v2", self.marks_v2
        )

    def test_layer1_verified_source_immutability_and_kinds(self) -> None:
        with self.assertRaises(ValueError):
            self.harness.register_verified_source("src_alpha", "v1", "verified_pixels", self.marks_v2)
        with self.assertRaises(ValueError):
            self.harness.mutate_canvas("src_alpha", "v1", "v1", self.marks_v2)

        for idx, kind in enumerate(sorted(VALID_SOURCE_KINDS)):
            desc = self.harness.register_verified_source(
                f"src_kind_{idx}", "v1", kind, self.marks_v1
            )
            self.assertEqual(desc["source_kind"], kind)
            self.assertFalse(desc["observed"])
            self.assertFalse(desc["observation_claim"])
            assert_no_raw_buffer_leakage(desc)

    def test_layer2_all_ten_operators_and_replay_digest(self) -> None:
        self.assertEqual(len(CLOSED_FORM_OPERATORS), 10)
        bbox_env = self.harness.execute_measurement(
            "measure_bbox",
            {"selector": {"type": "colour_match", "rgba": [0, 255, 170, 255], "tolerance": 0}},
            source_id="src_alpha",
            source_version="v1",
        )
        self.assertEqual(bbox_env["result"]["x_min"], 16)
        self.assertEqual(bbox_env["result"]["y_min"], 16)
        self.assertEqual(bbox_env["result"]["x_max"], 63)
        self.assertEqual(bbox_env["result"]["y_max"], 63)
        self.assertEqual(bbox_env["result"]["width"], 48)
        self.assertEqual(bbox_env["result"]["height"], 48)
        self.assertEqual(bbox_env["result"]["matching_pixels"], 2304)

        recomputed = compute_replay_digest(
            source_id=bbox_env["source_id"],
            source_version=bbox_env["source_version"],
            secondary_source=bbox_env["secondary_source"],
            operator_name=bbox_env["operator_name"],
            operator_version=bbox_env["operator_version"],
            parameters=bbox_env["parameters"],
            result=bbox_env["result"],
        )
        self.assertEqual(bbox_env["replay_digest"], recomputed)

    def test_refusal_contract_strict_compliance(self) -> None:
        for qtype in ("semantic_meaning", "aesthetic_judgment", "agent_intent"):
            env = self.harness.execute_measurement(
                qtype,
                {"query_type": qtype, "question": f"Test {qtype}?"},
                source_id="src_alpha",
                source_version="v1",
            )
            self.assertEqual(env["status"], "refused")
            self.assertEqual(env["result"], "unknown")
            self.assertEqual(env["refusal_code"], REFUSAL_CODE_SEMANTIC)
            self.assertEqual(env["refusal_reason"], REFUSAL_REASON_SEMANTIC)

    def test_layer3_derived_representations(self) -> None:
        for rep_type in (
            "change_mask",
            "difference_map",
            "motion_summary",
            "colour_delta",
            "spatial_relationship",
        ):
            rep = self.harness.compute_derived_representation(
                rep_type, ("src_alpha", "v1"), ("src_alpha", "v2")
            )
            self.assertEqual(rep["epistemic_label"], "derived")
            self.assertEqual(len(rep["source_refs"]), 2)
            self.assertTrue(bool(rep["rationale"]))
            assert_no_raw_buffer_leakage(rep)

    def test_layer4_semantic_boundary_validation(self) -> None:
        for valid_label in ("derived", "inferred", "uncertain"):
            checked = self.harness.validate_external_semantic_annotation({
                "epistemic_label": valid_label,
                "confidence": "low",
            })
            self.assertEqual(checked["epistemic_label"], valid_label)
            self.assertFalse(checked["observed"])
            self.assertFalse(checked["is_measurement"])

        with self.assertRaises(ValueError):
            self.harness.validate_external_semantic_annotation({
                "epistemic_label": "verified",
                "confidence": "high",
            })
        with self.assertRaises(PermissionError):
            self.harness.validate_external_semantic_annotation({
                "epistemic_label": "inferred",
                "confidence": "medium",
                "observed": True,
            })

    def test_layer5_ghost_isolation_and_observation_gate(self) -> None:
        ghost_marks = [DrawingMark("g1", "rect", 80, 80, 90, 90, (255, 0, 255, 255))]
        ghost_rnd = self.harness.create_electric_render(
            render_type="predicted",
            source="inference",
            confidence="medium",
            uncertainty_state="predicted",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_alpha", "ref_version": "v2"}],
            ghost_marks=ghost_marks,
        )
        self.assertFalse(ghost_rnd["observed"])
        self.assertEqual(ghost_rnd["layer_isolation"], "ghost_isolation_plane")

        sample = self.harness.execute_measurement(
            "sample_colour", {"x": 85, "y": 85}, source_id="src_alpha", source_version="v2"
        )
        self.assertEqual(sample["result"]["rgba"], [0, 0, 0, 0])

        with self.assertRaises(PermissionError):
            self.harness.issue_observation_receipt(
                render_id=ghost_rnd["render_id"],
                captured_source_id="src_alpha",
                captured_source_version="v2",
            )

    def test_gate1_gov01_strict_provenance_and_render_type_binding(self) -> None:
        # Register a legitimate captured_frame source
        self.harness.register_verified_source("src_cap_legit", "v1", "captured_frame", self.marks_v1)
        self.harness.register_verified_source("src_cap_other", "v1", "captured_frame", self.marks_v2)

        # 1. A 'derived' render cannot receive an observation receipt even for a captured_frame
        rnd_derived = self.harness.create_electric_render(
            render_type="derived",
            source="measurement",
            confidence="high",
            uncertainty_state="derived",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_cap_legit", "ref_version": "v1"}],
        )
        with self.assertRaises(PermissionError):
            self.harness.issue_observation_receipt(
                render_id=rnd_derived["render_id"],
                captured_source_id="src_cap_legit",
                captured_source_version="v1",
            )

        # 2. An 'observed' render bound to src_cap_legit@v1 cannot receive a receipt for src_cap_other@v1
        rnd_obs = self.harness.create_electric_render(
            render_type="observed",
            source="verified_pixels",
            confidence="high",
            uncertainty_state="verified",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_cap_legit", "ref_version": "v1"}],
        )
        with self.assertRaises(PermissionError):
            self.harness.issue_observation_receipt(
                render_id=rnd_obs["render_id"],
                captured_source_id="src_cap_other",
                captured_source_version="v1",
            )

        # 3. Exact match succeeds
        rcpt = self.harness.issue_observation_receipt(
            render_id=rnd_obs["render_id"],
            captured_source_id="src_cap_legit",
            captured_source_version="v1",
        )
        promoted = self.harness.promote_render_to_observed(rnd_obs["render_id"], rcpt)
        self.assertEqual(promoted["status"], "ok")
        self.assertTrue(promoted["observed"])

    def test_gate2_sec01_sec02_closure_isolation(self) -> None:
        # Verify neither _vault nor _ledger_hmac_key exist on harness.__dict__
        self.assertNotIn("_vault", self.harness.__dict__)
        self.assertNotIn("_ledger_hmac_key", self.harness.__dict__)

    def test_gate3_sec03_prv01_api01_schemas_and_redaction(self) -> None:
        # Invalid source_id regex rejected
        with self.assertRaises(ValueError):
            self.harness.register_verified_source(
                "invalid/path/secret", "v1", "verified_pixels", self.marks_v1
            )

        # Extra semantic key inside valid operator is refused as semantic overreach
        sem_extra = self.harness.execute_measurement(
            "measure_bbox",
            {"selector": {"type": "non_transparent"}, "semantic_hypothesis": "is this a house?"},
            source_id="src_alpha",
            source_version="v1",
        )
        self.assertEqual(sem_extra["status"], "refused")
        self.assertEqual(sem_extra["refusal_code"], REFUSAL_CODE_SEMANTIC)

        # Extra unknown non-semantic key is rejected with ERR_INVALID_PARAMETERS
        unknown_extra = self.harness.execute_measurement(
            "measure_dimensions",
            {"unexpected_flag": 123},
            source_id="src_alpha",
            source_version="v1",
        )
        self.assertEqual(unknown_extra["status"], "error")
        self.assertEqual(unknown_extra["refusal_code"], "ERR_INVALID_PARAMETERS")

        # Out-of-bounds sample_colour returns deterministic error envelope instead of crashing
        oob = self.harness.execute_measurement(
            "sample_colour",
            {"x": 999, "y": -5},
            source_id="src_alpha",
            source_version="v1",
        )
        self.assertEqual(oob["status"], "error")
        self.assertEqual(oob["refusal_code"], "ERR_INVALID_PARAMETERS")

        # Prompt redaction mode hashes free-text prompts instead of echoing raw strings
        redacting_harness = ElectricEyeFixtureHarness(redact_refused_prompts=True)
        redacting_harness.register_verified_source("src_red", "v1", "verified_pixels", self.marks_v1)
        ref_env = redacting_harness.execute_measurement(
            "semantic_meaning",
            {"query_type": "semantic_meaning", "question": "Sensitive secret prompt text"},
            source_id="src_red",
            source_version="v1",
        )
        self.assertNotIn("question", ref_env["parameters"])
        self.assertIn("question_sha256", ref_env["parameters"])
        self.assertEqual(ref_env["parameters"]["question_length"], 28)

    def test_gate4_aud01_acc01_audit_ledger_and_dynamic_staleness(self) -> None:
        # Render bound to src_alpha@v1 is already behind src_alpha@v2 by 1 version
        rnd_v1 = self.harness.create_electric_render(
            render_type="derived",
            source="measurement",
            confidence="high",
            uncertainty_state="derived",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_alpha", "ref_version": "v1"}],
        )
        evaluated = self.harness.get_render(rnd_v1["render_id"], evaluate_staleness=True)
        self.assertEqual(evaluated["uncertainty_state"], "stale")
        self.assertEqual(evaluated["border_pattern"], "⋀⋁⋀⋁")
        self.assertIn("BEHIND BY 1 VERSION", evaluated["uncertainty_badge_text"])

        # Trigger blocked observation attempts and verify hash-chained security audit ledger
        self.harness.promote_render_to_observed(rnd_v1["render_id"], receipt=None)
        ledger = self.harness.get_security_audit_ledger()
        self.assertGreaterEqual(len(ledger), 1)
        self.assertTrue(self.harness.verify_security_audit_ledger())

    def test_gate5_tst01_line_dot_4conn_metrics_and_tolerances(self) -> None:
        # Test line and dot primitives and 4-connectivity vs 8-connectivity
        diag_marks = [
            DrawingMark("d1", "dot", 10, 10, 10, 10, (255, 0, 0, 255)),
            DrawingMark("d2", "dot", 11, 11, 11, 11, (255, 0, 0, 255)),
            DrawingMark("ln1", "line", 20, 20, 25, 20, (0, 255, 0, 255)),
        ]
        self.harness.register_verified_source("src_prim", "v1", "drawing_state", diag_marks)

        # Under 8-connectivity, d1(10,10) and d2(11,11) form 1 object (+ ln1 = 2 objects)
        obj_8 = self.harness.execute_measurement(
            "count_objects", {"connectivity": 8, "min_area_px": 1}, source_id="src_prim", source_version="v1"
        )
        self.assertEqual(obj_8["result"]["object_count"], 2)

        # Under 4-connectivity, d1(10,10) and d2(11,11) are disjoint (+ ln1 = 3 objects)
        obj_4 = self.harness.execute_measurement(
            "count_objects", {"connectivity": 4, "min_area_px": 1}, source_id="src_prim", source_version="v1"
        )
        self.assertEqual(obj_4["result"]["object_count"], 3)

        # Test Manhattan and Chebyshev distance metrics
        d_man = self.harness.execute_measurement(
            "measure_distance", {"p1": {"x": 10, "y": 10}, "p2": {"x": 13, "y": 14}, "metric": "manhattan"}
        )
        self.assertEqual(d_man["result"]["distance"], 7.0)
        d_cheb = self.harness.execute_measurement(
            "measure_distance", {"p1": {"x": 10, "y": 10}, "p2": {"x": 13, "y": 14}, "metric": "chebyshev"}
        )
        self.assertEqual(d_cheb["result"]["distance"], 4.0)

        # Test non-zero colour tolerance
        cnt_tol = self.harness.execute_measurement(
            "count_pixels",
            {"predicate": {"type": "colour_match", "rgba": [250, 5, 5, 255], "tolerance": 10}},
            source_id="src_prim",
            source_version="v1",
        )
        self.assertEqual(cnt_tol["result"]["matching_pixels"], 2)

    def test_full_e1_to_e10_report_and_thresholds(self) -> None:
        report = run_all_scenarios()
        self.assertTrue(report["all_scenarios_passed"])
        self.assertTrue(report["all_criteria_passed"])

        metrics = report["metrics"]
        self.assertEqual(metrics["deterministic_replay_equality"], 1.0)
        self.assertEqual(metrics["provenance_completeness"], 1.0)
        self.assertEqual(metrics["false_observation_count"], 0)
        self.assertEqual(metrics["blocked_observation_attempts"], 3)
        self.assertLess(metrics["cpu_time"], 50.0)
        self.assertLess(metrics["wall_latency"], 50.0)
        self.assertGreaterEqual(metrics["source_count"], 6)
        self.assertGreaterEqual(metrics["measurement_count"], 25)
        self.assertEqual(metrics["derived_representation_count"], 5)
        self.assertEqual(set(metrics["render_count_by_type"].keys()), set(VALID_RENDER_TYPES))
        self.assertGreater(metrics["output_size"], 0)

        for st in VALID_UNCERTAINTY_STATES:
            self.assertIn(
                st,
                {
                    o["uncertainty_state"]
                    for o in report["scenarios"]["E7_uncertainty_glow"]["no_colour_overlays"]
                },
            )

        assert_no_raw_buffer_leakage(report)

    def test_next_gate1_simulated_boundary_and_non_static_key_enforcement(self) -> None:
        # 1. Static fixture key is strictly rejected in any non-disposable profile
        with self.assertRaises(PermissionError):
            ElectricEyeFixtureHarness(
                profile="non_fixture",
                hmac_key_provider=lambda: DISPOSABLE_FIXTURE_STATIC_HMAC_KEY,
            )
        with self.assertRaises(PermissionError):
            SimulatedPrivilegedGateBoundary(
                hmac_key_provider=lambda: DISPOSABLE_FIXTURE_STATIC_HMAC_KEY
            )
        with self.assertRaises(PermissionError):
            SimulatedPrivilegedGateBoundary(
                hmac_key_provider=lambda: b"short-weak-key"
            )

        # 2. SimulatedPrivilegedGateBoundary enforces capability token, provenance binding,
        # signature verification, promotion gating, and false_observation_count == 0
        priv_token = "cap_tok_privileged_capture_service_01"
        boundary = SimulatedPrivilegedGateBoundary(
            hmac_key_provider=lambda: b"kms-injected-32-byte-secret-key-material-v1",
            capture_service_token=priv_token,
        )
        desc = boundary.get_boundary_descriptor()
        self.assertEqual(desc["boundary_mode"], "simulated_in_memory_rpc")
        self.assertFalse(desc["claims_real_process_isolation"])

        pilot_harness = ElectricEyeFixtureHarness(
            profile="controlled_pilot_simulation",
            simulated_gate_boundary=boundary,
        )
        pilot_harness.register_verified_source("src_cap_sim", "v1", "captured_frame", self.marks_v1)

        rnd_obs = pilot_harness.create_electric_render(
            render_type="observed",
            source="verified_pixels",
            confidence="high",
            uncertainty_state="verified",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_cap_sim", "ref_version": "v1"}],
        )

        # Calling issue_observation_receipt without capability token is rejected
        with self.assertRaises(PermissionError):
            pilot_harness.issue_observation_receipt(
                render_id=rnd_obs["render_id"],
                captured_source_id="src_cap_sim",
                captured_source_version="v1",
                caller_capability_token=None,
            )
        # Calling with wrong capability token is rejected
        with self.assertRaises(PermissionError):
            pilot_harness.issue_observation_receipt(
                render_id=rnd_obs["render_id"],
                captured_source_id="src_cap_sim",
                captured_source_version="v1",
                caller_capability_token="cap_tok_forged",
            )

        # Calling with valid capability token issues signed receipt
        valid_rcpt = pilot_harness.issue_observation_receipt(
            render_id=rnd_obs["render_id"],
            captured_source_id="src_cap_sim",
            captured_source_version="v1",
            caller_capability_token=priv_token,
        )

        # Tampered signature is rejected by simulated boundary verification
        tampered_rcpt = copy.deepcopy(valid_rcpt)
        tampered_rcpt["receipt_signature"] = "hmac-sha256:" + ("f" * 64)
        rej = pilot_harness.promote_render_to_observed(rnd_obs["render_id"], tampered_rcpt)
        self.assertEqual(rej["status"], "rejected")
        self.assertFalse(rej["observed"])
        self.assertEqual(pilot_harness.false_observation_count, 0)

        # Valid receipt promotes render cleanly
        ok = pilot_harness.promote_render_to_observed(rnd_obs["render_id"], valid_rcpt)
        self.assertEqual(ok["status"], "ok")
        self.assertTrue(ok["observed"])
        self.assertEqual(pilot_harness.false_observation_count, 0)

    def test_next_gate2_default_privacy_redaction_and_opaque_source_ids(self) -> None:
        # 1. Default non-fixture harness redacts refused prompts automatically
        default_harness = ElectricEyeFixtureHarness()
        default_harness.register_verified_source("src_priv_01", "v1", "verified_pixels", self.marks_v1)
        ref = default_harness.execute_measurement(
            "semantic_meaning",
            {"query_type": "semantic_meaning", "question": "Does this contain private user text?"},
            source_id="src_priv_01",
            source_version="v1",
        )
        self.assertEqual(ref["status"], "refused")
        self.assertNotIn("question", ref["parameters"])
        self.assertIn("question_sha256", ref["parameters"])

        # 2. Attempting redact_refused_prompts=False outside disposable_fixture profile is forbidden
        with self.assertRaises(ValueError):
            ElectricEyeFixtureHarness(redact_refused_prompts=False, profile="non_fixture")

        # 3. Source IDs encoding filenames, extensions, paths, or user identifiers are rejected
        unsafe_ids = [
            "src_user_alice",
            "src_email_contact",
            "src_file_screenshot",
            "src_capture_png",
            "src_home_charlie",
            "src_ssn_01",
            "src_secret_token",
            "src_20261029_frame",
            "src_UpperCase_01",
            "src_with.dot",
            "src_with-dash",
        ]
        for bad_id in unsafe_ids:
            with self.assertRaises(ValueError, msg=f"Expected {bad_id} to be rejected"):
                default_harness.register_verified_source(bad_id, "v1", "verified_pixels", self.marks_v1)

        # 4. Verify E1-E10 scenarios in run_all_scenarios() remain 100% byte-identical to baseline
        report_now = run_all_scenarios()
        baseline_path = os.path.join(os.path.dirname(__file__), "electric_eye_report_before.json")
        if os.path.exists(baseline_path):
            with open(baseline_path, "r", encoding="utf-8") as f:
                report_before = json.load(f)
            self.assertEqual(
                canonical_json(report_now["scenarios"]),
                canonical_json(report_before["scenarios"]),
            )

    def test_next_gate3_dual_clock_audit_ledger_sink_and_tamper_detection(self) -> None:
        sink = InMemoryAppendOnlyAuditSink()
        clock_ticks = iter([
            "2026-10-29T23:25:01.000000Z",
            "2026-10-29T23:25:02.000000Z",
            "2026-10-29T23:25:03.000000Z",
        ])
        h = ElectricEyeFixtureHarness(
            profile="controlled_pilot_simulation",
            audit_sink=sink,
            utc_clock_fn=lambda: next(clock_ticks),
        )
        h.register_verified_source("src_aud_01", "v1", "verified_pixels", self.marks_v1)

        # Trigger 1: semantic query refusal entry
        h.execute_measurement(
            "semantic_meaning",
            {"query_type": "semantic_meaning", "question": "What is the mood?"},
            source_id="src_aud_01",
            source_version="v1",
        )
        # Trigger 2: blocked unreceipted observation promotion entry
        rnd = h.create_electric_render(
            render_type="derived",
            source="measurement",
            confidence="high",
            uncertainty_state="derived",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_aud_01", "ref_version": "v1"}],
        )
        h.promote_render_to_observed(rnd["render_id"], receipt=None)

        # Verify ledger & external sink continuity, dual clocks, and sequence ordering
        self.assertTrue(h.verify_security_audit_ledger())
        summary = h.export_audit_chain_summary()
        sink_head = sink.export_chain_head()
        self.assertEqual(summary["entry_count"], 2)
        self.assertEqual(summary["head_seq"], 2)
        self.assertEqual(summary["chain_head_digest"], sink_head["chain_head_digest"])
        self.assertEqual(summary["head_utc_wall_timestamp"], "2026-10-29T23:25:02.000000Z")
        self.assertEqual(summary["head_logical_clock"], "seq:00000002@2026-10-29T22:35:26.000Z")

        entries = h.get_security_audit_ledger()
        self.assertEqual(entries[0]["event_type"], "semantic_query_refused")
        self.assertEqual(entries[1]["event_type"], "unreceipted_promotion_blocked")

        # Verify tamper detection on both internal ledger and append-only sink
        h._audit_ledger[0]["detail"] = "tampered detail"
        self.assertFalse(h.verify_security_audit_ledger())
        with self.assertRaises(ValueError):
            sink.append_entry(h._audit_ledger[0])

    def test_next_gate4_default_staleness_nested_schemas_fuzz_and_boundaries(self) -> None:
        # 1. Default-on staleness detection across v1 -> v2 -> v3 + explicit opt-out
        h = ElectricEyeFixtureHarness()
        h.register_verified_source("src_stale_01", "v1", "verified_pixels", self.marks_v1)
        rnd = h.create_electric_render(
            render_type="derived",
            source="measurement",
            confidence="high",
            uncertainty_state="derived",
            provenance_refs=[{"ref_type": "source", "ref_id": "src_stale_01", "ref_version": "v1"}],
        )
        # While v1 is latest, default get_render() returns 'derived'
        self.assertEqual(h.get_render(rnd["render_id"])["uncertainty_state"], "derived")

        # Advance to v2: default get_render() and list_agent_visible_renders() automatically mark stale
        h.mutate_canvas("src_stale_01", "v1", "v2", self.marks_v2)
        stale_v2 = h.get_render(rnd["render_id"])
        self.assertEqual(stale_v2["uncertainty_state"], "stale")
        self.assertEqual(stale_v2["border_pattern"], "⋀⋁⋀⋁")
        self.assertIn("BEHIND BY 1 VERSION", stale_v2["uncertainty_badge_text"])
        feed_v2 = h.list_agent_visible_renders()
        self.assertEqual(feed_v2[0]["uncertainty_state"], "stale")

        # Advance to v3: badge updates to BEHIND BY 2 VERSIONS
        h.mutate_canvas("src_stale_01", "v2", "v3", self.marks_v1)
        stale_v3 = h.get_render(rnd["render_id"])
        self.assertIn("BEHIND BY 2 VERSIONS", stale_v3["uncertainty_badge_text"])

        # Explicit opt-out preserves creation-time state
        raw_stored = h.get_render(rnd["render_id"], evaluate_staleness=False)
        self.assertEqual(raw_stored["uncertainty_state"], "derived")

        # 2. Closed nested parameter key allowlists reject unknown inner keys
        bad_nested_cases = [
            ("measure_bbox", {"selector": {"type": "non_transparent", "injected_key": 1}}),
            ("count_pixels", {"predicate": {"type": "non_transparent", "extra_inner": True}}),
            ("measure_distance", {"p1": {"x": 1, "y": 2, "z": 3}, "p2": {"x": 4, "y": 5}}),
            ("measure_angle", {"p1": {"x": 1}, "p2": {"x": 4, "y": 5}}),
            ("sample_colour", {"x": True, "y": 10}),
        ]
        for op_name, bad_params in bad_nested_cases:
            env = h.execute_measurement(
                op_name, bad_params, source_id="src_stale_01", source_version="v1"
            )
            self.assertEqual(env["status"], "error", msg=f"Failed on {op_name}: {bad_params}")
            self.assertEqual(env["refusal_code"], "ERR_INVALID_PARAMETERS")

        # 3. Property-based fuzz testing across all 10 operators (deterministic seed)
        rng = random.Random(20261029)
        fuzz_values = [
            None,
            True,
            False,
            -999,
            128,
            500,
            3.1415,
            "bad_str",
            [1, 2],
            {"unexpected_nested": "val"},
            {"type": "colour_match", "rgba": [0, 255, 999, 255]},
            {"x": -1, "y": 200},
        ]
        for op_name in sorted(CLOSED_FORM_OPERATORS):
            for i in range(20):
                fuzzed_params = {f"fuzz_key_{i}": rng.choice(fuzz_values)}
                res = h.execute_measurement(
                    op_name,
                    fuzzed_params,
                    source_id="src_stale_01",
                    source_version="v1",
                )
                self.assertEqual(res["status"], "error")
                self.assertEqual(res["refusal_code"], "ERR_INVALID_PARAMETERS")
                assert_no_raw_buffer_leakage(res)

        # 4. Boundary tests:
        # (a) MAX_MARKS_PER_SOURCE = 1024 allowed, 1025 rejected
        marks_1024 = [
            DrawingMark(f"m_{i}", "dot", i & 127, (i >> 7) & 127, i & 127, (i >> 7) & 127, (255, 0, 0, 255))
            for i in range(MAX_MARKS_PER_SOURCE)
        ]
        desc_1024 = h.register_verified_source("src_max_marks", "v1", "drawing_state", marks_1024)
        self.assertEqual(desc_1024["mark_count"], MAX_MARKS_PER_SOURCE)
        marks_1025 = marks_1024 + [DrawingMark("m_overflow", "dot", 0, 0, 0, 0, (255, 0, 0, 255))]
        with self.assertRaises(ValueError):
            h.register_verified_source("src_over_marks", "v1", "drawing_state", marks_1025)

        # (b) MAX_REPORTED_REGIONS = 512 truncation on >512 disjoint components
        # Place 600 non-adjacent dots at even coordinates (2*x, 2*y) so 8-connectivity keeps them disjoint
        disjoint_600 = []
        for idx in range(600):
            gx = (idx % 30) * 2
            gy = (idx // 30) * 2
            disjoint_600.append(DrawingMark(f"d_{idx}", "dot", gx, gy, gx, gy, (0, 200, 255, 255)))
        h.register_verified_source("src_many_reg", "v1", "verified_pixels", disjoint_600)
        many_obj = h.execute_measurement(
            "count_objects",
            {"connectivity": 8, "min_area_px": 1},
            source_id="src_many_reg",
            source_version="v1",
        )
        self.assertEqual(many_obj["status"], "ok")
        self.assertEqual(many_obj["result"]["object_count"], 600)
        self.assertEqual(len(many_obj["result"]["objects"]), MAX_REPORTED_REGIONS)
        assert_no_raw_buffer_leakage(many_obj)

        # (c) MAX_CACHE_ENTRIES = 64 LRU eviction
        for c_idx in range(70):
            sid = f"src_lru_{c_idx:02d}"
            m_lru = [DrawingMark(f"mk_{c_idx}", "dot", c_idx % 128, (c_idx * 3) % 128, c_idx % 128, (c_idx * 3) % 128, (10, 20, 30, 255))]
            h.register_verified_source(sid, "v1", "verified_pixels", m_lru)
            h.execute_measurement("count_objects", {"connectivity": 8, "min_area_px": 1}, source_id=sid, source_version="v1")
        self.assertEqual(len(h._cc_cache), MAX_CACHE_ENTRIES)
        self.assertEqual(h._get_raster_cache_size(), MAX_CACHE_ENTRIES)
        self.assertNotIn(("src_lru_00", "v1", 8), h._cc_cache)
        # Re-querying evicted entry recomputes deterministically
        evicted_rerun = h.execute_measurement(
            "count_objects", {"connectivity": 8, "min_area_px": 1}, source_id="src_lru_00", source_version="v1"
        )
        self.assertEqual(evicted_rerun["status"], "ok")
        self.assertEqual(evicted_rerun["result"]["object_count"], 1)

    def test_exec_gates_1_2_3_out_of_process_daemons_and_verification(self) -> None:
        from exec_gate_daemons import run_exec_gate_verification_suite

        suite_res = run_exec_gate_verification_suite()
        self.assertTrue(suite_res["distinct_os_pids_verified"])
        self.assertTrue(suite_res["daemons_terminated_cleanly"])

        # EXEC-GATE-1 assertions
        g1 = suite_res["exec_gate_1"]
        self.assertFalse(g1["has_local_vault_or_hmac_keys_in_harness"])
        self.assertTrue(g1["so_peercred_enforced"])
        self.assertTrue(g1["peer_auth_rejected_unauthorized_uid"])
        self.assertTrue(g1["prohibited_method_rejected"])
        self.assertTrue(g1["rpc_nonce_replay_blocked"])
        self.assertTrue(g1["missing_capability_token_rejected"])
        self.assertTrue(g1["capability_token_replay_blocked"])
        self.assertTrue(g1["forged_receipt_rejected"])
        self.assertTrue(g1["valid_receipt_promoted"])
        self.assertEqual(g1["false_observation_count"], 0)
        self.assertEqual(g1["deterministic_replay_equality"], 1.0)
        self.assertEqual(g1["provenance_completeness"], 1.0)
        self.assertTrue(g1["ghost_isolation_intact"])
        self.assertTrue(g1["diagnostic_hidden_by_default"])

        # EXEC-GATE-2 assertions
        g2 = suite_res["exec_gate_2"]
        self.assertTrue(g2["durable_file_verified"]["valid"])
        self.assertGreater(g2["audit_head"]["fsync_count"], 0)
        self.assertTrue(g2["fail_closed_verified"])
        self.assertTrue(g2["raw_or_pii_rejected_by_sink"])
        self.assertTrue(g2["tamper_detected_on_disk"])
        self.assertTrue(g2["refused_prompt_redacted"])

        # EXEC-GATE-3 assertions
        g3 = suite_res["exec_gate_3"]
        self.assertTrue(g3["non_synthetic_rejected"])
        self.assertTrue(g3["unsafe_id_rejected"])
        self.assertTrue(g3["unsafe_mark_rejected"])
        self.assertFalse(g3["normal_run_halted"])
        self.assertIsNone(g3["normal_run_triggered_ks"])
        self.assertTrue(g3["fault_injection_ks_triggered"])
        self.assertTrue(g3["halt_enforced_on_harness"])
        self.assertTrue(g3["vault_locked_after_ks"])
        self.assertTrue(g3["forensic_evidence_preserved"])

    def test_prod_gates_1_2_3_4_enclave_and_verification(self) -> None:
        from prod_gate_enclave import run_prod_gate_implementation_and_verification_suite

        res = run_prod_gate_implementation_and_verification_suite()
        self.assertEqual(res["implementation_decision"], "implementation_verified")

        # PROD-GATE-1
        pg1 = res["prod_gate_1_implementation_status"]
        self.assertEqual(pg1["status"], "VERIFIED")
        self.assertTrue(pg1["hsm_kms_key_custody"]["raw_key_export_blocked"])
        self.assertTrue(pg1["hsm_kms_key_custody"]["unauthorized_principal_blocked"])
        self.assertTrue(pg1["hsm_kms_key_custody"]["sign_and_verify_ok"])
        self.assertTrue(pg1["hsm_kms_key_custody"]["key_rotation_verified"])
        self.assertTrue(pg1["hsm_kms_key_custody"]["fail_closed_after_zeroization"])
        self.assertTrue(pg1["fixed_short_socket_paths"]["all_under_32_bytes"])

        # PROD-GATE-2
        pg2 = res["prod_gate_2_implementation_status"]
        self.assertEqual(pg2["status"], "VERIFIED")
        self.assertEqual(pg2["differential_fuzz_10000"]["passed_cases"], 10000)
        self.assertEqual(pg2["differential_fuzz_10000"]["divergences"], 0)
        self.assertTrue(pg2["multi_resolution_benchmarks"]["256x256"]["within_latency_target"])
        self.assertTrue(pg2["multi_resolution_benchmarks"]["512x512"]["within_latency_target"])
        self.assertTrue(pg2["multi_resolution_benchmarks"]["1080p_1920x1080"]["within_latency_target"])
        self.assertTrue(pg2["rollback_on_perturbation_drill"]["fallback_triggered"])

        # PROD-GATE-3
        pg3 = res["prod_gate_3_implementation_status"]
        self.assertEqual(pg3["status"], "VERIFIED")
        self.assertEqual(pg3["pii_canary_1000_benchmark"]["redacted_frames"], 1000)
        self.assertEqual(pg3["pii_canary_1000_benchmark"]["false_negatives"], 0)
        self.assertTrue(pg3["ks9_unconsented_capture_blocked"])
        self.assertTrue(pg3["ks9_full_desktop_capture_blocked"])
        self.assertTrue(pg3["ks10_canary_egress_halt_and_purge_verified"])
        self.assertEqual(pg3["wcag_and_screen_reader_validation"]["classification_accuracy"], 1.0)
        self.assertEqual(pg3["wcag_and_screen_reader_validation"]["false_observation_beliefs"], 0)

        # PROD-GATE-4
        pg4 = res["prod_gate_4_implementation_status"]
        self.assertEqual(pg4["status"], "VERIFIED")
        self.assertEqual(pg4["final_verification"]["verified_signatures"], 6)
        self.assertTrue(pg4["change_control_tamper_rejection_drill"]["rejected_as_expected"])
        self.assertTrue(pg4["unilateral_rollback_drill"]["within_60s_sla"])

    def test_electric_vision_agent_bridge_multi_resolution_and_vocabularies(self) -> None:
        from electric_vision_agent_bridge import (
            SUPPORTED_RESOLUTIONS,
            ElectricVisionAgentSession,
        )

        prism_marks = [
            {"k": "fill", "pts": [[0.1, 0.1], [0.4, 0.1], [0.25, 0.4]], "c": [0, 229, 255], "a": 1.0},
            {"k": "glow", "x": 0.5, "y": 0.5, "r": 0.12, "c": [255, 179, 0], "a": 0.9},
            {"k": "ring", "x": 0.7, "y": 0.7, "r": 0.1, "w": 0.02, "c": [213, 0, 249], "a": 1.0},
            {"k": "line", "pts": [[0.15, 0.8], [0.85, 0.8]], "w": 0.015, "c": [20, 120, 220], "a": 1.0},
            {"k": "spray", "x": 0.3, "y": 0.6, "r": 0.08, "n": 16, "s": 0.01, "c": [0, 255, 170], "a": 1.0},
        ]
        cyclopsdraw_marks = [
            {"shape": "rect", "x": 16, "y": 16, "width": 40, "height": 32, "fill": {"type": "solid", "color": "#1478dc"}},
            {"shape": "circle", "center": {"x": 96, "y": 64}, "radius": 14, "fill": {"type": "solid", "color": "#00ffaa"}},
            {"shape": "line", "from": {"x": 10, "y": 100}, "to": {"x": 110, "y": 100}, "stroke": {"color": "#ffb300", "width": 2}},
            {"shape": "polygon", "points": [{"x": 60, "y": 10}, {"x": 80, "y": 10}, {"x": 70, "y": 30}], "fill": {"type": "solid", "color": "#ff5252"}},
        ]

        with ElectricVisionAgentSession(default_resolution=(256, 256)) as session:
            # 1. Verify all 4 resolutions (128x128, 256x256, 512x512, 1920x1080)
            for idx, (w, h) in enumerate(SUPPORTED_RESOLUTIONS):
                sid = f"src_res_{idx}"
                reg = session.register_synthetic_canvas(
                    sid,
                    "v1",
                    prism_marks,
                    width=w,
                    height=h,
                    source_kind="captured_frame",
                    mirror="xy",
                )
                self.assertEqual(reg["native_resolution"]["width"], w)
                self.assertEqual(reg["native_resolution"]["height"], h)
                assert_no_raw_buffer_leakage(reg)

                insp1 = session.inspect_canvas(sid, "v1", issue_observation_receipt=(idx == 0))
                insp2 = session.inspect_canvas(sid, "v1", issue_observation_receipt=False)
                self.assertEqual(insp1["native_replay_digest"], insp2["native_replay_digest"])
                self.assertEqual(insp1["render"]["observed"], (idx == 0))
                if idx == 0:
                    self.assertIsNotNone(insp1["observation_receipt"])
                    self.assertIsNotNone(insp1["hsm_receipt_countersignature"])
                self.assertFalse(insp2["render"]["observed"])
                assert_no_raw_buffer_leakage(insp1)

            # 2. Verify cyclopsdraw before/after diff & Layer-3 derived representations
            session.register_synthetic_canvas("src_cdraw", "v1", cyclopsdraw_marks, width=256, height=256)
            shifted_cdraw = copy.deepcopy(cyclopsdraw_marks)
            shifted_cdraw[0]["x"] = 32
            session.register_synthetic_canvas("src_cdraw", "v2", shifted_cdraw, width=256, height=256)
            diff_rep = session.compare_canvases(("src_cdraw", "v1"), ("src_cdraw", "v2"))
            self.assertEqual(diff_rep["status"], "ok")
            self.assertFalse(diff_rep["observed"])
            self.assertGreater(diff_rep["native_diff_metrics"]["changed_pixels"], 0)
            self.assertEqual(len(diff_rep["derived_representations"]), 5)
            assert_no_raw_buffer_leakage(diff_rep)

            # 3. Verify GhostOverlay isolation preview (`ghost_isolation_plane`, 0 leakage)
            ghost_preview = session.preview_ghost_marks(
                "src_cdraw",
                "v2",
                [{"k": "dot", "x": 0.8, "y": 0.8, "r": 0.08, "c": [255, 0, 255], "a": 1.0}],
            )
            self.assertEqual(ghost_preview["layer_isolation"], "ghost_isolation_plane")
            self.assertFalse(ghost_preview["observed"])
            self.assertEqual(ghost_preview["ghost_pixel_leakage_bytes"], 0)
            self.assertTrue(ghost_preview["base_pixel_sha256_preserved"])

            # 4. Verify text-mark and PII hygiene rejection + semantic refusal
            with self.assertRaises(ValueError):
                session.register_synthetic_canvas(
                    "src_bad_txt",
                    "v1",
                    [{"shape": "text", "text": "hello", "x": 10, "y": 10}],
                )
            with self.assertRaises(ValueError):
                session.register_synthetic_canvas(
                    "src_bad_pii",
                    "v1",
                    [{"shape": "rect", "x": 1, "y": 1, "width": 5, "height": 5, "as": "user@example.com"}],
                )
            refused = session.evaluate_agent_query(
                "aesthetic_judgment",
                {"query_type": "aesthetic_judgment", "question": "Is this composition balanced?"},
                source_id="src_cdraw",
                source_version="v2",
            )
            self.assertEqual(refused["status"], "refused")
            self.assertEqual(refused["refusal_code"], REFUSAL_CODE_SEMANTIC)

            # 5. Verify < 100 ms session vault purge
            purge_res = session.purge_session_vault()
            self.assertEqual(purge_res["remaining_sources"], 0)
            self.assertLess(purge_res["elapsed_ms"], 100.0)

    def test_electric_vision_cli_terminal_conventions_and_subcommands(self) -> None:
        from electric_vision_cli import run_cli

        # 1. selftest (--json)
        code, out = run_cli(["--json", "selftest"])
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertTrue(parsed["all_passed"])

        # 2. inspect-marks with --no-color and --plain (must contain zero ANSI escapes and only 7-bit ASCII)
        marks_json = json.dumps([
            {"k": "dot", "x": 0.5, "y": 0.5, "r": 0.1, "c": [0, 229, 255], "a": 1.0}
        ])
        code_plain, out_plain = run_cli([
            "--no-color",
            "--plain",
            "--mode",
            "focus",
            "inspect-marks",
            "--marks",
            marks_json,
            "--width",
            "256",
            "--height",
            "256",
            "--issue-receipt",
        ])
        self.assertEqual(code_plain, 0)
        self.assertNotIn("\033[", out_plain)
        self.assertTrue(out_plain.isascii())
        self.assertIn("[VERIFIED | OBS:VERIFIED_RECEIPT]", out_plain)

        # 3. diff-marks, preview-ghost, refusal-check
        marks_b_json = json.dumps([
            {"k": "dot", "x": 0.6, "y": 0.5, "r": 0.1, "c": [0, 229, 255], "a": 1.0}
        ])
        code_diff, out_diff = run_cli([
            "--no-color",
            "--mode",
            "focus",
            "diff-marks",
            "--marks-a",
            marks_json,
            "--marks-b",
            marks_b_json,
        ])
        self.assertEqual(code_diff, 0)
        self.assertIn("UNOBSERVED_RENDER", out_diff)
        self.assertIn("Motion Trajectories:", out_diff)

        code_ghost, out_ghost = run_cli([
            "--plain",
            "preview-ghost",
            "--base-marks",
            marks_json,
            "--ghost-marks",
            marks_b_json,
        ])
        self.assertEqual(code_ghost, 0)
        self.assertIn("ghost_isolation_plane", out_ghost)

        code_ref, out_ref = run_cli([
            "--json",
            "refusal-check",
            "--query",
            "What emotion does this canvas convey?",
        ])
        self.assertEqual(code_ref, 0)
        ref_data = json.loads(out_ref)
        self.assertEqual(ref_data["status"], "refused")
        self.assertEqual(ref_data["refusal_code"], REFUSAL_CODE_SEMANTIC)

        # 4. watch-stream CLI (--no-color --plain --mode focus and --json)
        stream_frames_json = json.dumps([
            [{"k": "dot", "x": 0.2, "y": 0.5, "r": 0.08, "c": [0, 229, 255], "a": 1.0}],
            [{"k": "dot", "x": 0.4, "y": 0.5, "r": 0.08, "c": [0, 229, 255], "a": 1.0}],
            [{"k": "dot", "x": 0.4, "y": 0.5, "r": 0.08, "c": [0, 229, 255], "a": 1.0}],
        ])
        code_ws, out_ws = run_cli([
            "--no-color",
            "--plain",
            "--mode",
            "focus",
            "watch-stream",
            "--frames",
            stream_frames_json,
            "--width",
            "128",
            "--height",
            "128",
            "--window-size",
            "2",
            "--settle-ticks",
            "1",
            "--issue-receipts",
        ])
        self.assertEqual(code_ws, 0)
        self.assertNotIn("\033[", out_ws)
        self.assertTrue(out_ws.isascii())
        self.assertIn("Continuous Electric Eye Stream", out_ws)
        self.assertIn("Epistemic Watchdog: 1 Prior Observation(s) Superseded -> STALE", out_ws)
        self.assertIn("settled=True", out_ws)

        # 5. Direct CLI inspection of emblem/world.json (CYCLOPS_DRAW_WORLD_V3) and emblem/drawing.json
        world_path = os.path.join(os.path.dirname(__file__), "emblem", "world.json")
        drawing_path = os.path.join(os.path.dirname(__file__), "emblem", "drawing.json")
        if os.path.isfile(world_path):
            code_w, out_w = run_cli([
                "--json",
                "inspect-marks",
                "--marks",
                world_path,
                "--width",
                "512",
                "--height",
                "512",
            ])
            self.assertEqual(code_w, 0)
            w_rep = json.loads(out_w)
            self.assertGreater(
                w_rep["native_simd_metrics"]["op_connected_components_4c"]["component_count"],
                0,
            )
        if os.path.isfile(drawing_path):
            code_d, out_d = run_cli([
                "--json",
                "inspect-marks",
                "--marks",
                drawing_path,
                "--width",
                "512",
                "--height",
                "512",
            ])
            self.assertEqual(code_d, 0)
            d_rep = json.loads(out_d)
            self.assertGreater(
                d_rep["native_simd_metrics"]["op_connected_components_4c"]["component_count"],
                0,
            )

    def test_continuous_electric_eye_stream_temporal_operators_and_stale_decay(self) -> None:
        from electric_vision_agent_bridge import ElectricVisionAgentSession

        f1 = [
            {"shape": "star", "center": {"x": 40, "y": 128}, "outerRadius": 20, "innerRadius": 9, "spikes": 5, "color": "#00e5ff"},
        ]
        f2 = [
            {"shape": "star", "center": {"x": 80, "y": 128}, "outerRadius": 20, "innerRadius": 9, "spikes": 5, "color": "#00e5ff"},
        ]
        f3 = [
            {"shape": "star", "center": {"x": 140, "y": 128}, "outerRadius": 20, "innerRadius": 9, "spikes": 5, "color": "#00e5ff"},
        ]
        # f4 returns to f2 to trigger oscillating_cycle (f2 -> f3 -> f2)
        f4 = copy.deepcopy(f2)
        # f5 and f6 stay identical to f4 to trigger settled regime (2 consecutive zero-delta ticks)
        f5 = copy.deepcopy(f2)
        f6 = copy.deepcopy(f2)

        with ElectricVisionAgentSession(default_resolution=(256, 256)) as session:
            # Also verify extended cyclopsdraw shapes (ellipse, regular-polygon, star, crescent) & single sha256 prefix
            ext_reg = session.register_synthetic_canvas(
                "src_ext_shapes",
                "v1",
                [
                    {"shape": "ellipse", "center": {"x": 64, "y": 64}, "rx": 24, "ry": 12, "rotation": 30, "color": "#00e5ff"},
                    {"shape": "regular-polygon", "center": {"x": 192, "y": 64}, "outerRadius": 20, "sides": 6, "color": "#ffb300"},
                    {"shape": "crescent", "center": {"x": 128, "y": 192}, "outerRadius": 22, "innerRadius": 16, "offset": 8, "color": "#d500f9"},
                ],
                width=256,
                height=256,
            )
            self.assertTrue(ext_reg["native_pixel_sha256"].startswith("sha256:"))
            self.assertNotIn("sha256:sha256:", ext_reg["native_pixel_sha256"])
            ev_ext = session.evict_synthetic_canvas("src_ext_shapes", "v1")
            self.assertTrue(ev_ext["evicted"])
            self.assertEqual(ev_ext["zeroized_bytes"], 256 * 256 * 4)

            stream = session.create_continuous_stream(
                "src_cont_eye",
                width=256,
                height=256,
                max_window_frames=4,
                settle_ticks=2,
                checkpoint_hamming_threshold=2,
            )

            # Tick 1 (v1): Verified with receipt
            t1 = stream.ingest_frame("v1", f1, issue_observation_receipt=True)
            self.assertTrue(t1["observation_state"]["observed"])
            self.assertEqual(t1["observation_state"]["uncertainty_state"], "verified")
            self.assertTrue(stream.check_receipt_freshness("v1")["fresh"])
            self.assertEqual(
                t1["temporal_window_summary"]["settling_and_oscillation_detector"]["regime"],
                "initializing",
            )

            # Stage a GhostOverlay prediction on v1 matching f2's star at x=80
            staged_ghost = stream.stage_ghost_prediction("v1", f2, ghost_id="gh_predict_v2")
            self.assertEqual(staged_ghost["layer_isolation"], "ghost_isolation_plane")
            self.assertFalse(staged_ghost["observed"])

            # Tick 2 (v2): Mutates canvas -> v1 must decay from 'verified' to 'stale'!
            t2 = stream.ingest_frame(
                "v2",
                f2,
                issue_observation_receipt=True,
                verify_staged_ghost_id="gh_predict_v2",
            )
            self.assertIn("v1", t2["decayed_to_stale_versions"])
            v1_freshness = stream.check_receipt_freshness("v1")
            self.assertFalse(v1_freshness["fresh"])
            self.assertEqual(v1_freshness["uncertainty_state"], "stale")
            self.assertEqual(v1_freshness["refusal_code"], "ERR_STALE_OBSERVATION_SUPERSEDED")
            self.assertTrue(stream.check_receipt_freshness("v2")["fresh"])
            self.assertIsNotNone(t2["ghost_realization_audit"])
            self.assertGreater(
                t2["ghost_realization_audit"]["realization_convergence_score"],
                0.5,
            )
            self.assertGreater(
                t2["transition_from_previous"]["dirty_tiles_16x16"]["dirty_tile_count"],
                0,
            )

            # Tick 3 (v3): Star accelerates from dx=+40 (40->80) to dx=+60 (80->140) -> positive ax
            t3 = stream.ingest_frame("v3", f3, issue_observation_receipt=False)
            self.assertIn("v2", t3["decayed_to_stale_versions"])
            kin3 = t3["temporal_window_summary"]["kinematic_trajectory_chain"]
            self.assertEqual(len(kin3["velocities"]), 2)
            self.assertEqual(len(kin3["accelerations"]), 1)
            self.assertGreater(kin3["accelerations"][0]["ax_px_per_tick2"], 0.0)
            self.assertIsNotNone(kin3["predicted_next_centroid"])
            self.assertEqual(
                kin3["predicted_next_centroid"]["target_plane"],
                "ghost_isolation_plane",
            )
            self.assertEqual(
                t3["temporal_window_summary"]["settling_and_oscillation_detector"]["regime"],
                "evolving",
            )

            # Tick 4 (v4): Revisit f2 -> oscillating_cycle detected!
            t4 = stream.ingest_frame("v4", f4, issue_observation_receipt=False)
            det4 = t4["temporal_window_summary"]["settling_and_oscillation_detector"]
            self.assertEqual(det4["regime"], "oscillating_cycle")
            self.assertTrue(det4["oscillating_cycle_detected"])
            self.assertIsNone(t4["ring_buffer_eviction"])  # 4 frames == max_window_frames

            # Tick 5 (v5): Identical to v4; also overflows max_window_frames=4 -> v1 evicted & zeroized!
            t5 = stream.ingest_frame("v5", f5, issue_observation_receipt=False)
            self.assertIsNotNone(t5["ring_buffer_eviction"])
            self.assertTrue(t5["ring_buffer_eviction"]["evicted"])
            self.assertEqual(
                t5["ring_buffer_eviction"]["evicted_source"]["source_version"],
                "v1",
            )
            self.assertEqual(
                t5["ring_buffer_eviction"]["zeroized_bytes"],
                256 * 256 * 4,
            )
            self.assertTrue(
                t5["transition_from_previous"]["dirty_tiles_16x16"]["fast_path_identical_sha"]
            )
            self.assertEqual(
                t5["transition_from_previous"]["dirty_tiles_16x16"]["dirty_tile_count"],
                0,
            )

            # Tick 6 (v6): Second consecutive zero-delta frame -> regime transitions to 'settled'!
            t6 = stream.ingest_frame("v6", f6, issue_observation_receipt=True)
            det6 = t6["temporal_window_summary"]["settling_and_oscillation_detector"]
            self.assertEqual(det6["regime"], "settled")
            self.assertTrue(det6["settled"])
            self.assertEqual(
                t6["temporal_window_summary"]["active_versions"],
                ["v3", "v4", "v5", "v6"],
            )
            self.assertEqual(
                t6["temporal_window_summary"]["evicted_frame_count"],
                2,
            )
            assert_no_raw_buffer_leakage(t6)

    def test_owner_supplied_picture_is_accepted_and_live_capture_still_refused(self) -> None:
        """On the owner's word (2026-10-30, 2026-11-04 & 2026-11-16): a picture the owner sends to remake or enhance is accepted under the
        honest origin 'owner_supplied_picture', the Companion's own desktop glance is accepted under
        'companion_desktop_glance', and Electric Browser's in-memory viewport frame is accepted under
        'electric_browser_frame' (including via ContinuousElectricEyeStream.ingest_frame(origin_attestation=...));
        every other non-synthetic origin, live webcam or generic screen capture above all, stays refused."""
        import shutil
        import tempfile
        from electric_vision_agent_bridge import ElectricVisionAgentSession
        marks = [{"mark_id": "a", "shape": "rect", "x0": 40, "y0": 40, "x1": 160, "y1": 120, "rgba": [200, 40, 40, 255]}]
        folder = tempfile.mkdtemp(prefix="cee-test-")
        self.addCleanup(shutil.rmtree, folder, True)
        with ElectricVisionAgentSession(run_dir=folder, default_resolution=(512, 512)) as session:
            accepted = session.register_synthetic_canvas("src_gate_one", "v1", marks, width=512, height=512,
                                                         source_kind="image_bytes", origin_attestation="owner_supplied_picture")
            self.assertEqual(accepted["origin_attestation"], "owner_supplied_picture")
            glance_ok = session.register_synthetic_canvas("src_gate_one", "v2", marks, width=512, height=512,
                                                          source_kind="image_bytes", origin_attestation="companion_desktop_glance")
            self.assertEqual(glance_ok["origin_attestation"], "companion_desktop_glance")
            browser_ok = session.register_synthetic_canvas("src_gate_one", "v3", marks, width=512, height=512,
                                                           source_kind="captured_frame", origin_attestation="electric_browser_frame")
            self.assertEqual(browser_ok["origin_attestation"], "electric_browser_frame")
            stream = session.create_continuous_stream("src_companion_glance", width=128, height=128, max_window_frames=4)
            rgba_128 = bytes([20, 60, 30, 255]) * (128 * 128)
            ingested = stream.ingest_frame(
                "v1",
                [],
                rgba=rgba_128,
                origin={"kind": "companion_desktop_glance"},
                origin_attestation="companion_desktop_glance",
            )
            self.assertEqual(ingested["tick"], 1)
            eb_stream = session.create_continuous_stream("src_eb_frame", width=128, height=128, max_window_frames=4)
            eb_ingested = eb_stream.ingest_frame(
                "v1",
                [],
                rgba=rgba_128,
                origin={"kind": "electric_browser_frame"},
                origin_attestation="electric_browser_frame",
            )
            self.assertEqual(eb_ingested["tick"], 1)
            for n, origin in enumerate(("live_webcam_capture", "live_screen_capture", "screen_capture", "anything_else"), 4):
                with self.subTest(origin=origin), self.assertRaisesRegex(PermissionError, "ERR_NON_SYNTHETIC_INPUT_PROHIBITED"):
                    session.register_synthetic_canvas("src_gate_one", f"v{n}", marks, width=512, height=512, origin_attestation=origin)

    def test_quarantine_allowlist_accepts_electric_browser_frame_and_preserves_eviction_and_refusals(self) -> None:
        """Directly verifies EXEC-GATE-3 zero-PII quarantine allowlist ('synthetic_fixture',
        'owner_supplied_picture', 'companion_desktop_glance', 'electric_browser_frame'),
        SOURCE_ID_REGEX / long-digit / prohibited-token enforcement, refusal of generic live capture
        with ERR_NON_SYNTHETIC_INPUT_PROHIBITED, and explicit_bzero ring-buffer eviction."""
        from contextlib import ExitStack
        import hashlib
        import socket
        import struct
        from unittest.mock import Mock, patch
        import exec_gate_daemons as gates
        import electric_vision_agent_bridge as bridge

        class _FakeConn:
            def __init__(self, request: dict) -> None:
                self.request = request
                self.responses: list = []

            def __enter__(self) -> "_FakeConn":
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def settimeout(self, timeout: float) -> None:
                pass

            def close(self) -> None:
                pass

            def getsockopt(self, level: int, option: int, size: int) -> bytes:
                assert (level, option, size) == (socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                return struct.pack("3i", 101, 42, 43)

        digest16 = hashlib.sha256(b"electric-browser:v0:d_99887766").digest()[:16]
        eb_doc_sid = "src_eb_" + "".join(chr(ord("a") + ((b >> 4) & 0xF)) + chr(ord("a") + (b & 0xF)) for b in digest16)

        allowed = (
            ("src_eb_frame", "synthetic_fixture"),
            ("src_eb_frame", "owner_supplied_picture"),
            ("src_eb_frame", "companion_desktop_glance"),
            ("src_eb_frame", "electric_browser_frame"),
            (eb_doc_sid, "electric_browser_frame"),
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
        requests: list[_FakeConn] = []
        for idx, (sid, origin) in enumerate(allowed, 1):
            requests.append(_FakeConn({
                "method": "inspect_source",
                "params": {
                    "source_id": sid,
                    "source_version": f"v{idx}",
                    "source_kind": "captured_frame",
                    "marks": [{"mark_id": "m1"}],
                    "origin_attestation": origin,
                },
            }))
        for idx, origin in enumerate(refused_live, 10):
            requests.append(_FakeConn({
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
            requests.append(_FakeConn({
                "method": "inspect_source",
                "params": {
                    "source_id": sid,
                    "source_version": f"v{idx}",
                    "source_kind": "captured_frame",
                    "marks": [{"mark_id": "m1"}],
                    "origin_attestation": "electric_browser_frame",
                },
            }))
        requests.append(_FakeConn({
            "method": "inspect_source",
            "params": {
                "source_id": "src_eb_frame",
                "source_version": "v30",
                "source_kind": "captured_frame",
                "marks": [{"mark_id": "user_avatar"}],
                "origin_attestation": "electric_browser_frame",
            },
        }))
        requests.append(_FakeConn({"method": "get_status"}))
        requests.append(_FakeConn({"method": "shutdown"}))

        server = Mock()
        server.accept.side_effect = [(c, None) for c in requests]
        with ExitStack() as stack:
            stack.enter_context(patch.object(gates.socket, "socket", return_value=server))
            stack.enter_context(patch.object(gates, "die_with_parent"))
            stack.enter_context(patch.object(gates.os.path, "exists", return_value=False))
            stack.enter_context(patch.object(gates.os, "chmod"))
            stack.enter_context(patch.object(gates, "_recv_framed_json", side_effect=lambda c: c.request))
            if hasattr(gates, "_recv_rpc_request"):
                stack.enter_context(patch.object(gates, "_recv_rpc_request", side_effect=lambda c: c.request))
            stack.enter_context(patch.object(gates, "_send_framed_json", side_effect=lambda c, p: c.responses.append(p)))
            gates._quarantine_and_kill_switch_daemon_loop("/fake/q", "/fake/evidence", None, None, 42, 43, Mock())

        for idx, (_sid, origin) in enumerate(allowed):
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
        vault["register_rgba"](eb_doc_sid, "v1", 128, 128, bytes([12, 34, 56, 255]) * (128 * 128))
        evicted = vault["evict_source"]((eb_doc_sid, "v1"))
        self.assertTrue(evicted["evicted"])
        self.assertEqual(evicted["zeroized_bytes"], 128 * 128 * 4)
        sample_buf = bytearray(b"electric-browser-frame-bytes")
        wiped = bridge.explicit_zeroize_bytearray(sample_buf)
        self.assertEqual(wiped, len(sample_buf))
        self.assertEqual(sample_buf, bytearray(len(sample_buf)))

    def test_component_box_of_a_region_wholly_beyond_128_pixels(self) -> None:
        """The native components (prod_gate_enclave) run _connected_components_from_runs at full resolution: a region wholly
        beyond x or y = 128 gets its own box (its minimum was seeded with CANVAS_WIDTH, CANVAS_HEIGHT, 128)."""
        comps = ElectricEyeFixtureHarness._connected_components_from_runs([(200, 300, 301), (201, 300, 301)], connectivity=4)
        self.assertEqual(len(comps), 1)
        box = comps[0]["bbox"]
        self.assertEqual((box["x_min"], box["y_min"], box["x_max"], box["y_max"]), (300, 200, 301, 201))


if __name__ == "__main__":
    unittest.main(verbosity=2)
