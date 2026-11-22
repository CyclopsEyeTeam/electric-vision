"""Owner, 13-14 Nov 2026: a PlayHub screen's frames may be measured by Electric Eye, said honestly as
origin_attestation="play_engine_screen" (frames of the player's own game window, held in memory only). Any other live
capture stays refused, and the Work-chat MCP cannot claim this origin. Run: python3 test_play_engine_screen.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from electric_vision_agent_bridge import ElectricVisionAgentSession


def frame(w=512, h=512):
    px = bytearray(w * h * 4)
    for y in range(100, 140):
        for x in range(200, 260):
            i = (y * w + x) * 4
            px[i:i + 4] = bytes((40, 160, 50, 255))
    return bytes(px)


def main():
    with tempfile.TemporaryDirectory(prefix="ee-") as tmp, ElectricVisionAgentSession(run_dir=os.path.join(tmp, "e")) as s:
        r = s.register_pixel_canvas("src_play_screen", "v1", frame(), 512, 512, source_kind="captured_frame", origin_attestation="play_engine_screen")
        assert r.get("origin_attestation", "play_engine_screen") == "play_engine_screen", r
        m = s.inspect_canvas("src_play_screen", "v1")
        comps = m["native_simd_metrics"]["op_connected_components_4c"]["components"]
        assert len(comps) == 1 and comps[0]["bbox"]["x_min"] == 200 and comps[0]["bbox"]["width"] == 60, comps
        try:
            s.register_pixel_canvas("src_other", "v1", frame(), 512, 512, source_kind="captured_frame", origin_attestation="random_live_capture")
        except PermissionError:
            pass
        else:
            raise AssertionError("an unknown live origin must stay refused")
    mcp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "electric_vision_mcp.py"), encoding="utf-8").read()
    assert '"play_engine_screen"' not in mcp, "the Work-chat MCP must not offer play_engine_screen"
    print("ok: play_engine_screen accepted, unknown origins refused, not in the MCP")


if __name__ == "__main__":
    main()
