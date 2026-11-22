"""
visual_sources.py

Turns real visual sources into RGBA pictures for Electric Vision: image files, SVG (file or text),
video frames, and HTML pages such as play-engine games captured with headless Chrome.

Every loader is bounded in size and time and works offline: Chrome runs with a dead proxy and,
for SVG, a Content-Security-Policy that forbids every fetch; ffmpeg reads only the given file.
Callers pass absolute paths the Cyclops side has already checked against the chat's folders.
"""

from __future__ import annotations

from contextlib import contextmanager
import html
import io
import json
import os
import shutil
import selectors
import signal
import stat
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageSequence

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})
SVG_SUFFIXES = frozenset({".svg"})
VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"})
PAGE_SUFFIXES = frozenset({".html", ".htm"})

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_VIDEO_BYTES = 1024 * 1024 * 1024
MAX_SVG_TEXT_BYTES = 4 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
MAX_PAGE_SIZE = (1920, 1080)
CHROME = "/usr/bin/google-chrome"
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"

Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS


def classify(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in SVG_SUFFIXES:
        return "svg"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in PAGE_SUFFIXES:
        return "page"
    raise ValueError(
        f"ERR_UNSUPPORTED_VISUAL_FILE: {Path(path).name!r}; Electric Vision reads images "
        f"({', '.join(sorted(IMAGE_SUFFIXES))}), SVG, video ({', '.join(sorted(VIDEO_SUFFIXES))}) and HTML pages"
    )


def _open_confined(parts: Sequence[str], root_fd: int, *, directory: bool = False) -> int:
    """Walk a pinned descriptor, rejecting traversal and symlinks at every component."""
    if any(p in ("", ".", "..") or "/" in p or "\x00" in p for p in parts):
        raise ValueError("ERR_VISUAL_PATH_TRAVERSAL")
    parent = os.dup(root_fd)
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        if not parts:
            if directory:
                return os.dup(parent)
            raise ValueError("ERR_VISUAL_NOT_A_FILE")
        return os.open(parts[-1], flags | (os.O_DIRECTORY if directory else 0), dir_fd=parent)
    finally:
        os.close(parent)


def _absolute_parts(path: str) -> Tuple[str, ...]:
    if not Path(path).is_absolute():
        raise ValueError("ERR_VISUAL_PATH_NOT_ABSOLUTE: Cyclops passes checked absolute paths only")
    # Inspect raw components before pathlib can normalize away '.' components.
    parts = tuple(p for p in str(path).split("/") if p)
    if any(p in (".", "..") or "\x00" in p for p in parts):
        raise ValueError("ERR_VISUAL_PATH_TRAVERSAL")
    return parts


def _open_absolute(path: str, *, directory: bool = False) -> int:
    parts = _absolute_parts(path)
    root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        return _open_confined(parts, root_fd, directory=directory)
    finally:
        os.close(root_fd)


@contextmanager
def _checked_file(path: str, limit: int, *, root_fd: Optional[int] = None):
    """Keep the checked regular descriptor open; callers never reopen the original path."""
    fd = (_open_absolute(path) if root_fd is None
          else _open_confined(str(path).split("/"), root_fd))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"ERR_VISUAL_NOT_A_FILE: {Path(path).name}")
        if st.st_size > limit:
            raise ValueError(f"ERR_VISUAL_FILE_TOO_LARGE: {Path(path).name} is {st.st_size} bytes (limit {limit})")
        handle = os.fdopen(fd, "rb")
        fd = None
        with handle:
            yield handle
    finally:
        if fd is not None:
            os.close(fd)


def _read_bounded(handle, limit: int) -> bytes:
    data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("ERR_VISUAL_FILE_TOO_LARGE: file grew beyond its byte limit")
    return data


@contextmanager
def _private_snapshot(path: str, limit: int):
    """Give external decoders one bounded private file, never an unchecked pathname."""
    with _checked_file(path, limit) as source, tempfile.TemporaryDirectory(prefix="ev-media-") as work:
        snapshot = Path(work) / ("source" + Path(path).suffix)
        total = 0
        with snapshot.open("xb") as target:
            os.chmod(snapshot, 0o600)
            while True:
                chunk = source.read(min(65536, limit + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise ValueError("ERR_VISUAL_FILE_TOO_LARGE: file grew beyond its byte limit")
                target.write(chunk)
        yield snapshot


def load_image(path: str, *, frame: int = 0) -> Image.Image:
    """Decodes an image file (first frame, or `frame` of an animation) as RGBA."""
    p = Path(path)
    with _checked_file(path, MAX_FILE_BYTES) as source:
        data = _read_bounded(source, MAX_FILE_BYTES)
    with Image.open(io.BytesIO(data)) as img:
        if img.width * img.height > MAX_SOURCE_PIXELS:
            raise ValueError(f"ERR_VISUAL_TOO_LARGE: {img.width}x{img.height}")
        if frame:
            for index, part in enumerate(ImageSequence.Iterator(img)):
                if index == frame:
                    return part.convert("RGBA")
            raise ValueError(f"ERR_VISUAL_FRAME_MISSING: {p.name} has fewer than {frame + 1} frames")
        return img.convert("RGBA")


class _ChromeProfile:
    """A short private profile directory (Chrome's socket path must stay under ~100 bytes)."""

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix="evc-", dir="/tmp")
        os.chmod(self.path, 0o700)
        return self.path

    def __exit__(self, *_: object) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _run_owned(argv: Sequence[str], *, env: dict, timeout: float, max_output: int = 0) -> bytes:
    """Bound output and wall time; kill the owned session even if its leader exits first."""
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE if max_output else subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        if proc.stdout is not None:
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    for key, _ in selector.select(left):
                        chunk = os.read(key.fd, min(65536, max_output + 1 - len(output)))
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            output.extend(chunk)
                            if len(output) > max_output:
                                raise ValueError("ERR_VISUAL_OUTPUT_TOO_LARGE")
        proc.wait(timeout=max(0.001, deadline - time.monotonic()))
        if proc.returncode:
            raise RuntimeError(f"ERR_VISUAL_SUBPROCESS_FAILED: exit {proc.returncode}")
        return bytes(output)
    finally:
        # A successful leader can leave children behind. Never target our own process group.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


def _bounded_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0 or max(width, height) > 16384 or width * height > MAX_SOURCE_PIXELS:
        raise ValueError(f"ERR_VISUAL_TOO_LARGE: {width}x{height}")


def _chrome_screenshot(
    url: str, width: int, height: int, *, wait_ms: int, transparent: bool, allow_origin: str = ""
) -> Image.Image:
    if not os.access(CHROME, os.X_OK):
        raise RuntimeError("ERR_VISUAL_RENDERER_MISSING: google-chrome is needed to draw SVG and pages")
    with _ChromeProfile() as profile:
        shot = os.path.join(profile, "shot.png")
        argv = [
            CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--disable-extensions", "--disable-sync", "--mute-audio", f"--user-data-dir={profile}/p",
            # No network: every request goes to a closed local port.
            # (A page server of our own, when given, is the one address reached directly.)
            "--proxy-server=127.0.0.1:9", f"--proxy-bypass-list=<-loopback>{';' + allow_origin if allow_origin else ''}",
            "--hide-scrollbars", f"--window-size={width},{height}", f"--virtual-time-budget={wait_ms}",
            *(["--default-background-color=00000000"] if transparent else []),
            f"--screenshot={shot}", url,
        ]
        env = {"PATH": "/usr/bin:/bin", "HOME": profile, "TMPDIR": profile, "LANG": "C.UTF-8"}
        try:
            _run_owned(argv, env=env, timeout=20)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ERR_VISUAL_RENDER_TIMEOUT: the page did not finish drawing within 20 s")
        if not os.path.exists(shot):
            raise RuntimeError("ERR_VISUAL_RENDER_FAILED: Chrome produced no picture")
        with Image.open(shot) as img:
            _bounded_dimensions(img.width, img.height)
            return img.convert("RGBA")


def _svg_size(svg_text: str) -> Tuple[int, int]:
    """Reads the SVG's own width/height or viewBox; falls back to 512x512."""
    import re

    head = svg_text[: svg_text.find(">", svg_text.find("<svg")) + 1] if "<svg" in svg_text else ""

    def attr(name: str) -> Optional[float]:
        m = re.search(rf'\b{name}\s*=\s*["\']\s*([0-9.]+)\s*(px)?\s*["\']', head)
        return float(m.group(1)) if m else None

    w, h = attr("width"), attr("height")
    if not (w and h):
        m = re.search(r'\bviewBox\s*=\s*["\']\s*[-0-9.]+[\s,]+[-0-9.]+[\s,]+([0-9.]+)[\s,]+([0-9.]+)', head)
        if m:
            w, h = float(m.group(1)), float(m.group(2))
    if not (w and h):
        return 512, 512
    scale = min(1.0, MAX_PAGE_SIZE[0] / w, MAX_PAGE_SIZE[1] / h)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def render_svg(svg_text: str) -> Image.Image:
    """Draws SVG text exactly as a browser shows it, with every external fetch forbidden."""
    if len(svg_text.encode("utf-8")) > MAX_SVG_TEXT_BYTES:
        raise ValueError(f"ERR_SVG_TOO_LARGE: SVG text exceeds {MAX_SVG_TEXT_BYTES} bytes")
    if "<svg" not in svg_text:
        raise ValueError("ERR_NOT_SVG: no <svg> element found")
    width, height = _svg_size(svg_text)
    body = svg_text[svg_text.find("<svg"):]
    page = (
        "<!doctype html><meta charset=utf-8>"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; img-src data:; "
        "style-src 'unsafe-inline'; font-src data:\">"
        f"<style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden}}"
        f"svg{{display:block;width:{width}px;height:{height}px}}</style>{body}"
    )
    with _ChromeProfile() as work:
        page_path = os.path.join(work, "svg.html")
        with open(page_path, "w", encoding="utf-8") as handle:
            handle.write(page)
        return _chrome_screenshot(Path(page_path).as_uri(), width, height, wait_ms=500, transparent=True)


def load_svg(path: str) -> Image.Image:
    with _checked_file(path, MAX_SVG_TEXT_BYTES) as source:
        text = _read_bounded(source, MAX_SVG_TEXT_BYTES).decode("utf-8", errors="replace")
    return render_svg(text)


class _PageServer:
    """A read-only HTTP server for one folder on 127.0.0.1, so a page's JavaScript modules load as on the web."""

    def __init__(self, root: Path, *, root_fd: Optional[int] = None,
                 page_parts: Optional[Sequence[str]] = None, page_bytes: Optional[bytes] = None) -> None:
        import functools
        import http.server
        import threading

        root_fd = _open_absolute(str(root), directory=True) if root_fd is None else os.dup(root_fd)

        class Quiet(http.server.SimpleHTTPRequestHandler):
            timeout = 3.0

            def log_message(self, *_: object) -> None:
                pass

            def do_POST(self) -> None:  # pages may only read
                self.send_error(405)

            def send_head(self):
                # Walk a pinned root, refusing symlinks at every hop (including indexes).
                parts = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).split("/")
                if any(p in ("..", ".") or "\x00" in p for p in parts):
                    self.send_error(403)
                    return None
                parts = [p for p in parts if p]
                if not parts or self.path.split("?", 1)[0].endswith("/"):
                    parts.append("index.html")
                try:
                    if page_parts is not None and tuple(parts) == tuple(page_parts):
                        data = page_bytes
                        if data is None or len(data) > MAX_FILE_BYTES:
                            raise ValueError("not a bounded page snapshot")
                    else:
                        # Nested HTML/media assets share the pinned page root and byte ceiling.
                        with _checked_file("/".join(parts), MAX_FILE_BYTES, root_fd=root_fd) as source:
                            data = _read_bounded(source, MAX_FILE_BYTES)
                    handle = io.BytesIO(data)
                    try:
                        self.send_response(200)
                        self.send_header("Content-type", self.guess_type(parts[-1]))
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                    except BaseException:
                        handle.close()
                        raise
                    return handle
                except (OSError, ValueError):
                    self.send_error(404)
                    return None

        try:
            self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
            self.httpd.daemon_threads = False
        except BaseException:
            os.close(root_fd)
            raise
        self.root_fd = root_fd
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"127.0.0.1:{self.httpd.server_address[1]}"

    def __exit__(self, *_: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        os.close(self.root_fd)


def capture_page(
    path: str, *, width: int, height: int, at_ms: Sequence[int], root: Optional[str] = None, query: str = ""
) -> List[Image.Image]:
    """
    Screenshots an HTML page (e.g. a play-engine game) after each virtual-time point in `at_ms`.
    The page is served from `root` (default: its own folder) by a private local server.
    """
    parts = _absolute_parts(path)
    folder = Path(root) if root is not None else Path(path).parent
    root_parts = _absolute_parts(str(folder))
    if parts[:len(root_parts)] != root_parts or len(parts) <= len(root_parts):
        raise ValueError("ERR_PAGE_OUTSIDE_ROOT: the page is not inside its served folder")
    rel_parts = parts[len(root_parts):]
    root_fd = _open_absolute(str(folder), directory=True)
    try:
        with _checked_file("/".join(rel_parts), MAX_FILE_BYTES, root_fd=root_fd) as source:
            page_bytes = _read_bounded(source, MAX_FILE_BYTES)
        width = max(64, min(MAX_PAGE_SIZE[0], int(width)))
        height = max(64, min(MAX_PAGE_SIZE[1], int(height)))
        with _PageServer(folder, root_fd=root_fd, page_parts=rel_parts, page_bytes=page_bytes) as origin:
            rel = urllib.parse.quote("/".join(rel_parts), safe="/")
            url = f"http://{origin}/{rel}" + (f"?{query}" if query else "")
            return [
                _chrome_screenshot(url, width, height, wait_ms=max(1, int(t)), transparent=False, allow_origin=origin)
                for t in at_ms
            ]
    finally:
        os.close(root_fd)


@contextmanager
def checked_video(path: str):
    with _private_snapshot(path, MAX_VIDEO_BYTES) as snapshot:
        yield snapshot, _video_info_snapshot(snapshot)


def video_info(path: str) -> dict:
    with checked_video(path) as (_, info):
        return info


def _video_info_snapshot(p: Path) -> dict:
    output = _run_owned(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", str(p)],
        env={"PATH": "/usr/bin:/bin"}, timeout=15, max_output=65536,
    )
    try:
        data = json.loads(output or b"{}")
        stream = (data.get("streams") or [{}])[0]
        _bounded_dimensions(int(stream["width"]), int(stream["height"]))
        num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration_s": round(float(data.get("format", {}).get("duration", 0.0)), 3),
            "fps": round(float(num) / float(den or 1), 3) if float(den or 1) else None,
        }
    except (KeyError, ValueError, IndexError, TypeError):
        raise ValueError(f"ERR_NOT_A_VIDEO: {p.name} has no readable video stream")


def video_frames(path: str, times_s: Sequence[float]) -> List[Image.Image]:
    """Decodes one frame at each time (seconds) with ffmpeg, as RGBA."""
    with checked_video(path) as (snapshot, _):
        return _video_frames_snapshot(snapshot, times_s)


def _video_frames_snapshot(p: Path, times_s: Sequence[float]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for t in times_s:
        output = _run_owned(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
             "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(p), "-map", "0:v:0",
             "-frames:v", "1", "-vf", "scale=w='min(1920,iw)':h='min(1080,ih)':force_original_aspect_ratio=decrease",
             "-threads", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            env={"PATH": "/usr/bin:/bin"}, timeout=20, max_output=16 * 1024 * 1024,
        )
        if not output:
            raise ValueError(f"ERR_VIDEO_FRAME_MISSING: no frame at {t} s in {p.name}")
        with Image.open(io.BytesIO(output)) as img:
            _bounded_dimensions(img.width, img.height)
            if img.width > MAX_PAGE_SIZE[0] or img.height > MAX_PAGE_SIZE[1]:
                raise ValueError("ERR_VIDEO_DECODE_DIMENSIONS: decoder exceeded scaled bounds")
            frames.append(img.convert("RGBA"))
    return frames


def fit_to_canvas(img: Image.Image, width: int, height: int) -> Tuple[bytes, dict]:
    """
    Scales the picture to fit the canvas without cropping (keeping its aspect), centres it on a
    transparent canvas and returns RGBA bytes plus where it was placed.
    """
    scale = min(width / img.width, height / img.height)
    w = max(1, int(round(img.width * scale)))
    h = max(1, int(round(img.height * scale)))
    # BOX (down) and BILINEAR (up) never overshoot, so edges leave no stray specks that would count as objects.
    resized = img if (w, h) == img.size else img.resize((w, h), Image.BOX if scale < 1 else Image.BILINEAR)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ox, oy = (width - w) // 2, (height - h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas.tobytes(), {
        "source_width": img.width,
        "source_height": img.height,
        "scale": round(scale, 6),
        "placed_at": {"x": ox, "y": oy, "width": w, "height": h},
    }


BACKGROUND_TOLERANCE = 16


def key_background(img: Image.Image, mode: str = "auto") -> Tuple[Image.Image, dict]:
    """
    Electric Eye measures shapes on non-transparent pixels, so an opaque picture would be one big
    object. "auto" treats the colour filling most of the border as empty background (made
    transparent within a small tolerance); "keep" leaves the picture as it is; "#rrggbb" keys that colour.
    """
    rgba = img.convert("RGBA")
    if mode == "keep":
        return rgba, {"background": "kept"}
    if mode == "auto":
        w, h = rgba.size
        border = [rgba.getpixel((x, y)) for x in range(0, w, max(1, w // 64)) for y in (0, h - 1)]
        border += [rgba.getpixel((x, y)) for y in range(0, h, max(1, h // 64)) for x in (0, w - 1)]
        if all(p[3] == 0 for p in border):
            return rgba, {"background": "already_transparent"}
        counts: dict = {}
        for p in border:
            key = tuple(c // 8 for c in p[:3])
            counts.setdefault(key, []).append(p)
        best = max(counts.values(), key=len)
        if len(best) < 0.6 * len(border):
            return rgba, {"background": "kept", "reason": "no single colour fills most of the border"}
        colour = tuple(sorted(c[i] for c in best)[len(best) // 2] for i in range(3))
    else:
        text = mode.lstrip("#")
        if len(text) != 6:
            raise ValueError("background must be 'auto', 'keep' or a colour like '#102030'")
        colour = tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
    solid = Image.new("RGB", rgba.size, colour)  # type: ignore[arg-type]
    delta = ImageChops.difference(rgba.convert("RGB"), solid)
    near = ImageChops.lighter(ImageChops.lighter(delta.getchannel(0), delta.getchannel(1)), delta.getchannel(2))
    keep_mask = near.point(lambda v: 255 if v > BACKGROUND_TOLERANCE else 0)
    # A 3x3 opening drops the 1-2 px specks that compression noise leaves along edges.
    keep_mask = keep_mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    alpha = ImageChops.multiply(rgba.getchannel("A"), keep_mask)
    rgba.putalpha(alpha)
    return rgba, {"background": "#%02x%02x%02x" % colour, "made_transparent": True, "tolerance": BACKGROUND_TOLERANCE,
                  "speckle_filter": "3x3 opening"}


def default_canvas(img_w: int, img_h: int, supported: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    """Wide, large pictures go on 1920x1080; others on the smallest square canvas that keeps their detail."""
    if img_w >= 1.4 * img_h and img_w > 512 and (1920, 1080) in supported:
        return 1920, 1080
    side = max(img_w, img_h)
    for w, h in sorted((r for r in supported if r[0] == r[1]), key=lambda r: r[0]):
        if side <= w:
            return w, h
    return (512, 512) if (512, 512) in supported else tuple(supported[-1])  # type: ignore[return-value]


# Plain names for the colours a shape is made of, so "#d62828" reads as red without guessing.
_NAMED = [("black", (0, 0, 0)), ("white", (255, 255, 255)), ("grey", (128, 128, 128)), ("red", (220, 30, 30)), ("orange", (245, 140, 20)),
          ("yellow", (240, 220, 40)), ("green", (30, 160, 60)), ("teal", (20, 150, 150)), ("cyan", (40, 200, 230)), ("blue", (30, 70, 220)),
          ("navy", (20, 30, 110)), ("purple", (130, 50, 170)), ("pink", (240, 120, 180)), ("brown", (130, 80, 40)), ("dark red", (120, 15, 20)),
          ("dark green", (15, 80, 30)), ("light blue", (150, 200, 245))]


def colour_name(rgb: Tuple[int, int, int]) -> str:
    return min(_NAMED, key=lambda n: sum((a - b) ** 2 for a, b in zip(rgb, n[1])))[0]


def component_colours(img: Image.Image, components: Sequence[dict], *, per_component: int = 2) -> List[dict]:
    """Sample native RGBA boxes, never a downscaled/composited presentation PNG."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    out = []
    for comp in components:
        box = comp.get("bbox") or {}
        try:
            x0, y0, x1, y1 = int(box["x_min"]), int(box["y_min"]), int(box["x_max"]) + 1, int(box["y_max"]) + 1
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        counts: dict = {}
        for r, g, b, a in img.crop((x0, y0, x1, y1)).getdata():
            if a > 0:
                key = ((r >> 4) << 4 | 8, (g >> 4) << 4 | 8, (b >> 4) << 4 | 8)
                counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        if not total:
            continue
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:per_component]
        out.append({"region_id": comp.get("region_id"), "colours": [
            {"hex": "#%02x%02x%02x" % rgb, "name": colour_name(rgb), "share": round(n / total, 3)} for rgb, n in top]})
    return out


def describe_svg_text(svg_text: str) -> str:
    """A short label for an inline SVG in replies (never the full text)."""
    return html.unescape(svg_text.strip()[:60]).replace("\n", " ")
