"""Finite, source-bound field recording, independent of reads and Electric Eye.

Only this worker captures; immutable original files outlive every offered packet.
The stream does not grant input authority or infer model attention/native ticks.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

from PIL import Image, ImageDraw

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 8_388_608
PNG_OVERHEAD_BYTES = 65_536


class _BoundedPNG(io.BytesIO):
    def write(self, value):
        if self.tell() + len(value) > MAX_IMAGE_BYTES:
            raise ValueError('PNG exceeds 8 MiB image limit')
        return super().write(value)


def _png(image):
    with _BoundedPNG() as output:
        image.save(output, format='PNG')
        return output.getvalue()


def _write_new(path, payload):
    # Own spool paths are never replaced or reused, including packet retries.
    with path.open('xb') as stream:
        stream.write(payload)


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.status-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _default_byte_budget(binding, max_frames):
    """Logical original-PNG allowance from the verified RGB field geometry.

    This reserves no RAM or disk; only successfully captured bytes are written.
    Each original still has its independent 8 MiB PNG encoding bound.
    """
    if not isinstance(binding,dict):raise ValueError('Invalid capture budget binding')
    crop,size=binding.get('crop'),binding.get('size')
    if (not isinstance(crop,(list,tuple)) or len(crop)!=4
            or not isinstance(size,(list,tuple)) or len(size)!=2
            or any(type(value) is not int for value in [*crop,*size])
            or min(*size,crop[2],crop[3])<1 or min(crop[:2])<0
            or size[0]*size[1]>MAX_PIXELS or crop[2]*crop[3]>MAX_PIXELS
            or crop[0]+crop[2]>size[0] or crop[1]+crop[3]>size[1]):
        raise ValueError('Capture budget requires valid field crop geometry')
    return max_frames*min(MAX_IMAGE_BYTES,crop[2]*crop[3]*3+PNG_OVERHEAD_BYTES)


class ContinuousVision:
    def __init__(self, source, binding: dict, directory: Path, *, started: float, deadline: float,
                 clock=time.monotonic, fps=5, max_frames=466, max_bytes=None, on_terminal=None):
        if (not math.isfinite(started) or not math.isfinite(deadline) or deadline <= started
                or isinstance(fps, bool) or not math.isfinite(fps) or not 0 < fps <= 1000
                or type(max_frames) is not int or not 1 <= max_frames <= 466):
            raise ValueError('Invalid continuous capture bounds')
        if max_bytes is None:
            max_bytes=_default_byte_budget(binding,max_frames)
        elif type(max_bytes) is not int or not 1 <= max_bytes <= max_frames*MAX_IMAGE_BYTES:
            raise ValueError('Invalid continuous byte bound')
        self.on_terminal = on_terminal
        self.source, self.binding = source, copy.deepcopy(binding)
        self.directory = Path(directory)
        self.clock, self.started, self.deadline = clock, started, deadline
        self.fps, self.max_frames, self.max_bytes = fps, max_frames, max_bytes
        self._lock = threading.RLock()
        self._packets_lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = None
        self._frames, self._packets, self._gaps = [], {}, []
        self._phase, self._reason = 'initialized', None
        self._bytes, self._missed = 0, 0
        self._last_status = -math.inf
        self._expiry_accounted = False
        self._next_due, self._inflight = started, None
        # Binding provenance is public/native scalar identity only. Never spool endpoints.
        self._public_binding = {k: copy.deepcopy(binding[k]) for k in
                                ('pid', 'window_id', 'runtime', 'world_id', 'segment_id', 'size', 'crop', 'binding')
                                if k in binding}
        if len(json.dumps(self._public_binding)) > 4096:
            raise ValueError('Binding exceeds bounded metadata')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(self.directory.iterdir()):
            raise ValueError('Use an empty dedicated continuous stream directory')
        (self.directory / 'manifest.jsonl').touch(mode=0o600)
        with self._lock:
            self._save_status(force=True)

    def start(self):
        with self._lock:
            if self._thread is not None or self._cancel.is_set() or self._phase != 'initialized':
                return
            if self.clock() >= self.deadline:
                self._finish('expired', 'deadline')
                return
            self._phase = 'recording'
            self._thread = threading.Thread(target=self._run, name='alien-field-recording', daemon=True)
            self._thread.start()

    def stop(self, reason='closed'):
        # Publication and cancellation share a short gate containing no disk I/O.
        with self._lock:
            if reason == 'expired' and self.clock() >= self.deadline:
                self._expire(self._next_due, self._inflight)
            self._cancel.set()
            if self._phase in ('initialized', 'recording'):
                self._phase, self._reason = 'stopped', str(reason)[:240]

    def join(self, timeout=7.0):
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def _state(self):
        return dict(phase=self._phase, target_fps=self.fps, captured_frames=len(self._frames),
                    missed_slots=self._missed, bytes_used=self._bytes, max_bytes=self.max_bytes, started=self.started, deadline=self.deadline,
                    first_capture_time=self._frames[0]['capture_time'] if self._frames else None,
                    last_capture_time=self._frames[-1]['capture_time'] if self._frames else None,
                    worker_alive=bool(self._thread and self._thread.is_alive()), reason=self._reason,
                    gaps=copy.deepcopy(self._gaps))

    def status(self):
        with self._lock:
            return self._state()

    def _save_status(self, force=False):
        with self._lock:
            now = self.clock()
            if not force and now - self._last_status < .5:
                return
            snapshot = self._state()
            self._last_status = now
        # Stop never waits for status or manifest disk I/O.
        _atomic_json(self.directory / 'status.json', snapshot)

    def _finish(self, phase, reason, gap=False):
        # Called under this object's lock, never the manager's lock.
        if self._phase in ('initialized', 'recording'):
            self._phase, self._reason = phase, str(reason)[:240]
            if gap:
                self._gaps.append(dict(kind=phase, time=self.clock(), reason=self._reason))

    def _validate(self, picture, capture):
        now = time.monotonic_ns()
        if not isinstance(capture, dict) or not isinstance(picture, Image.Image):
            raise ValueError('Invalid field capture')
        stamp, start = capture.get('captured_ns'), capture.get('capture_started_ns')
        size, crop = capture.get('window_size'), capture.get('crop')
        if (type(stamp) is not int or type(start) is not int or not 0 <= start <= stamp <= now
                or now - start > 2_500_000_000 or capture.get('source_scope') != 'alien_field'
                or capture.get('capture_method') not in ('xcomposite_named_window_pixmap', 'x11_client_window_drawable')
                or type(capture.get('pid')) is not int or capture['pid'] < 1
                or type(capture.get('window_id')) is not int or capture['window_id'] < 1
                or capture['pid'] != self.binding.get('pid') or capture['window_id'] != self.binding.get('window_id')
                or not isinstance(size, (list, tuple)) or len(size) != 2
                or not isinstance(crop, (list, tuple)) or len(crop) != 4
                or any(type(v) is not int for v in [*size, *crop])
                or min(*size, crop[2], crop[3]) < 1 or min(crop[:2]) < 0
                or size[0] * size[1] > MAX_PIXELS or picture.width * picture.height > MAX_PIXELS
                or crop[0] + crop[2] > size[0] or crop[1] + crop[3] > size[1]
                or tuple(crop[2:]) != picture.size or list(size) != self.binding.get('size')
                or list(crop) != self.binding.get('crop')
                or (capture.get('native_tick') is not None and
                    (type(capture['native_tick']) is not int or capture['native_tick'] < 0))):
            raise ValueError('Field sample has invalid binding, geometry, method, scope or freshness')
        wire = json.dumps(capture, allow_nan=False)
        if len(wire) > 8192:
            raise ValueError('Capture metadata exceeds bound')
        return json.loads(wire)

    def _expire(self, next_due, discarded=None):
        """Account only in-window slots; an attempted discarded sample is separate."""
        self._finish('expired', 'deadline')
        if self._phase != 'expired' or self._expiry_accounted:
            return
        self._expiry_accounted = True
        if discarded is not None:
            self._gaps.append(dict(kind='discarded_capture', from_time=discarded,
                                   to_time=self.deadline, reason='deadline'))
        # Slots are half-open [started, deadline); tolerate floating point drift.
        missed = max(0, math.ceil((self.deadline - next_due) * self.fps - 1e-6))
        if missed:
            self._missed += missed
            self._gaps.append(dict(kind='missed_cadence', from_time=next_due,
                                   to_time=self.deadline, missed_slots=missed))

    def _append_manifest(self, frame):
        path = self.directory / 'manifest.jsonl'
        offset = path.stat().st_size
        try:
            with path.open('a') as manifest:
                manifest.write(json.dumps(frame, sort_keys=True, allow_nan=False) + '\n')
        except Exception:
            self._rollback_manifest(offset)
            raise
        return offset

    def _rollback_manifest(self, offset):
        # One capture owner: only this unpublished tail can exist after offset.
        # Already published manifest entries are never changed.
        with (self.directory / 'manifest.jsonl').open('r+b') as manifest:
            manifest.truncate(offset)

    def _run(self):
        due = self.started
        try:
            while not self._cancel.is_set():
                now = self.clock()
                if now >= self.deadline:
                    with self._lock:
                        self._expire(due)
                    break
                if now < due:
                    self._cancel.wait(min(due - now, self.deadline - now))
                    continue
                missed = max(0, int((now - due) * self.fps))
                with self._lock:
                    if self._cancel.is_set():
                        break
                    if self.clock() >= self.deadline:
                        self._expire(due)
                        break
                    if missed:
                        self._missed += missed
                        self._gaps.append(dict(kind='missed_cadence', from_time=due,
                                               to_time=due + missed / self.fps, missed_slots=missed))
                        due += missed / self.fps
                    attempted = due
                    due += 1 / self.fps
                    self._next_due, self._inflight = due, attempted
                picture = None
                path = None
                manifest_offset = None
                published = False
                try:
                    picture, capture = self.source.capture(copy.deepcopy(self.binding))
                    capture = self._validate(picture, capture)
                    raw = _png(picture)
                    with self._lock:
                        if self._cancel.is_set():
                            break
                        if self.clock() >= self.deadline:
                            self._expire(due, attempted)
                            break
                        if len(self._frames) >= self.max_frames or self._bytes + len(raw) > self.max_bytes:
                            self._finish('limited', 'original frame/byte limit', gap=True)
                            break
                        sequence = len(self._frames) + 1
                    path = self.directory / f'frame-{sequence:06}.png'
                    _write_new(path, raw)
                    frame = dict(sequence=sequence, frame_id=uuid.uuid4().hex, image_path=str(path),
                                 sha256=hashlib.sha256(raw).hexdigest(), capture_time=capture['captured_ns'] / 1e9,
                                 capture=capture, binding=copy.deepcopy(self._public_binding))
                    manifest_offset = self._append_manifest(frame)
                    # The final gate follows *all* potentially blocking publication I/O.
                    with self._lock:
                        if self._cancel.is_set():
                            break
                        if self.clock() >= self.deadline:
                            self._expire(due, attempted)
                            break
                        self._frames.append(frame)
                        self._bytes += len(raw)
                        self._inflight = None
                        published = True
                        if len(self._frames) >= self.max_frames:
                            self._finish('limited', 'original frame limit', gap=True)
                        continuing = self._phase == 'recording'
                    self._save_status()
                    if not continuing:
                        break
                finally:
                    if picture is not None:
                        picture.close()
                    with self._lock:
                        self._inflight = None
                    if not published:
                        if manifest_offset is not None:
                            self._rollback_manifest(manifest_offset)
                        if path is not None:
                            path.unlink(missing_ok=True)
        except Exception as error:
            with self._lock:
                self._finish('failed', f'{type(error).__name__}: {error}', gap=True)
        finally:
            with self._lock:
                if self._cancel.is_set():
                    self._finish('stopped', 'closed')
            # Notify outside the capture lock: the owner may take its manager lock
            # and request cancellation. Publish the terminal cause before final disk I/O.
            try:
                if self.on_terminal is not None:
                    self.on_terminal(self.status())
            finally:
                try:
                    self._save_status(force=True)
                except OSError as error:
                    with self._lock:
                        self._finish('failed', f'status storage failed: {error}', gap=True)

    def _panel(self, frames, path):
        columns = min(6, len(frames))
        rows = math.ceil(len(frames) / 6)
        panel = Image.new('RGB', (columns * 256, rows * 180), '#141923')
        try:
            draw = ImageDraw.Draw(panel)
            for index, frame in enumerate(frames):
                original = Path(frame['image_path'])
                raw = original.read_bytes()
                if len(raw) > MAX_IMAGE_BYTES or hashlib.sha256(raw).hexdigest() != frame['sha256']:
                    raise ValueError('Original frame hash/size mismatch')
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > MAX_PIXELS:
                        raise ValueError('Original frame exceeds pixel bound')
                    image.thumbnail((256, 160))
                    x, y = (index % 6) * 256, (index // 6) * 180
                    panel.paste(image, (x + (256 - image.width) // 2, y + (160 - image.height) // 2))
                    draw.text((x + 4, y + 162), f"#{frame['sequence']} {frame['capture_time']:.3f}s", fill='white')
            raw = _png(panel)
            try:
                _write_new(path, raw)
            except Exception:
                # No packet refers to this path until the write/close succeeds.
                path.unlink(missing_ok=True)
                raise
            return hashlib.sha256(raw).hexdigest()
        finally:
            panel.close()

    def packet(self, after_sequence: int = 0, max_frames: int = 60):
        if type(after_sequence) is not int or not 0 <= after_sequence <= 466:
            raise ValueError('Invalid stream cursor')
        if type(max_frames) is not int or not 1 <= max_frames <= 60:
            raise ValueError('Intervals contain 1 to 60 frames')
        with self._packets_lock:
            if after_sequence in self._packets:
                return copy.deepcopy(self._packets[after_sequence])
            with self._lock:
                frames = copy.deepcopy(self._frames[after_sequence:after_sequence + max_frames])
                if not frames:
                    return None
                detail = copy.deepcopy(self._frames[-1])
                state = self._state()
            panel_path = self.directory / f'interval-after-{after_sequence:06}.png'
            digest = self._panel(frames, panel_path)
            result = dict(packet_id=uuid.uuid4().hex, panel_kind='temporal_overview', frame_count=len(frames),
                          sequence_start=frames[0]['sequence'], sequence_end=frames[-1]['sequence'],
                          next_sequence=frames[-1]['sequence'], frames=frames,
                          interval_started=frames[0]['capture_time'], interval_ended=frames[-1]['capture_time'],
                          image_path=str(panel_path), sha256=digest, detail_image_path=detail['image_path'],
                          detail_sha256=detail['sha256'], detail_frame_id=detail['frame_id'],
                          detail_sequence=detail['sequence'], detail_capture_time=detail['capture_time'],
                          detail_capture=detail['capture'], detail_frame=detail,
                          capture_status=state, gaps=state['gaps'])
            self._packets[after_sequence] = result
            return copy.deepcopy(result)
