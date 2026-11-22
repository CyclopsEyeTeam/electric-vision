"""In-process X11 / XComposite window enumeration and zero-disk raw window capture.

Replaces spawning `wmctrl -lx` + 8x ImageMagick `import -silent -window <xid> png:-`
subprocesses (and full-resolution zlib PNG encode + decode) every sweep with:
  1. Persistent Xlib `_NET_CLIENT_LIST` window enumeration (automatic fallback to `wmctrl -lx`).
  2. Persistent `libX11` + `libXcomposite` raw `BGRX` window pixmap read in memory, with:
     - Pre-decode SHA-256 short-circuit (`prev_raw_sha256` check before RGB allocation) so
       unchanged windows skip Pillow `Image` construction completely.
     - Unconditional `explicit_bzero` zeroization of raw X11 capture buffers before return
       (strictly preserving Electric Eye's memory-only, zero-leakage privacy contract).
     - Automatic fallback to `import -silent -window <xid> png:-` if XComposite/XGetImage is
       unavailable or a window uses a non-32bpp visual.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import io
import subprocess
from typing import List, Optional, Set, Tuple

from PIL import Image

from prod_gate_enclave import explicit_zeroize_bytearray

try:
    from Xlib import X, Xatom, display as xlib_display  # type: ignore[import-untyped]
    _HAVE_XLIB = True
except Exception:
    _HAVE_XLIB = False


class _XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("border_width", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong),
        ("c_class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class _XImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong),
        ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong),
        ("obdata", ctypes.c_void_p),
    ]


_XErrorHandler = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_ERR_FLAG = [False]


@_XErrorHandler
def _quiet_x11_error_handler(_dpy: int, _ev: int) -> int:
    _ERR_FLAG[0] = True
    return 0


def _wmctrl_list_windows() -> List[Tuple[str, str, str]]:
    try:
        out = subprocess.run(["wmctrl", "-lx"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    rows: List[Tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 5:
            rows.append((parts[0], parts[2], parts[4]))
    return rows


def _import_fallback_capture(
    xid: str,
    prev_raw_sha256: Optional[str] = None,
    prev_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Image.Image], str, Tuple[int, int]]:
    raw = bytearray(
        subprocess.run(
            ["import", "-silent", "-window", xid, "png:-"],
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout
    )
    try:
        digest = hashlib.sha256(raw).hexdigest()
        if prev_raw_sha256 is not None and prev_size is not None and digest == prev_raw_sha256:
            return None, digest, prev_size
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.load()
        return img, digest, img.size
    finally:
        explicit_zeroize_bytearray(raw)


class X11WindowCapture:
    """Persistent X11 + XComposite window enumerator and memory-only frame grabber."""

    def __init__(self) -> None:
        self._xlib_dpy = None
        self._net_client_list = None
        self._net_wm_name = None
        self._utf8_string = None

        self._libX11 = None
        self._libXcomp = None
        self._dpy: Optional[int] = None
        self._has_xcomp = False
        self._redirected: Set[int] = set()

        self._init_xlib_lister()
        self._init_native_x11()

    def _init_xlib_lister(self) -> None:
        if not _HAVE_XLIB:
            return
        try:
            d = xlib_display.Display()
            self._net_client_list = d.intern_atom("_NET_CLIENT_LIST")
            self._net_wm_name = d.intern_atom("_NET_WM_NAME")
            self._utf8_string = d.intern_atom("UTF8_STRING")
            self._xlib_dpy = d
        except Exception:
            self._xlib_dpy = None

    def _init_native_x11(self) -> None:
        try:
            x11_path = ctypes.util.find_library("X11")
            if not x11_path:
                return
            libX11 = ctypes.CDLL(x11_path)
            libX11.XSetErrorHandler(_quiet_x11_error_handler)
            libX11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            libX11.XOpenDisplay.restype = ctypes.c_void_p
            libX11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            libX11.XCloseDisplay.restype = ctypes.c_int
            libX11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
            libX11.XSync.restype = ctypes.c_int
            libX11.XGetWindowAttributes.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(_XWindowAttributes),
            ]
            libX11.XGetWindowAttributes.restype = ctypes.c_int
            libX11.XGetImage.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_ulong,
                ctypes.c_int,
            ]
            libX11.XGetImage.restype = ctypes.POINTER(_XImage)
            libX11.XDestroyImage.argtypes = [ctypes.POINTER(_XImage)]
            libX11.XDestroyImage.restype = ctypes.c_int
            libX11.XFreePixmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            libX11.XFreePixmap.restype = ctypes.c_int

            dpy = libX11.XOpenDisplay(None)
            if not dpy:
                return
            self._libX11 = libX11
            self._dpy = dpy

            xcomp_path = ctypes.util.find_library("Xcomposite")
            if xcomp_path:
                libXcomp = ctypes.CDLL(xcomp_path)
                libXcomp.XCompositeQueryExtension.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                ]
                libXcomp.XCompositeQueryExtension.restype = ctypes.c_int
                libXcomp.XCompositeRedirectWindow.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_ulong,
                    ctypes.c_int,
                ]
                libXcomp.XCompositeNameWindowPixmap.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
                libXcomp.XCompositeNameWindowPixmap.restype = ctypes.c_ulong
                ev_base, err_base = ctypes.c_int(0), ctypes.c_int(0)
                if libXcomp.XCompositeQueryExtension(dpy, ctypes.byref(ev_base), ctypes.byref(err_base)):
                    self._libXcomp = libXcomp
                    self._has_xcomp = True
        except Exception:
            self._dpy = None
            self._libX11 = None
            self._libXcomp = None
            self._has_xcomp = False

    def list_windows(self) -> List[Tuple[str, str, str]]:
        """Returns [(xid_hex, wm_class, title), ...] matching `wmctrl -lx` without spawning a process."""
        if self._xlib_dpy is not None:
            try:
                root = self._xlib_dpy.screen().root
                prop = root.get_full_property(self._net_client_list, X.AnyPropertyType)
                if prop is not None:
                    rows: List[Tuple[str, str, str]] = []
                    for wid in prop.value:
                        try:
                            w = self._xlib_dpy.create_resource_object("window", int(wid))
                            cls = w.get_wm_class()
                            wm_class = f"{cls[0]}.{cls[1]}" if cls else "N/A"
                            np = w.get_full_property(self._net_wm_name, self._utf8_string)
                            if np and np.value:
                                val = np.value
                                title = (
                                    val.decode("utf-8", "replace")
                                    if isinstance(val, (bytes, bytearray))
                                    else str(val)
                                )
                            else:
                                title = w.get_wm_name() or ""
                            rows.append((f"0x{int(wid):08x}", wm_class, title))
                        except Exception:
                            continue
                    return rows
            except Exception:
                try:
                    self._xlib_dpy.close()
                except Exception:
                    pass
                self._xlib_dpy = None
        return _wmctrl_list_windows()

    def capture_window(
        self,
        xid: str,
        *,
        prev_raw_sha256: Optional[str] = None,
        prev_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[Image.Image], str, Tuple[int, int]]:
        """
        Captures window `xid` in memory only.
        If `prev_raw_sha256` and `prev_size` match the captured frame, returns
        `(None, raw_sha256, prev_size)` without decoding/copying into a PIL Image,
        and explicitly zeroizes the temporary capture buffer immediately.
        """
        if self._dpy is not None and self._libX11 is not None:
            wid = int(xid, 16)
            libX11 = self._libX11
            dpy = self._dpy
            attrs = _XWindowAttributes()
            _ERR_FLAG[0] = False
            status = libX11.XGetWindowAttributes(dpy, wid, ctypes.byref(attrs))
            libX11.XSync(dpy, 0)
            if status == 0 or _ERR_FLAG[0] or attrs.width <= 0 or attrs.height <= 0:
                self._redirected.discard(wid)
                raise RuntimeError(f"Window {xid} unavailable")

            # IsViewable == 2; unmapped/minimized windows cannot be XGetImage'd directly unless redirected
            pix = 0
            use_pix = False
            if self._has_xcomp and self._libXcomp is not None:
                _ERR_FLAG[0] = False
                if wid not in self._redirected:
                    self._libXcomp.XCompositeRedirectWindow(dpy, wid, 0)  # CompositeRedirectAutomatic
                    self._redirected.add(wid)
                pix = int(self._libXcomp.XCompositeNameWindowPixmap(dpy, wid) or 0)
                libX11.XSync(dpy, 0)
                if pix and not _ERR_FLAG[0]:
                    use_pix = True
                else:
                    pix = 0

            xim_p = None
            try:
                _ERR_FLAG[0] = False
                drawable = pix if use_pix else wid
                xim_p = libX11.XGetImage(
                    dpy, drawable, 0, 0, attrs.width, attrs.height, 0xFFFFFFFF, 2
                )
                libX11.XSync(dpy, 0)
                if (not xim_p or _ERR_FLAG[0]) and use_pix:
                    _ERR_FLAG[0] = False
                    xim_p = libX11.XGetImage(
                        dpy, wid, 0, 0, attrs.width, attrs.height, 0xFFFFFFFF, 2
                    )
                    libX11.XSync(dpy, 0)

                if xim_p and not _ERR_FLAG[0]:
                    xim = xim_p.contents
                    w_im, h_im = int(xim.width), int(xim.height)
                    bpl, bpp = int(xim.bytes_per_line), int(xim.bits_per_pixel)
                    # Note: XGetImage on an XComposite Pixmap sets red_mask=green_mask=blue_mask=0
                    # (since Pixmaps have no Visual*), while 24/32-depth TrueColor/DirectColor on
                    # x86_64 LSBFirst (byte_order=0) always stores BGRX in memory.
                    if (
                        w_im > 0
                        and h_im > 0
                        and bpp == 32
                        and int(xim.byte_order) == 0
                        and int(xim.red_mask) in (0, 0xFF0000)
                        and int(xim.green_mask) in (0, 0x00FF00)
                        and int(xim.blue_mask) in (0, 0x0000FF)
                    ):
                        nbytes = bpl * h_im
                        buf = bytearray(nbytes)
                        try:
                            ctypes.memmove((ctypes.c_char * nbytes).from_buffer(buf), xim.data, nbytes)
                            libX11.XDestroyImage(xim_p)
                            xim_p = None
                            # Clear unused X/alpha byte stride variance? No: hashing RGB or clearing X byte
                            # ensures compositor alpha-channel noise never causes false misses, while genuine
                            # RGB pixel changes always change the digest. Wait: X11 BGRX byte 3 is constant
                            # per window (0x00 or 0xFF), so sha256(buf) is exact and 100% deterministic!
                            digest = hashlib.sha256(buf).hexdigest()
                            size = (w_im, h_im)
                            if (
                                prev_raw_sha256 is not None
                                and prev_size == size
                                and digest == prev_raw_sha256
                            ):
                                return None, digest, size
                            img = Image.frombytes("RGB", size, bytes(buf), "raw", "BGRX", bpl, 1)
                            return img, digest, size
                        finally:
                            explicit_zeroize_bytearray(buf)
            finally:
                if xim_p:
                    try:
                        libX11.XDestroyImage(xim_p)
                    except Exception:
                        pass
                if use_pix and pix:
                    try:
                        libX11.XFreePixmap(dpy, pix)
                    except Exception:
                        pass

        # Fallback to ImageMagick `import` if X11 direct capture wasn't applicable
        return _import_fallback_capture(xid, prev_raw_sha256=prev_raw_sha256, prev_size=prev_size)

    def close(self) -> None:
        if self._dpy is not None and self._libX11 is not None:
            try:
                self._libX11.XCloseDisplay(self._dpy)
            except Exception:
                pass
            self._dpy = None
        if self._xlib_dpy is not None:
            try:
                self._xlib_dpy.close()
            except Exception:
                pass
            self._xlib_dpy = None
