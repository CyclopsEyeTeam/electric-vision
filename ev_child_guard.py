"""Ties an Electric Vision helper process to the life of the process that started it."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import signal

_PR_SET_PDEATHSIG = 1


def die_with_parent() -> None:
    """Ask Linux to SIGKILL this helper when its parent exits, so a killed server leaves no daemons."""
    parent = multiprocessing.parent_process()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        return
    # A forked helper inherits the server's chat lock; drop it so only the server itself holds it.
    try:
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}").endswith("/server.lock"):
                    os.close(int(fd))
            except OSError:
                pass
    except OSError:
        pass
    # The parent may already have gone between fork and prctl; then nobody will signal us.
    if parent is not None and os.getppid() != parent.pid:
        os._exit(0)
