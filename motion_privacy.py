"""Prevent the operating system from writing camera buffers into crash dumps."""
import ctypes
import os
import resource
import sys


def disable_process_dumps():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform.startswith("linux"):
        # RLIMIT_CORE alone is ignored when Linux pipes dumps to a collector.
        # PR_SET_DUMPABLE=0 also prevents systemd-coredump from saving frames.
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        if prctl(4, 0, 0, 0, 0) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
