#!/bin/sh
set -eu
MOTION_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$MOTION_ROOT"
if [ -f "$MOTION_ROOT/.runtime/usr/lib/libtk8.6.so" ]; then
    export LD_LIBRARY_PATH="$MOTION_ROOT/.runtime/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TCL_LIBRARY="$MOTION_ROOT/.runtime/usr/lib/tcl8.6"
    export TK_LIBRARY="$MOTION_ROOT/.runtime/usr/lib/tk8.6"
fi
"$MOTION_ROOT/.venv/bin/python" "$MOTION_ROOT/ensure_native.py" "$@"
exec "$MOTION_ROOT/.venv/bin/python" "$MOTION_ROOT/motion_tracker.py" "$@"
