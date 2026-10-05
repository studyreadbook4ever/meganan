#!/bin/sh
set -eu
MOTION_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MOTION_PYTHON=${MOTION_PYTHON:-"$MOTION_ROOT/.venv/bin/python"}
if ! "$MOTION_PYTHON" -m pybind11 --cmakedir >/dev/null 2>&1; then
    echo "Install build requirements: $MOTION_PYTHON -m pip install -r $MOTION_ROOT/requirements-build.txt" >&2
    exit 1
fi
cmake -S "$MOTION_ROOT" -B "$MOTION_ROOT/build/native" \
    -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$MOTION_PYTHON" "$@"
cmake --build "$MOTION_ROOT/build/native" --parallel "${MOTION_BUILD_JOBS:-2}"
# Publish atomically: a running GUI can still have the old extension mapped.
# Linking/truncating that mapped file in place can crash the running process.
"$MOTION_PYTHON" - "$MOTION_ROOT" <<'PY'
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile
root = Path(sys.argv[1])
name = "_motion_native" + sysconfig.get_config_var("EXT_SUFFIX")
source = root / "build" / "native" / "python" / name
with tempfile.NamedTemporaryFile(dir=root, prefix=".motion-native-", delete=False) as output:
    temporary = Path(output.name)
try:
    shutil.copyfile(source, temporary)
    temporary.chmod(0o755)
    os.replace(temporary, root / name)
finally:
    temporary.unlink(missing_ok=True)
PY
