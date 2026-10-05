# Publication validation — 2026-10-05

This records the source publication checks for meganan. It is not a claim that
all operating systems, CPUs, cameras, or Python versions have been tested.

- Environment: Linux x86_64 (Arch Linux), CPython 3.14.5, X11/Tk, CPU inference.
- Fresh virtual environment without system-site packages: both requirements
  files installed successfully; `python -m pip check` reported no broken
  requirements. Exactly one OpenCV distribution is installed:
  `opencv-contrib-python==5.0.0.93`.
- Native build: CMake and C++17 build succeeded against the pinned SDK headers.
- Full unittest discovery with the hash-verified public image fixtures:
  **125 tests passed in 39.950 seconds, no failures or skips**. This includes
  real Tk child-process checks, numerical parity against the Python reference,
  camera-error cleanup, and JSON/SSE behavior. Unit tests do not acquire the
  real webcam.
- Fresh-environment live camera/Tk/JSON/SSE check: 139 GUI updates, 140 streamed states, 131 tracked-face updates; FPS slider exercised at 6 → 12 → 1 → 6. No image canvas items, no errors, and the camera reopened after shutdown.
- Real headless launcher: 5-second run processed 19 frames and exited successfully with no worker error.
- Camera startup failure: the headless CLI returns a nonzero exit code and
  finalizes its aggregate report; the test uses a nonexistent device.
- Branding: GUI, CLI help, HTTP index, OpenAPI title and server header use
  `meganan`. Internal `motion_*` import paths remain unchanged.
- Documentation: local README links and example syntax checked. Distribution
  package commands for Ubuntu/Debian/Fedora are installation guidance, not
  records of end-to-end tests on those systems.
- License review: the existing root Unlicense remains scoped to independently
  authored code. Third-party model/code licenses, copyright, adaptation notices,
  and upstream NOTICE are retained. All 31 license-text manifest hashes match;
  model and vendor-header provenance was also checked. See
  [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt).
- Publication inventory excludes virtual environments, local runtime bundles,
  compiled extensions, generated measurements, caches, and camera photographs.
  The README image is a capture of the running geometry-only GUI.

Reproduce the unit checks from the repository root after the README setup:

```sh
./build_native.sh
.venv/bin/python -m pip check
.venv/bin/python tools/fetch_test_fixtures.py --destination /tmp/meganan-public-fixtures
MOTION_TEST_IMAGE_DIR=/tmp/meganan-public-fixtures .venv/bin/python -m unittest discover -p 'test_motion_*.py' -v
```

The original migration benchmarks, with their measurement limits, are preserved
separately in [CPP_ENGINE_VALIDATION.txt](../CPP_ENGINE_VALIDATION.txt).
