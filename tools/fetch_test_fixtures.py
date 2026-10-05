#!/usr/bin/env python3
"""Fetch hash-pinned public MediaPipe parity fixtures outside the repository.

Usage:
  MOTION_TEST_IMAGE_DIR=$(python tools/fetch_test_fixtures.py) \
    .venv/bin/python -m unittest -v test_motion_native_inference

These are upstream public test images, not camera recordings. They are fetched
on demand and are not vendored or distributed with meganan.
"""

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request


FIXTURES = {
    "portrait.jpg": "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e",
    "pose.jpg": "c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b",
}
BASE_URL = "https://storage.googleapis.com/mediapipe-assets/"


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in FIXTURES.items():
        path = destination / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with urllib.request.urlopen(BASE_URL + name, timeout=30) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Upstream fixture changed: {name}; SHA-256 verification failed")
        with tempfile.NamedTemporaryFile(dir=destination, prefix=name + ".", delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(data)
                output.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return destination.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path(tempfile.gettempdir()) / "meganan-public-fixtures")
    args = parser.parse_args()
    print(fetch(args.destination))


if __name__ == "__main__":
    main()
