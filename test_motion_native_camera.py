"""Native camera lifecycle checks that do not acquire the real webcam."""
import gc
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import _motion_native
from _motion_native import CameraFrame, NativeCamera


class NativeCameraLifecycleTests(unittest.TestCase):
    def test_start_disables_process_crash_image_retention(self):
        code = """
import ctypes, resource, sys
sys.path.insert(0, sys.argv[1])
from _motion_native import NativeCamera
camera = NativeCamera('/dev/null')
try:
    camera.start()
except RuntimeError:
    pass
finally:
    camera.close()
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(3, 0, 0, 0, 0) == 0  # Linux PR_GET_DUMPABLE
"""
        completed = subprocess.run([sys.executable, "-c", code,
                                    str(Path(_motion_native.__file__).resolve().parent)], capture_output=True,
                                   text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_invalid_device_error_is_immediate_and_close_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = NativeCamera(str(Path(directory) / "absent-camera"))
            started = time.perf_counter()
            with self.assertRaisesRegex(RuntimeError, "Cannot open"):
                camera.start()
            self.assertLess(time.perf_counter() - started, 2)
            with self.assertRaisesRegex(RuntimeError, "Cannot open"):
                camera.next(0)
            camera.close()
            camera.close()

    def test_regular_file_does_not_leak_descriptors(self):
        descriptor_dir = Path("/proc/self/fd")
        with tempfile.NamedTemporaryFile() as regular_file:
            before = len(list(descriptor_dir.iterdir()))
            for _ in range(8):
                camera = NativeCamera(regular_file.name)
                with self.assertRaisesRegex(RuntimeError, "Query camera capabilities"):
                    camera.start()
                camera.close()
                del camera
            gc.collect()
            self.assertEqual(len(list(descriptor_dir.iterdir())), before)

    def test_close_before_start(self):
        camera = NativeCamera()
        self.assertEqual(camera.device, "/dev/video0")
        camera.close()
        camera.close()
        self.assertIsNone(camera.next(0, 0))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            camera.start()

    def test_next_before_start_and_timeout_validation(self):
        camera = NativeCamera()
        with self.assertRaisesRegex(RuntimeError, "not been started"):
            camera.next(0, 0)
        for timeout in (-1, float("nan"), float("inf"), 3601, 1e300):
            with self.assertRaises(ValueError):
                camera.next(0, timeout)
        camera.close()

    def test_failed_start_destruction(self):
        for _ in range(8):
            camera = NativeCamera("/dev/null")
            with self.assertRaises(RuntimeError):
                camera.start()
            del camera
        gc.collect()

    def test_invalid_device_paths(self):
        for path in ("", "abc\x00def"):
            with self.assertRaises(ValueError):
                NativeCamera(path)

    def test_frame_has_no_public_pixel_access_or_constructor(self):
        self.assertTrue(hasattr(CameraFrame, "width"))
        self.assertTrue(hasattr(CameraFrame, "height"))
        self.assertFalse(hasattr(CameraFrame, "rgb"))
        self.assertFalse(hasattr(CameraFrame, "__array__"))
        with self.assertRaises(TypeError):
            CameraFrame()


if __name__ == "__main__":
    unittest.main()
