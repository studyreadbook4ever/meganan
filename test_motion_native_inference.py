"""Native ABI, lifecycle, and model parity checks (never open the camera).

For detection parity, set MOTION_TEST_IMAGE_DIR to a directory containing the
official MediaPipe test images portrait.jpg and pose.jpg. The optional fixture
test uses only those public test assets; no user camera frame is saved.
Run tools/fetch_test_fixtures.py to download and verify their pinned SHA-256s.
"""

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

# A native regression must not write a memory dump containing image buffers.
from motion_privacy import disable_process_dumps
disable_process_dumps()

import numpy as np

import _motion_native as native
from motion_face import FaceTracker
from motion_pose_mediapipe import MediaPipePose
from motion_topology import FACE_CONTOURS


ROOT = Path(__file__).resolve().parent
SDK = Path(importlib.util.find_spec("mediapipe").origin).parent
LIBRARY = str(SDK / "tasks/c/libmediapipe.so")
POSE = str(ROOT / "pose_probe_assets/pose_landmarker_lite.task")
FACE = str(ROOT / "pose_probe_assets/face_landmarker.task")


def assert_abi(test):
    from mediapipe.tasks.python.core.base_options_c import MpBaseOptionsC
    from mediapipe.tasks.python.components.containers.landmark_c import (
        MpNormalizedLandmarkC, MpNormalizedLandmarksC,
    )
    from mediapipe.tasks.python.vision.pose_landmarker import (
        MpPoseLandmarkerOptionsC, MpPoseLandmarkerResultC,
    )
    from mediapipe.tasks.python.vision.face_landmarker import (
        MpFaceLandmarkerOptionsC, MpFaceLandmarkerResultC,
    )
    structs = (
        MpBaseOptionsC, MpNormalizedLandmarkC, MpNormalizedLandmarksC,
        MpPoseLandmarkerOptionsC, MpPoseLandmarkerResultC,
        MpFaceLandmarkerOptionsC, MpFaceLandmarkerResultC,
    )
    actual = native.mediapipe_abi_layout()
    for struct in structs:
        expected = {"size": ctypes.sizeof(struct)}
        expected.update((name, getattr(struct, name).offset) for name, _ in struct._fields_)
        test.assertEqual(actual[struct.__name__[:-1]], expected, struct.__name__)


class NativeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Never call a model using an ABI that this installed SDK does not match.
        assert_abi(unittest.TestCase())
        cls.engine = native.NativeLandmarkers(LIBRARY, POSE, FACE)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_abi_matches_installed_sdk(self):
        assert_abi(self)

    def test_blank_output_matches_python_adapters(self):
        with MediaPipePose() as pose, FaceTracker() as face:
            frame = np.zeros((96, 128, 3), np.uint8)
            expected_pose, expected_face = pose.infer(frame, 1), face.infer(frame, 1)
            actual_pose, actual_face = self.engine.infer_bgr(frame, 1)
        np.testing.assert_array_equal(actual_pose, expected_pose)
        np.testing.assert_array_equal(actual_face["landmarks"], expected_face["landmarks"])
        self.assertEqual(actual_face["detected"], False)
        self.assertEqual(actual_face["bbox"], None)
        self.assertEqual(actual_face["contours"], FACE_CONTOURS)
        self.assertEqual(actual_pose.dtype, np.float32)
        self.assertEqual(actual_face["landmarks"].dtype, np.float32)

    def test_invalid_frames_and_timestamps_do_not_poison_engine(self):
        invalid = [None, [], np.zeros((16, 16), np.uint8),
                   np.zeros((16, 16, 4), np.uint8), np.zeros((16, 16, 3), np.float32),
                   np.zeros((0, 16, 3), np.uint8)]
        for frame in invalid:
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError):
                self.engine.infer_bgr(frame, 1)
        frame = np.zeros((48, 64, 3), np.uint8)
        for timestamp in (-1, float("nan"), float("inf"), 1e30):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.engine.infer_bgr(frame, timestamp)
        pose, face = self.engine.infer_bgr(frame, 2)
        self.assertEqual(pose.shape, (17, 3))
        self.assertFalse(face["detected"])

    def test_noncontiguous_arrays_and_duplicate_timestamps(self):
        frame = np.zeros((64, 80, 3), np.uint8)[::-1, ::2, ::-1]
        for timestamp in (3.0, 3.0, 0.0):
            pose, face = self.engine.infer_bgr(frame, timestamp)
            np.testing.assert_array_equal(pose, np.zeros((17, 3), np.float32))
            self.assertFalse(face["detected"])

    def test_face_only_and_pose_only_modes_close_idempotently(self):
        for pose_path, face_path in ((POSE, ""), ("", FACE)):
            with self.subTest(pose=bool(pose_path), face=bool(face_path)):
                engine = native.NativeLandmarkers(LIBRARY, pose_path, face_path)
                try:
                    result = engine.infer_bgr(np.zeros((48, 64, 3), np.uint8), 0)
                    self.assertEqual(result[0].shape, (17, 3))
                    self.assertEqual(result[1]["landmarks"].shape, (0, 3))
                finally:
                    engine.close()
                engine.close()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    engine.infer_bgr(np.zeros((48, 64, 3), np.uint8), 1)

    def test_missing_models_and_partial_constructor_cleanup(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            native.NativeLandmarkers(LIBRARY)
        with self.assertRaisesRegex(RuntimeError, "Model file not found"):
            native.NativeLandmarkers(LIBRARY, POSE, "/nonexistent/motion-face.task")
        engine = native.NativeLandmarkers(LIBRARY, "", FACE)
        engine.close()

    def test_inference_and_close_on_separate_threads_do_not_deadlock(self):
        engine = native.NativeLandmarkers(LIBRARY, POSE, FACE)
        entered = threading.Event()
        errors = []

        def infer_until_closed():
            try:
                entered.set()
                for index in range(100):
                    engine.infer_bgr(np.zeros((480, 640, 3), np.uint8), index / 30)
            except RuntimeError as error:
                if "closed" not in str(error):
                    errors.append(error)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=infer_until_closed, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(2))
        # Permit the worker to enter native inference while this thread retains
        # its independent scheduling capability (the engine must release GIL).
        time.sleep(0.025)
        closer = threading.Thread(target=engine.close, daemon=True)
        closer.start()
        closer.join(5)
        worker.join(5)
        self.assertFalse(closer.is_alive(), "close blocked while inference needed the GIL")
        self.assertFalse(worker.is_alive(), "inference failed to return after close")
        self.assertEqual(errors, [])
        engine.close()

    def test_native_only_worker_shutdown_without_python_sdk_pinning(self):
        # Importing the Python MediaPipe SDK masks DSO unloading problems by
        # retaining its ctypes handle. Exercise real app lifetime in isolation.
        code = '''
from motion_privacy import disable_process_dumps
disable_process_dumps()
import sys, threading, numpy as np
sys.path.insert(0, sys.argv[1])
import _motion_native as native
assert "mediapipe" not in sys.modules
errors = []
def work():
    try:
        for _ in range(2):
            engine = native.NativeLandmarkers(sys.argv[2], sys.argv[3], sys.argv[4])
            engine.infer_bgr(np.zeros((120, 160, 3), np.uint8), 0)
            engine.close()
            del engine
    except BaseException as error:
        errors.append(error)
worker = threading.Thread(target=work)
worker.start()
worker.join()
assert not errors, errors
assert "mediapipe" not in sys.modules
print("native-only worker shutdown passed")
'''
        result = subprocess.run([
            sys.executable, "-X", "faulthandler", "-c", code,
            str(Path(native.__file__).parent), LIBRARY, POSE, FACE,
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native-only worker shutdown passed", result.stdout)

    @unittest.skipUnless(os.environ.get("MOTION_TEST_IMAGE_DIR"), "Set MOTION_TEST_IMAGE_DIR for public-image parity")
    def test_detected_face_and_body_match_python_on_public_images(self):
        import cv2
        directory = Path(os.environ["MOTION_TEST_IMAGE_DIR"])
        faces_seen = bodies_seen = 0
        engine = native.NativeLandmarkers(LIBRARY, POSE, FACE)
        try:
            with MediaPipePose() as pose, FaceTracker() as face:
                timestamp = 0
                for name in ("portrait.jpg", "pose.jpg"):
                    frame = cv2.imread(str(directory / name))
                    self.assertIsNotNone(frame, name)
                    # Original, mirror, translate: exercise tracking-state
                    # updates and strided BGR inputs over a deterministic clip.
                    frames = [frame, frame[:, ::-1], np.roll(frame, 8, axis=1)]
                    for image in frames:
                        timestamp += 1 / 30
                        expected_pose = pose.infer(image, timestamp)
                        expected_face = face.infer(image, timestamp)
                        actual_pose, actual_face = engine.infer_bgr(image, timestamp)
                        np.testing.assert_allclose(actual_pose, expected_pose, rtol=0, atol=1e-6)
                        self.assertEqual(actual_face["detected"], expected_face["detected"])
                        np.testing.assert_allclose(actual_face["landmarks"], expected_face["landmarks"], rtol=0, atol=1e-6)
                        if actual_face["bbox"] is None:
                            self.assertIsNone(expected_face["bbox"])
                        else:
                            np.testing.assert_allclose(actual_face["bbox"], expected_face["bbox"], rtol=0, atol=1e-6)
                        faces_seen += actual_face["detected"]
                        bodies_seen += bool(np.count_nonzero(actual_pose[:, 2] > 0.5))
        finally:
            engine.close()
        self.assertGreater(faces_seen, 0, "Fixture sequence must exercise actual face output")
        self.assertGreater(bodies_seen, 0, "Fixture sequence must exercise actual body output")


@unittest.skipUnless(shutil.which("c++"), "Fault-injection fixture requires the build compiler")
class NativeInferenceCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="motion-abi-failures-")
        cls.library_path = str(Path(cls.directory.name) / "libfaults.so")
        subprocess.run([
            "c++", "-std=c++17", "-shared", "-fPIC", "-O1",
            "-I" + str(ROOT / "native/third_party/mediapipe"),
            str(ROOT / "native/tests/mediapipe_failure_shim.cpp"),
            "-o", cls.library_path,
        ], check=True, capture_output=True)
        cls.library = ctypes.CDLL(cls.library_path)
        for name in ("MotionTestFailFaceCreate", "MotionTestFailFaceClose", "MotionTestFailPoseInfer"):
            getattr(cls.library, name).restype = None

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tearDown(self):
        self.assertEqual(self.library.MotionTestTasks(), 0)
        self.assertEqual(self.library.MotionTestImages(), 0)
        self.assertEqual(self.library.MotionTestErrors(), 0)

    def test_failed_second_model_creation_closes_first_model(self):
        self.library.MotionTestFailFaceCreate()
        with self.assertRaisesRegex(RuntimeError, "injected MediaPipe failure"):
            native.NativeLandmarkers(self.library_path, POSE, FACE)

    def test_failed_inference_releases_image_and_error_and_can_retry(self):
        engine = native.NativeLandmarkers(self.library_path, POSE, FACE)
        try:
            self.library.MotionTestFailPoseInfer()
            with self.assertRaisesRegex(RuntimeError, "injected MediaPipe failure"):
                engine.infer_bgr(np.zeros((48, 64, 3), np.uint8), 0)
            self.assertEqual(self.library.MotionTestImages(), 0)
            self.assertEqual(self.library.MotionTestErrors(), 0)
            self.assertEqual(engine.infer_bgr(np.zeros((48, 64, 3), np.uint8), 0)[0].shape, (17, 3))
        finally:
            engine.close()

    def test_failed_close_closes_other_model_and_retains_failed_handle_for_retry(self):
        engine = native.NativeLandmarkers(self.library_path, POSE, FACE)
        self.library.MotionTestFailFaceClose()
        try:
            with self.assertRaisesRegex(RuntimeError, "Close face landmarker.*injected"):
                engine.close()
            self.assertEqual(self.library.MotionTestTasks(), 1)
            self.assertEqual(self.library.MotionTestErrors(), 0)
            engine.close()
            engine.close()
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
