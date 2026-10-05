"""MoveNet preprocessing/inference parity against the retained Python backend."""
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import unittest

import cv2
import numpy as np

import _motion_native
from _motion_native import NativeMoveNet

ROOT = Path(__file__).resolve().parent
LIBRARY = Path(find_spec("ai_edge_litert").origin).parent / "libLiteRt.so"
MODEL = ROOT / "pose_probe_assets/movenet_lightning_int8_v4.tflite"


def reference_preprocess(image):
    height, width = image.shape[:2]
    scale = min(192 / width, 192 / height)
    rw, rh = round(width * scale), round(height * scale)
    rgb = cv2.cvtColor(cv2.resize(image, (rw, rh)), cv2.COLOR_BGR2RGB)
    left, top = (192 - rw) // 2, (192 - rh) // 2
    padded = cv2.copyMakeBorder(rgb, top, 192 - rh - top, left,
                               192 - rw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[np.newaxis]


class NativeMoveNetTests(unittest.TestCase):
    def test_native_only_process_exits_cleanly(self):
        # Importing the Python Interpreter wrapper can keep an extra DSO
        # reference alive and hide native unload/TLS destruction failures.
        code = """
import gc, sys
import numpy as np
sys.path.insert(0, sys.argv[3])
from _motion_native import NativeMoveNet
for _ in range(3):
    engine = NativeMoveNet(sys.argv[1], sys.argv[2])
    result = engine.infer_bgr(np.zeros((480, 640, 3), np.uint8))
    assert result.shape == (17, 3)
    engine.close()
    del engine
    gc.collect()
"""
        completed = subprocess.run([sys.executable, "-c", code, str(LIBRARY), str(MODEL),
                                    str(Path(_motion_native.__file__).resolve().parent)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_camera_size_preprocessing_matches_opencv(self):
        rng = np.random.default_rng(831)
        images = [np.zeros((480, 640, 3), np.uint8),
                  np.full((480, 640, 3), 255, np.uint8),
                  rng.integers(0, 256, (480, 640, 3), np.uint8)]
        for image in images:
            np.testing.assert_array_equal(NativeMoveNet.preprocess_bgr(image),
                                          reference_preprocess(image))

    def test_padding_and_channel_order(self):
        for height, width in ((480, 640), (640, 480), (192, 192), (241, 321)):
            image = np.empty((height, width, 3), np.uint8)
            image[:] = (15, 80, 220)
            np.testing.assert_array_equal(NativeMoveNet.preprocess_bgr(image),
                                          reference_preprocess(image))

    def test_invalid_input_is_rejected(self):
        for image in (np.zeros((10, 10), np.uint8),
                      np.zeros((10, 10, 4), np.uint8),
                      np.zeros((10, 10, 3), np.float32),
                      np.zeros((0, 10, 3), np.uint8),
                      np.zeros((10, 10, 3), np.uint8)[:, ::2]):
            with self.assertRaises(ValueError):
                NativeMoveNet.preprocess_bgr(image)

    def test_native_inference_matches_python_interpreter(self):
        from ai_edge_litert.interpreter import Interpreter
        reference = Interpreter(model_path=str(MODEL), num_threads=2)
        reference.allocate_tensors()
        inp = reference.get_input_details()[0]
        out = reference.get_output_details()[0]
        native = NativeMoveNet(str(LIBRARY), str(MODEL))
        try:
            rng = np.random.default_rng(519)
            for height, width in ((480, 640), (640, 480), (192, 192)):
                image = rng.integers(0, 256, (height, width, 3), np.uint8)
                reference.set_tensor(inp["index"], reference_preprocess(image))
                reference.invoke()
                points = reference.get_tensor(out["index"])[0, 0]
                scale = min(192 / width, 192 / height)
                left = (192 - round(width * scale)) // 2
                top = (192 - round(height * scale)) // 2
                xy = (points[:, [1, 0]] * 192 - [left, top]) / scale / [width, height]
                expected = np.column_stack((xy, points[:, 2]))
                actual = native.infer_bgr(image)
                self.assertEqual(actual.shape, (17, 3))
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        finally:
            native.close()

    def test_closed_engine_rejects_inference_and_close_is_idempotent(self):
        engine = NativeMoveNet(str(LIBRARY), str(MODEL))
        engine.close()
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            engine.infer_bgr(np.zeros((480, 640, 3), np.uint8))

    def test_missing_library_and_model_propagate_error(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot load LiteRT"):
            NativeMoveNet("/not/a/library.so", str(MODEL))
        with self.assertRaisesRegex(RuntimeError, "Load MoveNet"):
            NativeMoveNet(str(LIBRARY), "/not/a/model.tflite")


if __name__ == "__main__":
    unittest.main()
