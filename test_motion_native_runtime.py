"""Exercise the production C++ worker branch with controlled capture timing."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

import motion_native
import motion_tracker
from motion_api import MotionAPIServer, StateStore
from test_motion_features import pose


class NativeWorkerTests(unittest.TestCase):
    def test_headless_cli_reports_camera_startup_failure_to_callers(self):
        # An invalid device must fail the actual CLI, so service supervisors
        # cannot mistake an inference/camera startup failure for success.
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            missing_device = str(Path(directory) / "missing-camera")
            script = (
                "from pathlib import Path; import motion_tracker; "
                "motion_tracker.ROOT = Path(__import__('sys').argv[1]); "
                "__import__('sys').argv = ['meganan', '--no-gui', '--port', '0', "
                "'--duration', '5', '--device', __import__('sys').argv[2]]; "
                "raise SystemExit(motion_tracker.main())"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, directory, missing_device],
                cwd=root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            reports = list(Path(directory).glob("runs/*/summary.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertTrue(report["error"])
            self.assertTrue(report["final"])
            self.assertEqual(report["processed_frames"], 0)

    def run_pipeline(self, actions, *, limit=30, on_infer=None):
        stop = threading.Event()
        settings = motion_tracker.RuntimeSettings(limit)
        store = StateStore()
        states = []
        times = []
        face_points = np.full((478, 3), .4, dtype=np.float32)
        face_points[:, 2] = -.1

        class Camera:
            closed = False
            sequence = 0

            def start(self):
                return self

            def next(self, after):
                self.sequence += 1
                if not actions:
                    stop.set()
                    return None
                action = actions.pop(0)
                if isinstance(action, Exception):
                    raise action
                if action is None:
                    return None
                return (self.sequence, time.perf_counter() - action,
                        SimpleNamespace(width=640, height=480), 30.)

            def close(self):
                self.closed = True

        class Inference:
            closed = False

            def infer(self, frame, timestamp):
                times.append(time.perf_counter())
                if on_infer:
                    on_infer(len(times), settings, stop)
                return pose(), {"detected": True, "landmarks": face_points,
                                "bbox": [.3, .3, .5, .5]}

            def close(self):
                self.closed = True

        camera, inference = Camera(), Inference()
        original_publish = store.publish

        def publish(payload):
            state = original_publish(payload)
            states.append(state)
            return state

        args = SimpleNamespace(model="mediapipe", engine="cpp", device="controlled-input",
                               face_details=True, fps_limit=limit)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(motion_tracker, "ROOT", Path(directory)), \
                patch.object(motion_native, "NativeCamera", return_value=camera), \
                patch.object(motion_native, "NativeInference", return_value=inference), \
                patch.object(store, "publish", side_effect=publish), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            worker = motion_tracker.MotionWorker(args, store, stop, settings)
            worker.start()
            worker.join(timeout=5)
            if worker.is_alive():
                stop.set()
                worker.join(timeout=2)
                self.fail("Native worker did not stop")
            summary = json.loads((worker.run_dir / "summary.json").read_text())
        self.assertTrue(camera.closed)
        self.assertTrue(inference.closed)
        self.assertTrue(summary["final"])
        self.assertEqual(summary["engine"], "cpp")
        return states, summary, times, store

    def test_native_results_reach_api_and_stale_missing_error_clear_both_trackers(self):
        states, summary, times, store = self.run_pipeline(
            [0, 2, None, RuntimeError("Controlled camera disconnect")])
        first, stale, missing, error = states
        self.assertTrue(first["body_tracked"])
        self.assertTrue(first["face_tracked"])
        self.assertEqual(len(first["face_landmarks"]), 478)
        self.assertEqual(first["source_resolution"], [640, 480])
        self.assertEqual(stale["status"], "STALE_FRAME")
        self.assertEqual(missing["camera_status"], "waiting")
        self.assertEqual(error["status"], "ERROR")
        for state in (stale, missing, error):
            self.assertFalse(state["tracked"])
            self.assertEqual(state["face_landmarks"], [])
            self.assertIsNone(state["face_motion"])
            self.assertIsNone(state["motion_speed"])
            self.assertFalse(np.asarray(state["points"]).any())
        self.assertIn("Controlled camera disconnect", summary["error"])
        # Exercise real HTTP projection for native-produced state too.
        store.publish(first)
        api = MotionAPIServer(store, port=0).start()
        try:
            with urlopen(f"http://127.0.0.1:{api.address[1]}/api/v1/state", timeout=2) as response:
                received = json.load(response)
            for key in ("face_landmarks", "points", "angles", "face_center", "face_motion"):
                self.assertEqual(received[key], first[key])
            self.assertFalse({"image", "frame", "rgb", "pixels"} & received.keys())
        finally:
            api.close()
            store.close()

    def test_native_branch_reacts_to_live_fps_changes(self):
        def change(index, settings, stop):
            if index == 3:
                settings.set_fps_limit(10)
            if index == 7:
                stop.set()

        states, summary, times, store = self.run_pipeline([0] * 8, limit=5, on_infer=change)
        intervals = np.diff(times)
        self.assertEqual(len(times), 7)
        self.assertGreaterEqual(float(intervals[:2].min()), .18)
        self.assertGreaterEqual(float(intervals[2:].min()), .09)
        self.assertEqual([s["fps_limit"] for s in states], [5, 5, 10, 10, 10, 10, 10])
        self.assertEqual(states[3]["fps"], 0.)
        self.assertIsNone(summary["error"])
        store.close()

    def test_stop_at_one_fps_does_not_wait_for_next_inference(self):
        def finish(index, settings, stop):
            stop.set()

        started = time.perf_counter()
        states, summary, times, store = self.run_pipeline([0] * 3, limit=1, on_infer=finish)
        self.assertLess(time.perf_counter() - started, .9)
        self.assertEqual(len(times), 1)
        self.assertIsNone(summary["error"])
        store.close()


if __name__ == "__main__":
    unittest.main()
