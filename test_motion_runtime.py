"""Camera-free regression tests for capture and worker failure/latency handling."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import motion_tracker
from test_motion_features import pose


class RecordingStore:
    def __init__(self):
        self.states = []

    def publish(self, state):
        snapshot = copy.deepcopy(state)
        self.states.append(snapshot)
        return snapshot


class FakeModel:
    def __init__(self):
        self.closed = False

    def infer(self, frame, timestamp):
        return pose()

    def close(self):
        self.closed = True


class TimedModel(FakeModel):
    """Record real inference starts; optional callback changes live settings."""

    def __init__(self, stop, count=6, on_frame=None):
        super().__init__()
        self.stop, self.count, self.on_frame = stop, count, on_frame
        self.starts = []

    def infer(self, frame, timestamp):
        self.starts.append(time.perf_counter())
        time.sleep(.002)
        if self.on_frame:
            self.on_frame(len(self.starts))
        if len(self.starts) >= self.count:
            self.stop.set()
        return pose()


class ScriptedCamera:
    def __init__(self, actions, stop):
        self.actions = iter(actions)
        self.stop = stop
        self.closed = False

    def start(self):
        return self

    def next(self, previous_sequence):
        try:
            action = next(self.actions)
        except StopIteration:
            self.stop.set()
            return None
        if isinstance(action, Exception):
            raise action
        return action() if callable(action) else action

    def close(self):
        self.closed = True


def sample(age=0):
    return 1, time.perf_counter() - age, np.zeros((480, 640, 3), dtype=np.uint8), 20.


class MotionRuntimeTests(unittest.TestCase):
    def run_worker(self, actions, *, stop=None, settings=None, model=None,
                   camera=None, face_model=None):
        stop = stop if stop is not None else threading.Event()
        camera = camera if camera is not None else ScriptedCamera(actions, stop)
        model = model if model is not None else FakeModel()
        store = RecordingStore()
        args = SimpleNamespace(model="movenet", device="fake-camera-only",
                               face_details=face_model is not None)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(motion_tracker, "ROOT", Path(temporary)), \
                patch.object(motion_tracker, "LatestCamera", return_value=camera), \
                patch.object(motion_tracker, "MoveNetPose", return_value=model), \
                patch.dict(sys.modules, {"motion_face": SimpleNamespace(
                    FaceTracker=lambda: face_model)}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            worker = motion_tracker.MotionWorker(args, store, stop, settings)
            worker.run()
            report = json.loads((worker.run_dir / "summary.json").read_text())
        return store.states, report, camera, model

    def test_missing_frames_and_camera_failure_clear_live_pose(self):
        states, report, camera, model = self.run_worker([
            sample, None, RuntimeError("Synthetic camera disconnect"),
        ])
        self.assertTrue(states[0]["tracked"])
        waiting, failed = states[1:]
        for state in (waiting, failed):
            self.assertFalse(state["tracked"])
            self.assertFalse(np.asarray(state["points"]).any())
            self.assertEqual(state["left_arm"], "UNKNOWN")
            self.assertIsNone(state["motion_speed"])
        self.assertEqual(waiting["camera_status"], "waiting")
        self.assertEqual(failed["status"], "ERROR")
        self.assertIn("Synthetic camera disconnect", report["error"])
        self.assertTrue(report["final"])
        self.assertTrue(camera.closed)
        self.assertTrue(model.closed)

    def test_source_frame_that_is_seconds_old_is_not_reported_as_live_tracking(self):
        states, report, camera, model = self.run_worker([lambda: sample(age=2)])
        self.assertTrue(states)
        self.assertFalse(any(state["tracked"] for state in states))
        self.assertTrue(camera.closed)
        self.assertTrue(model.closed)

    def test_capture_constructor_error_is_visible_to_consumer(self):
        camera = motion_tracker.LatestCamera("fake-camera-only")
        with patch.object(motion_tracker.cv2, "VideoCapture",
                          side_effect=RuntimeError("Synthetic startup failure")):
            camera._run()
        self.assertIn("Synthetic startup failure", camera.error)
        with self.assertRaisesRegex(RuntimeError, "Synthetic startup failure"):
            camera.next(0, timeout=0)

    def test_capture_consumes_newest_frame_and_releases_device(self):
        camera = motion_tracker.LatestCamera("fake-camera-only")
        first = np.full((3, 4, 3), 1, dtype=np.uint8)
        latest = np.full((3, 4, 3), 2, dtype=np.uint8)

        class Capture:
            released = False
            reads = 0

            def isOpened(self):
                return True

            def set(self, *unused):
                pass

            def read(self):
                self.reads += 1
                if self.reads == 2:
                    camera.stop_event.set()
                    return True, latest
                return True, first

            def release(self):
                self.released = True

        cap = Capture()
        with patch.object(motion_tracker.cv2, "VideoCapture", return_value=cap):
            camera._run()
        sequence, acquired, frame, fps = camera.next(0, timeout=0)
        self.assertEqual(sequence, 2)
        self.assertIs(frame, latest)
        self.assertIsNone(camera.next(sequence, timeout=0))
        self.assertTrue(cap.released)

    def test_runtime_settings_reject_invalid_limits_without_changing_current_value(self):
        settings = motion_tracker.RuntimeSettings(10)
        for value in (float("nan"), float("inf"), -float("inf"), 0, -1, 30.1, 31):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    settings.set_fps_limit(value)
                self.assertEqual(settings.get_fps_limit(), 10)
        settings.set_fps_limit(1)
        self.assertEqual(settings.get_fps_limit(), 1)
        settings.set_fps_limit(30)
        self.assertEqual(settings.get_fps_limit(), 30)

    def test_inference_start_rate_respects_five_and_ten_fps_caps(self):
        for limit in (5, 10):
            with self.subTest(limit=limit):
                stop = threading.Event()
                model = TimedModel(stop, count=6)
                settings = motion_tracker.RuntimeSettings(limit)
                states, report, camera, model = self.run_worker(
                    [sample] * 8, stop=stop, settings=settings, model=model)
                self.assertEqual(len(model.starts), 6)
                intervals = np.diff(model.starts)
                # Check only the maximum rate; slow scheduling never fails.
                self.assertGreaterEqual(float(intervals.sum()), 5 / limit * .95)
                self.assertGreaterEqual(float(intervals.min()), 1 / limit * .85)
                self.assertTrue(all(state["fps_limit"] == limit for state in states))
                self.assertEqual(report["fps_limit"], limit)
                self.assertTrue(report["final"])

    def test_live_limit_change_is_read_by_worker_and_published(self):
        stop = threading.Event()
        settings = motion_tracker.RuntimeSettings(5)

        def change_limit(frame_count):
            if frame_count == 3:
                settings.set_fps_limit(10)

        model = TimedModel(stop, count=7, on_frame=change_limit)
        states, report, camera, model = self.run_worker(
            [sample] * 9, stop=stop, settings=settings, model=model)
        self.assertEqual(len(model.starts), 7)
        intervals = np.diff(model.starts)
        self.assertGreaterEqual(float(intervals[:2].sum()), 2 / 5 * .95)
        self.assertGreaterEqual(float(intervals[2:].sum()), 4 / 10 * .95)
        self.assertEqual([state["fps_limit"] for state in states], [5, 5, 10, 10, 10, 10, 10])
        # The displayed rolling rate restarts when a different limit is read.
        self.assertEqual(states[3]["fps"], 0.)
        self.assertEqual(report["fps_limit"], 10)

    def test_stopping_interrupts_low_fps_pacing_wait(self):
        class ObservableStop(threading.Event):
            def __init__(self):
                super().__init__()
                self.wait_started = threading.Event()

            def wait(self, timeout=None):
                if timeout is not None and timeout > .5:
                    self.wait_started.set()
                return super().wait(timeout)

        stop = ObservableStop()
        camera = ScriptedCamera([sample] * 3, stop)
        model, store = FakeModel(), RecordingStore()
        args = SimpleNamespace(model="movenet", device="fake-camera-only", face_details=False)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(motion_tracker, "ROOT", Path(temporary)), \
                patch.object(motion_tracker, "LatestCamera", return_value=camera), \
                patch.object(motion_tracker, "MoveNetPose", return_value=model), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            worker = motion_tracker.MotionWorker(
                args, store, stop, motion_tracker.RuntimeSettings(1))
            worker.start()
            try:
                self.assertTrue(stop.wait_started.wait(timeout=2), "Worker never entered pacing wait")
                stop.set()
                worker.join(timeout=.75)
                self.assertFalse(worker.is_alive(), "Stop should interrupt the one-second FPS wait")
                self.assertEqual(worker.frames, 1)
                self.assertTrue(camera.closed)
                self.assertTrue(model.closed)
                report = json.loads((worker.run_dir / "summary.json").read_text())
                self.assertTrue(report["final"])
            finally:
                stop.set()
                worker.join(timeout=2)

    def test_close_errors_do_not_skip_other_resources_or_final_report(self):
        for failing_resource in ("camera", "model"):
            with self.subTest(failing_resource=failing_resource):
                stop = threading.Event()
                camera = ScriptedCamera([sample], stop)
                model = FakeModel()

                class FaceModel(FakeModel):
                    def infer(self, frame, timestamp):
                        return {"detected": False}

                face_model = FaceModel()
                failing = camera if failing_resource == "camera" else model

                def failed_close():
                    failing.closed = True
                    raise RuntimeError("Synthetic cleanup failure")

                failing.close = failed_close
                states, report, camera, model = self.run_worker(
                    [], stop=stop, camera=camera, model=model, face_model=face_model)
                self.assertTrue(camera.closed)
                self.assertTrue(model.closed)
                self.assertTrue(face_model.closed)
                self.assertTrue(report["final"])
                self.assertEqual(report["processed_frames"], 1)


if __name__ == "__main__":
    unittest.main()
