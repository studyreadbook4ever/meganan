"""C++ movement -> worker -> real HTTP/SSE -> real Tk, without camera input.

Only acquisition and model inference are scripted. The production worker takes
its C++ branch and uses the actual native analyzer and presentation classes.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class NativeGuiIntegrationTests(unittest.TestCase):
    def test_native_worker_http_sse_and_real_tk(self):
        if not os.environ.get("DISPLAY"):
            self.skipTest("Native Tk integration requires DISPLAY, but never a camera")
        root = Path(__file__).resolve().parent
        env = os.environ.copy()
        runtime_lib = root / ".runtime" / "usr" / "lib"
        if (runtime_lib / "libtk8.6.so").is_file():
            env["LD_LIBRARY_PATH"] = str(runtime_lib) + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            env["TCL_LIBRARY"] = str(runtime_lib / "tcl8.6")
            env["TK_LIBRARY"] = str(runtime_lib / "tk8.6")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--tk-child"],
                                cwd=root, env=env, capture_output=True, text=True, timeout=60)
        details = result.stdout + result.stderr
        if result.returncode == 77:
            self.skipTest(details.strip())
        self.assertEqual(result.returncode, 0, details)
        self.assertIn("Ran 5 tests", details)


def _run_tk_checks():
    import gc
    import json
    import math
    import queue
    import socket
    import tempfile
    import threading
    import time
    from types import SimpleNamespace
    from urllib.request import Request, urlopen
    from unittest.mock import patch

    import numpy as np
    import _motion_native
    import motion_native
    import motion_tracker
    from motion_api import StateStore, MotionAPIServer
    from test_motion_features import pose

    try:
        import tkinter as tk
        from motion_view import MotionView
    except (ImportError, OSError) as error:
        print(f"Tk unavailable: {error}")
        return 77
    try:
        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
    except tk.TclError as error:
        print(f"Tk unavailable: {error}")
        return 77

    class ObservableStop(threading.Event):
        def __init__(self):
            super().__init__()
            self.long_wait = threading.Event()

        def wait(self, timeout=None):
            if timeout is not None and timeout > .5:
                self.long_wait.set()
            return super().wait(timeout)

    class ScriptedNativeCamera:
        def __init__(self, stop):
            self.stop, self.queue = stop, queue.Queue()
            self.closed, self.sequence = False, 0

        def start(self):
            return self

        def next(self, previous_sequence):
            while not self.stop.is_set():
                try:
                    frame = self.queue.get(timeout=.01)
                except queue.Empty:
                    continue
                if isinstance(frame, Exception):
                    raise frame
                if frame is None:
                    return None
                self.sequence += 1
                return self.sequence, time.perf_counter() - frame.age, frame, 20.
            return None

        def close(self):
            self.closed = True

    class ScriptedNativeInference:
        def __init__(self):
            self.closed, self.starts, self.timestamps = False, [], []

        def infer(self, frame, timestamp):
            self.starts.append(time.perf_counter())
            self.timestamps.append(timestamp)
            return frame.points, frame.face

        def close(self):
            self.closed = True

    def mesh_detection(offset=(0., 0.)):
        theta = np.arange(478) * 2 * math.pi / 478
        mesh = np.column_stack((.5 + .12 * np.cos(theta),
                                .3 + .17 * np.sin(theta), np.full(478, -.02)))
        mesh[1, :2] = [.5, .3]
        mesh[:, :2] += offset
        return {"detected": True, "landmarks": mesh}

    def body_points():
        points = pose()
        points[5:11, 0] -= .1
        points[9, 1] = .1
        return points

    class NativeTkChecks(unittest.TestCase):
        def setUp(self):
            self.assertIs(motion_native.MotionAnalyzer, _motion_native.MotionAnalyzer)
            self.assertIs(motion_native.MotionPresentation, _motion_native.MotionPresentation)
            self.stop = ObservableStop()
            self.settings = motion_tracker.RuntimeSettings(30)
            self.store = StateStore()
            self.server = MotionAPIServer(self.store, port=0).start()
            self.url = f"http://{self.server.address[0]}:{self.server.address[1]}"
            self.camera = ScriptedNativeCamera(self.stop)
            self.model = ScriptedNativeInference()
            self.temporary = tempfile.TemporaryDirectory()
            self.patches = [
                patch.object(motion_tracker, "ROOT", Path(self.temporary.name)),
                patch.object(motion_native, "NativeCamera", return_value=self.camera),
                patch.object(motion_native, "NativeInference", return_value=self.model),
                patch.object(motion_tracker.cv2, "VideoCapture",
                             side_effect=AssertionError("No real camera in this integration test")),
            ]
            for item in self.patches:
                item.start()
            self.view = MotionView(self.request_close, self.settings.set_fps_limit, fps_limit=30)
            self.view.root.withdraw()
            self.view._layout(SimpleNamespace(width=980, height=720))
            args = SimpleNamespace(model="mediapipe", device="synthetic-native-test",
                                   face_details=True, engine="cpp", fps_limit=30)
            self.worker = motion_tracker.MotionWorker(args, self.store, self.stop, self.settings)
            self.worker.start()

        def tearDown(self):
            self.stop.set()
            self.worker.join(timeout=3)
            self.server.close()
            self.store.close()
            self.view.close()
            self.view._on_close = lambda: None
            self.view._on_fps_limit = None
            self.view = None
            for item in reversed(self.patches):
                item.stop()
            self.temporary.cleanup()
            self.assertFalse(self.worker.is_alive())
            self.assertTrue(self.camera.closed)
            self.assertTrue(self.model.closed)
            # Tcl objects from one test must be finalized on this Tk thread,
            # never during an unrelated worker allocation in the next test.
            gc.collect()

        def request_close(self):
            self.stop.set()
            self.view.root.after_idle(self.view.close)

        def get_json(self, path):
            with urlopen(self.url + path, timeout=2) as response:
                self.assertEqual(response.status, 200)
                return json.load(response)

        def wait_state(self, predicate):
            deadline = time.perf_counter() + 3
            while time.perf_counter() < deadline:
                self.view.root.update()
                state = self.store.snapshot()
                if predicate(state):
                    return state
                time.sleep(.002)
            self.fail(f"Native worker state never arrived: {self.store.snapshot()}")

        def send_frame(self, points=None, face=None, *, age=0):
            previous = self.store.snapshot()["sequence"]
            self.camera.queue.put(SimpleNamespace(width=640, height=480, points=points,
                                                  face=face, age=age))
            self.wait_state(lambda state: state["sequence"] > previous)
            return self.render_latest_http_and_check_sse()

        def render_latest_http_and_check_sse(self):
            state = self.get_json("/api/v1/state")
            self.assertEqual(state, self.store.snapshot())
            request = Request(self.url + "/api/v1/events",
                              headers={"Last-Event-ID": str(state["sequence"] - 1)})
            with urlopen(request, timeout=2) as response:
                self.assertIn("text/event-stream", response.headers["Content-Type"])
                self.assertEqual(response.readline().decode().strip(), "event: pose")
                self.assertEqual(response.readline().decode().strip(), f"id: {state['sequence']}")
                payload = response.readline().decode()
                self.assertTrue(payload.startswith("data: "))
                self.assertEqual(json.loads(payload[6:]), state)
            self.view.update(state)
            self.assertNotIn("image", {self.view.canvas.type(item)
                                       for item in self.view.canvas.find_all()})
            self.assertFalse({"image", "raw_image", "frame", "pixels", "jpeg", "png"} & state.keys())
            return state

        def text(self, key):
            return self.view.canvas.itemcget(self.view._items[key], "text")

        def assert_mesh_matches_api(self, state):
            view = self.view
            metadata = self.get_json("/api/v1/metadata")
            self.assertTrue(metadata["coordinates"]["gui_mirrored"])
            self.assertEqual(len(metadata["face_edges"]), 157)
            self.assertEqual(len(state["face_landmarks"]), 478)
            for index, item in enumerate(view._face_items):
                x, y, unused_z = state["face_landmarks"][index]
                expected = [view._sx + (1 - x) * view._sw - 1,
                            view._sy + y * view._sh - 1,
                            view._sx + (1 - x) * view._sw + 1,
                            view._sy + y * view._sh + 1]
                self.assertEqual(view.canvas.itemcget(item, "state"), "normal")
                np.testing.assert_allclose(view.canvas.coords(item), expected)
            for (first, second), item in zip(metadata["face_edges"], view._face_edge_items):
                expected = []
                for index in (first, second):
                    x, y, unused_z = state["face_landmarks"][index]
                    expected += [view._sx + (1 - x) * view._sw, view._sy + y * view._sh]
                self.assertEqual(view.canvas.itemcget(item, "state"), "normal")
                np.testing.assert_allclose(view.canvas.coords(item), expected)

        def assert_absence_cleared(self, state):
            self.assertFalse(state["tracked"])
            self.assertFalse(state["body_tracked"])
            self.assertFalse(state["face_tracked"])
            self.assertEqual(state["face_landmarks"], [])
            self.assertIsNone(state["face_motion"])
            self.assertIsNone(state["motion_speed"])
            self.assertEqual(state["left_arm"], "UNKNOWN")
            self.assertFalse(np.asarray(state["points"]).any())
            for item in (self.view._face_items + self.view._face_edge_items +
                         self.view._joint_items + self.view._edge_items):
                self.assertEqual(self.view.canvas.itemcget(item, "state"), "hidden")
            self.assertEqual(self.text("speed"), "--")
            self.assertEqual(self.text("status"), "SEARCHING")

        def test_face_only_478_points_velocity_http_sse_and_mirror(self):
            initial = self.send_frame(face=mesh_detection())
            self.assertEqual(initial["events"], ["FACE_ACQUIRED"])
            state = self.send_frame(face=mesh_detection((.03, .04)))
            self.assertTrue(state["tracked"])
            self.assertFalse(state["body_tracked"])
            self.assertEqual(state["tracking_level"], "FACE")
            self.assertEqual(state["face_center"]["confidence"], None)
            dt = self.model.timestamps[-1] - self.model.timestamps[-2]
            self.assertAlmostEqual(state["face_motion"]["dx"], .03 / dt)
            self.assertAlmostEqual(state["face_motion"]["dy"], .04 / dt)
            self.assertAlmostEqual(state["face_motion"]["speed"], .05 / dt)
            self.assertEqual(self.text("status"), "FACE TRACKED")
            self.assertEqual(self.text("speed_label"), "FACE MOTION")
            self.assertEqual(self.text("speed"), f"{state['face_motion']['speed']:.3f} image/s")
            self.assertEqual(self.text("angles"), "L  --     R  --")
            self.assertEqual(self.text("confidence"), "478 FACE POINTS")
            self.assertEqual(state["source_resolution"], [640, 480])
            self.assertEqual(state["camera_fps"], 20.)
            self.assert_mesh_matches_api(state)

        def test_body_actions_angles_events_and_face_mesh_match_api(self):
            for unused in range(8):
                state = self.send_frame(points=body_points(), face=mesh_detection())
            self.assertTrue(state["body_tracked"])
            self.assertTrue(state["face_tracked"])
            self.assertEqual(state["left_arm"], "UP")
            self.assertEqual(state["right_arm"], "DOWN")
            self.assertEqual(state["lean"], "LEFT")
            self.assertEqual(self.text("status"), "BODY + FACE")
            self.assertEqual(self.text("left_arm"), state["left_arm"])
            self.assertEqual(self.text("right_arm"), state["right_arm"])
            self.assertEqual(self.text("lean"), "RIGHT")
            self.assertEqual(self.text("angles"),
                             f"L  {state['angles']['left_elbow']:.0f}°     R  {state['angles']['right_elbow']:.0f}°")
            self.assertEqual(self.text("speed"), f"{state['motion_speed']:.2f} widths/s")
            self.assertEqual(self.text("fps"), f"{state['fps']:.1f} fps")
            self.assertEqual(self.text("inference"), f"{state['inference_ms']:.1f} ms")
            self.assertEqual(self.text("age"), f"{state['frame_age_ms']:.0f} ms")
            for event in state["recent_events"]:
                visible = {"LEAN_LEFT": "LEAN_RIGHT", "LEAN_RIGHT": "LEAN_LEFT"}.get(
                    event["event"], event["event"])
                self.assertIn(visible, self.text("events"))
            x, y, confidence = state["points"][5]
            self.assertGreater(confidence, .3)
            self.assertAlmostEqual(self.view.canvas.coords(self.view._joint_items[5])[0],
                                   self.view._sx + (1 - x) * self.view._sw - 5)
            self.assertAlmostEqual(self.view.canvas.coords(self.view._joint_items[5])[1],
                                   self.view._sy + y * self.view._sh - 5)
            self.assert_mesh_matches_api(state)

        def test_absence_and_missing_camera_frame_clear_published_geometry(self):
            self.send_frame(points=body_points(), face=mesh_detection())
            state = self.send_frame()
            self.assert_absence_cleared(state)
            self.assertIn("FACE_LOST", state["events"])
            self.send_frame(face=mesh_detection())
            previous = self.store.snapshot()["sequence"]
            self.camera.queue.put(None)
            self.wait_state(lambda current: current["sequence"] > previous)
            state = self.render_latest_http_and_check_sse()
            self.assertEqual(state["camera_status"], "waiting")
            self.assert_absence_cleared(state)

        def test_fps_slider_worker_pacing_and_window_close_release_resources(self):
            self.send_frame(face=mesh_detection())
            scale = self.view._fps_scale
            self.assertEqual(float(scale.cget("from")), 1.)
            self.assertEqual(float(scale.cget("to")), 30.)
            scale.set(6)
            self.view.root.tk.call(scale.cget("command"), str(scale.get()))
            self.assertEqual(self.settings.get_fps_limit(), 6.)
            for unused in range(3):
                state = self.send_frame(face=mesh_detection())
                self.assertEqual(state["fps_limit"], 6.)
            self.assertGreaterEqual(self.model.starts[-1] - self.model.starts[-2], 1 / 6 * .85)
            self.assertEqual(self.text("fps_limit"), "FPS LIMIT  6")
            self.assertEqual(self.text("fps"), f"{state['fps']:.1f} fps")
            scale.set(1)
            self.view.root.tk.call(scale.cget("command"), str(scale.get()))
            self.send_frame(face=mesh_detection())
            self.assertTrue(self.stop.long_wait.wait(timeout=2), "Worker never entered one-FPS wait")
            self.view._request_close()
            self.view.root.update_idletasks()
            self.assertTrue(self.stop.is_set())
            self.worker.join(timeout=.75)
            self.assertFalse(self.worker.is_alive(), "Closing must interrupt FPS pacing")
            self.assertTrue(self.camera.closed)
            self.assertTrue(self.model.closed)
            report = json.loads((self.worker.run_dir / "summary.json").read_text())
            self.assertEqual(report["engine"], "cpp")
            self.assertTrue(report["final"])
            self.assertIsNone(report["error"])
            self.server.close()
            self.store.close()
            self.assertFalse(self.server._thread.is_alive())
            with self.assertRaises(OSError):
                socket.create_connection(self.server.address, timeout=.2)

        def test_stale_and_failed_camera_clear_native_state_and_cleanup(self):
            self.send_frame(points=body_points(), face=mesh_detection())
            state = self.send_frame(points=body_points(), face=mesh_detection(), age=2.)
            self.assertEqual(state["status"], "STALE_FRAME")
            self.assertEqual(state["camera_status"], "lagging")
            self.assert_absence_cleared(state)
            self.send_frame(face=mesh_detection())
            self.camera.queue.put(RuntimeError("synthetic native camera disconnected"))
            self.wait_state(lambda current: current["status"] == "ERROR")
            state = self.render_latest_http_and_check_sse()
            self.assertIn("synthetic native camera disconnected", state["error"])
            self.assert_absence_cleared(state)
            self.assertEqual(self.text("status_detail"), "ERROR")
            self.worker.join(timeout=1)
            report = json.loads((self.worker.run_dir / "summary.json").read_text())
            self.assertTrue(report["final"])
            self.assertIn("synthetic native camera disconnected", report["error"])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeTkChecks)
    automatic_gc = gc.isenabled()
    gc.disable()
    try:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        gc.collect()
        if automatic_gc:
            gc.enable()


if __name__ == "__main__":
    if "--tk-child" in sys.argv:
        raise SystemExit(_run_tk_checks())
    unittest.main()
