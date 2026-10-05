"""Real Tk rendering checks using synthetic landmarks and a withdrawn window.

Discovery imports only the standard library. A child process loads the optional
workspace Tcl/Tk runtime before importing Tkinter, and never opens a camera.
"""
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest


class MotionGuiTests(unittest.TestCase):
    def test_real_tk_synthetic_rendering_and_fps_controls(self):
        if not os.environ.get("DISPLAY"):
            self.skipTest("Tk GUI checks require DISPLAY; no camera is required")
        task_root = Path(__file__).resolve().parent
        env = os.environ.copy()
        runtime_lib = task_root / ".runtime" / "usr" / "lib"
        if (runtime_lib / "libtk8.6.so").is_file():
            env["LD_LIBRARY_PATH"] = str(runtime_lib) + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            env["TCL_LIBRARY"] = str(runtime_lib / "tcl8.6")
            env["TK_LIBRARY"] = str(runtime_lib / "tk8.6")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--tk-child"],
            cwd=task_root, env=env, capture_output=True, text=True, timeout=60,
        )
        details = result.stdout + result.stderr
        if result.returncode == 77:
            self.skipTest(details.strip())
        self.assertEqual(result.returncode, 0, details)
        self.assertIn("Ran 6 tests", details)


def _run_tk_checks():
    # These imports must remain inside the subprocess entry point: system
    # Python may need LD_LIBRARY_PATH set before it loads _tkinter.
    try:
        import tkinter as tk
        from motion_view import MotionView
        from motion_face import FACE_CONTOURS
    except (ImportError, OSError) as exc:
        print(f"Tk runtime unavailable: {exc}")
        return 77

    from unittest.mock import patch
    camera_guard = patch("cv2.VideoCapture", side_effect=AssertionError("GUI tests must not open a camera"))
    camera_guard.start()
    callbacks = []
    try:
        view = MotionView(lambda: None, on_fps_limit=callbacks.append, fps_limit=20,
                          api_address="127.0.0.1:54321")
        view.root.withdraw()
    except tk.TclError as exc:
        camera_guard.stop()
        print(f"Tk display unavailable: {exc}")
        return 77

    def face_state():
        landmarks = []
        for index in range(478):
            theta = index * 2 * math.pi / 478
            # z is signed depth, not a confidence threshold.
            landmarks.append([.5 + .15 * math.cos(theta),
                              .5 + .25 * math.sin(theta), -.1])
        return {
            "tracked": True, "body_tracked": False, "face_tracked": True,
            "person_present": True, "tracking_level": "FACE", "status": "FACE_TRACKING",
            "face_landmarks": landmarks, "face_mode": "mesh",
            "face_center": {"x": .5, "y": .5, "confidence": None},
            "face_motion": {"dx": .01, "dy": -.02, "speed": .12345},
            "points": [[0., 0., 0.] for _ in range(17)],
            # Deliberately populated stale body values must not leak into UI.
            "left_arm": "UP", "right_arm": "DOWN", "lean": "LEFT",
            "angles": {"left_elbow": 50., "right_elbow": 140.},
            "motion_speed": 2., "fps": 5.99, "inference_ms": 21.,
            "frame_age_ms": 30., "events": [], "elapsed_s": 2.,
        }

    class SyntheticTkChecks(unittest.TestCase):
        def setUp(self):
            callbacks.clear()
            view._recent_events.clear()
            view._layout(SimpleNamespace(width=980, height=720))
            self.payload = face_state()
            view.update(self.payload)

        def text(self, key):
            return view.canvas.itemcget(view._items[key], "text")

        def test_recent_events_use_the_api_history_without_duplicates(self):
            self.payload.update(timestamp_unix_s=100., elapsed_s=10., events=[],
                                recent_events=[
                                    {"event": "LEFT_ARM_UP", "timestamp_unix_s": 98., "sequence": 40},
                                    {"event": "LEAN_LEFT", "timestamp_unix_s": 99., "sequence": 41},
                                ])
            view.update(self.payload)
            self.assertEqual(list(view._recent_events),
                             ["00:08  LEFT_ARM_UP", "00:09  LEAN_RIGHT"])
            view.update(self.payload)
            self.assertEqual(len(view._recent_events), 2)
            self.assertIn("LEFT_ARM_UP", self.text("events"))

        def test_face_mesh_is_mirrored_geometry_without_image_items(self):
            from motion_topology import BODY_EDGES
            self.assertEqual(MotionView.EDGES, BODY_EDGES)
            canvas = view.canvas
            self.assertEqual(self.text("status"), "FACE TRACKED")
            self.assertEqual(len(view._face_items), 478)
            self.assertEqual(len(FACE_CONTOURS), 157)
            self.assertEqual(len(view._face_edge_items), 157)
            for item in view._face_items + view._face_edge_items:
                self.assertEqual(canvas.itemcget(item, "state"), "normal")
            expected_x = view._sx + (1 - self.payload["face_landmarks"][0][0]) * view._sw
            self.assertAlmostEqual(canvas.coords(view._face_items[0])[0], expected_x - 1)
            self.assertAlmostEqual(view._sw / view._sh, 4 / 3)
            self.assertNotIn("image", {canvas.type(item) for item in canvas.find_all()})
            self.assertEqual(self.text("confidence"), "478 FACE POINTS")

        def test_face_only_gates_body_fields_and_displays_face_speed(self):
            for key in ("left_arm", "right_arm", "lean"):
                self.assertEqual(self.text(key), "UNKNOWN")
            self.assertEqual(self.text("angles"), "L  --     R  --")
            self.assertEqual(self.text("speed_label"), "FACE MOTION")
            self.assertEqual(self.text("speed"), "0.123 image/s")
            self.assertEqual(self.payload["lean"], "LEFT")
            self.assertEqual(self.payload["angles"]["left_elbow"], 50.)

        def test_face_absence_clears_all_landmarks_immediately(self):
            view.update(dict(self.payload, tracked=False, person_present=False,
                             face_tracked=False, face_landmarks=[], tracking_level="NONE"))
            for item in (view._face_items + view._face_edge_items
                         + view._joint_items + view._edge_items):
                self.assertEqual(view.canvas.itemcget(item, "state"), "hidden")
            self.assertEqual(view.canvas.itemcget(view._items["empty"], "state"), "normal")
            self.assertEqual(self.text("speed"), "--")

        def test_slider_bounds_callback_and_actual_fps_are_independent(self):
            scale = view._fps_scale
            self.assertEqual(float(scale.cget("from")), 1.)
            self.assertEqual(float(scale.cget("to")), 30.)
            command = scale.cget("command")
            for target in (1, 6, 20, 30):
                with self.subTest(target=target):
                    scale.set(target)
                    # Invoke the callback registered on the real Tk scale;
                    # the withdrawn window requires no simulated mouse input.
                    view.root.tk.call(command, str(scale.get()))
                    self.assertEqual(callbacks[-1], float(target))
                    self.assertEqual(self.text("fps_limit"), f"FPS LIMIT  {target}")
                    self.assertEqual(self.text("fps"), "6.0 fps")
            for target, expected in ((-10, 1.), (100, 30.)):
                scale.set(target)
                view.root.tk.call(command, str(scale.get()))
                self.assertEqual(callbacks[-1], expected)

        def test_minimum_window_keeps_events_and_footer_inside_bounds(self):
            view._layout(SimpleNamespace(width=900, height=700))
            view.update(dict(self.payload, events=["A LONG EVENT THAT MUST FIT " * 8] * 5))
            canvas = view.canvas
            self.assertEqual(self.text("api"), "API  127.0.0.1:54321")
            self.assertEqual(len(self.text("events").splitlines()), 3)
            self.assertLess(canvas.bbox(view._items["events"])[3], 700 - 104)
            self.assertLess(canvas.bbox(view._items["events"])[2], 900 - 24)
            self.assertLessEqual(canvas.bbox(view._items["api"])[2], 900 - 23)
            self.assertLess(canvas.bbox(view._items["footer"])[2],
                            canvas.bbox(view._items["api"])[0])
            self.assertLessEqual(canvas.bbox(view._items["status"])[2], 900 - 24)

    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTkChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        view.close()
        camera_guard.stop()


if __name__ == "__main__":
    if "--tk-child" in sys.argv:
        raise SystemExit(_run_tk_checks())
    unittest.main()
