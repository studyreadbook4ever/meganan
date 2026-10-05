"""meganan: local CPU motion service with a skeleton-only desktop view.

Only numeric pose/motion data leaves the inference worker. Camera images are
never passed to the API, GUI, or telemetry writer.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import threading
import time
import traceback

import cv2
import numpy as np

from motion_api import MotionAPIServer, StateStore
from motion_features import MotionAnalyzer
from motion_presentation import MotionPresentation

ROOT = Path(__file__).resolve().parent


class RuntimeSettings:
    def __init__(self, fps_limit=20):
        self.lock = threading.Lock()
        self.set_fps_limit(fps_limit)

    def set_fps_limit(self, value):
        value = float(value)
        if not math.isfinite(value) or not 1 <= value <= 30:
            raise ValueError("FPS limit must be between 1 and 30")
        with self.lock:
            self._fps_limit = value

    def get_fps_limit(self):
        with self.lock:
            return self._fps_limit


class LatestCamera:
    """Continuously drain the camera; inference consumes the newest frame."""

    def __init__(self, device="/dev/video0"):
        self.device = device
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.error = None
        self.timestamps = deque(maxlen=90)
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera")

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        cap = None
        try:
            from motion_privacy import disable_process_dumps
            disable_process_dumps()
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {self.device}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera stopped returning frames")
                now = time.perf_counter()
                with self.condition:
                    self.frame, self.timestamp = frame, now
                    self.sequence += 1
                    self.timestamps.append(now)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
        finally:
            if cap is not None:
                cap.release()

    def next(self, after, timeout=.3):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > after or self.error or self.stop_event.is_set(),
                timeout,
            )
            if self.error:
                raise RuntimeError(self.error)
            if self.sequence <= after:
                return None
            stamps = self.timestamps
            fps = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                   if len(stamps) >= 2 and stamps[-1] > stamps[0] else 0.)
            return self.sequence, self.timestamp, self.frame, fps

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2)


class MoveNetPose:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter
        self.interpreter = Interpreter(
            model_path=str(ROOT / "pose_probe_assets/movenet_lightning_int8_v4.tflite"),
            num_threads=2,
        )
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        if tuple(self.inp["shape"]) != (1, 192, 192, 3) or self.inp["dtype"] != np.uint8:
            raise RuntimeError("Unexpected MoveNet input format")

    def infer(self, frame, timestamp=None):
        height, width = frame.shape[:2]
        scale = min(192 / width, 192 / height)
        rw, rh = round(width * scale), round(height * scale)
        rgb = cv2.cvtColor(cv2.resize(frame, (rw, rh)), cv2.COLOR_BGR2RGB)
        left, top = (192 - rw) // 2, (192 - rh) // 2
        padded = cv2.copyMakeBorder(rgb, top, 192 - rh - top, left,
                                   192 - rw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.interpreter.set_tensor(self.inp["index"], padded[np.newaxis])
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.out["index"])[0, 0]
        xy = (output[:, [1, 0]] * 192 - [left, top]) / scale / [width, height]
        return np.column_stack((xy, output[:, 2]))

    def close(self):
        pass


class MotionWorker(threading.Thread):
    def __init__(self, args, store, stop, settings=None):
        super().__init__(daemon=True, name="motion-inference")
        self.args, self.store, self.stop = args, store, stop
        self.settings = settings or RuntimeSettings(getattr(args, "fps_limit", 20))
        self.started_at = time.perf_counter()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = ROOT / "runs" / stamp
        self.run_dir.mkdir(parents=True)
        self.frames = self.tracked_frames = 0
        self.body_tracked_frames = self.face_tracked_frames = 0
        self.event_counts = Counter()
        self.inference_times = deque(maxlen=3000)
        self.tracked_times = deque(maxlen=3000)
        self.latest_state = None
        self.error = None

    def summary(self, final=False):
        elapsed = time.perf_counter() - self.started_at
        timings = list(self.inference_times)
        report = {
            "model": self.args.model,
            "engine": getattr(self.args, "engine", "python"),
            "elapsed_s": round(elapsed, 2),
            "processed_frames": self.frames,
            "tracked_frames": self.tracked_frames,
            "body_tracked_frames": self.body_tracked_frames,
            "face_tracked_frames": self.face_tracked_frames,
            "fps_limit": self.settings.get_fps_limit(),
            "tracked_fraction": round(self.tracked_frames / max(1, self.frames), 3),
            "events": dict(self.event_counts),
            "inference_median_ms": round(float(np.median(timings)), 2) if timings else None,
            "inference_p95_ms": round(float(np.percentile(timings, 95)), 2) if timings else None,
            "recent_status": self.latest_state.get("status") if self.latest_state else None,
            "recent_fps": self.latest_state.get("fps") if self.latest_state else None,
            "final": final,
            "error": self.error,
            "raw_images_saved": False,
            "notes": "Unlabelled live test; tracking fraction is not accuracy. Timing window <=3000 frames.",
        }
        temporary = self.run_dir / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.run_dir / "summary.json")
        return report

    def publish(self, state):
        # Explicit numeric projection: a camera frame is never part of this object.
        state["points"] = np.round(state["points"], 6).tolist()
        state["face_landmarks"] = np.round(state.get("face_landmarks", []), 6).tolist()
        state["fps_limit"] = self.settings.get_fps_limit()
        state["model"] = self.args.model
        state["timestamp_unix_s"] = time.time()
        state["elapsed_s"] = round(time.perf_counter() - self.started_at, 3)
        self.latest_state = self.store.publish(state)
        self.event_counts.update(state["events"])

    def run(self):
        camera = model = face_model = None
        analyzer = MotionAnalyzer(aspect_ratio=4 / 3)
        presentation = MotionPresentation()
        native = getattr(self.args, "engine", "python") == "cpp"
        previous_sequence = 0
        completions = deque(maxlen=60)
        next_log = 0.
        last_analysis_timestamp = 0.
        previous_start = None
        previous_limit = None
        try:
            cv2.setNumThreads(2)
            cv2.ocl.setUseOpenCL(False)
            if native:
                from motion_native import (NativeCamera, NativeInference,
                                           MotionAnalyzer as NativeAnalyzer,
                                           MotionPresentation as NativePresentation)
                analyzer = NativeAnalyzer(aspect_ratio=4 / 3)
                presentation = NativePresentation()
                model = NativeInference(self.args.model, getattr(self.args, "face_details", False))
                camera = NativeCamera(self.args.device).start()
            elif self.args.model == "mediapipe":
                from motion_pose_mediapipe import MediaPipePose
                model = MediaPipePose()
            else:
                model = MoveNetPose()
            if not native and getattr(self.args, "face_details", False):
                from motion_face import FaceTracker
                face_model = FaceTracker()
            if not native:
                camera = LatestCamera(self.args.device).start()
            with (self.run_dir / "telemetry.jsonl").open("w", encoding="utf-8") as log:
                while not self.stop.is_set():
                    fps_limit = self.settings.get_fps_limit()
                    if fps_limit != previous_limit:
                        completions.clear()
                        previous_limit = fps_limit
                    if previous_start is not None:
                        wait_seconds = previous_start + 1 / fps_limit - time.perf_counter()
                        if wait_seconds > 0 and self.stop.wait(wait_seconds):
                            break
                    sample = camera.next(previous_sequence)
                    now = time.perf_counter()
                    if sample is None:
                        state = analyzer.update(None, now)
                        state = presentation.update(state, now)
                        last_analysis_timestamp = now
                        state.update(camera_status="waiting", fps=0., camera_fps=0.,
                                     inference_ms=None, frame_age_ms=None,
                                     reliable_points=0, source_resolution=[640, 480], error=None)
                        self.publish(state)
                        continue
                    previous_sequence, acquired, frame, camera_fps = sample
                    started = time.perf_counter()
                    previous_start = started
                    if native:
                        points, face = model.infer(frame, acquired)
                        width, height = frame.width, frame.height
                    else:
                        points = model.infer(frame, acquired)
                        face = face_model.infer(frame, acquired) if face_model is not None else None
                        height, width = frame.shape[:2]
                    finished = time.perf_counter()
                    inference_ms = (finished - started) * 1000
                    analyzer.aspect_ratio = width / height
                    frame_age = finished - acquired
                    # A delayed result must never masquerade as fresh movement.
                    stale = frame_age > .5
                    analysis_timestamp = max(acquired, last_analysis_timestamp + 1e-6)
                    if stale:
                        analysis_timestamp = max(finished, analysis_timestamp)
                    state = analyzer.update(None if stale else points, analysis_timestamp)
                    state = presentation.update(state, analysis_timestamp, face=None if stale else face)
                    last_analysis_timestamp = analysis_timestamp
                    if stale:
                        state["status"] = "STALE_FRAME"
                    completions.append(finished)
                    fps = ((len(completions) - 1) / (completions[-1] - completions[0])
                           if len(completions) >= 2 else 0.)
                    state.update(
                        camera_status="lagging" if stale else "running", fps=round(fps, 2),
                        camera_fps=round(camera_fps, 2), inference_ms=round(inference_ms, 2),
                        frame_age_ms=round((finished - acquired) * 1000, 2),
                        reliable_points=int((state["points"][:, 2] >= .3).sum()),
                        source_resolution=[width, height], error=None,
                    )
                    self.frames += 1
                    self.tracked_frames += int(state["tracked"])
                    self.body_tracked_frames += int(state.get("body_tracked", False))
                    self.face_tracked_frames += int(state.get("face_tracked", False))
                    self.inference_times.append(inference_ms)
                    self.publish(state)
                    if finished >= next_log:
                        report = self.summary()
                        # Aggregates and categorical events only, no images or trajectories.
                        log.write(json.dumps(report) + "\n")
                        log.flush()
                        print(json.dumps(report), flush=True)
                        next_log = finished + 10
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            state = analyzer.update(None, time.perf_counter())
            state = presentation.update(state, time.perf_counter())
            state.update(status="ERROR", camera_status="error", error=self.error,
                         fps=0., inference_ms=None, frame_age_ms=None)
            self.publish(state)
        finally:
            for resource in (camera, model, face_model):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as close_error:
                        print(f"Resource close failed: {close_error}", flush=True)
            self.summary(final=True)


def main():
    parser = argparse.ArgumentParser(prog="meganan", description=__doc__)
    parser.add_argument("--model", choices=("movenet", "mediapipe"), default="mediapipe")
    parser.add_argument("--engine", choices=("cpp", "python"), default="cpp",
                        help="C++ processing engine (default), or Python reference for comparisons")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds; 0 runs until closed")
    parser.add_argument("--fps-limit", type=int, choices=range(1, 31), default=20,
                        metavar="1..30", help="Maximum inference updates per second")
    parser.add_argument("--no-face-details", action="store_false", dest="face_details",
                        help="Use the pose model's five face points without the detailed face model")
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("duration must be nonnegative")
    stop = threading.Event()
    settings = RuntimeSettings(args.fps_limit)
    store = StateStore()
    server = MotionAPIServer(store, host=args.host, port=args.port).start()
    print(f"meganan API: http://{server.address[0]}:{server.address[1]}/api/v1/state", flush=True)
    worker = MotionWorker(args, store, stop, settings)
    view = None

    def close(*unused):
        stop.set()
        if view is not None:
            # SIGTERM can interrupt a Tk render callback; destroy only after
            # that callback returns, so no font/canvas calls use a dead Tk.
            try:
                view.root.after_idle(view.close)
            except Exception:
                pass

    signal.signal(signal.SIGINT, close)
    signal.signal(signal.SIGTERM, close)
    try:
        if not args.no_gui:
            from motion_view import MotionView
            view = MotionView(on_close=close, on_fps_limit=settings.set_fps_limit,
                              fps_limit=args.fps_limit,
                              api_address=f"{server.address[0]}:{server.address[1]}")
        worker.start()
        if view is not None:
            last_sequence = -1

            def refresh():
                nonlocal last_sequence
                if stop.is_set():
                    return
                state = store.snapshot()
                if state.get("sequence", 0) != last_sequence:
                    view.update(state)
                    last_sequence = state.get("sequence", 0)
                view.root.after(33, refresh)

            view.root.after(0, refresh)
            if args.duration:
                view.root.after(round(args.duration * 1000), close)
            view.root.mainloop()
        else:
            deadline = time.perf_counter() + args.duration if args.duration else float("inf")
            while not stop.wait(.2):
                if time.perf_counter() >= deadline or not worker.is_alive():
                    break
    finally:
        stop.set()
        if worker.ident is not None:
            worker.join(timeout=6)
        server.close()
        store.close()
        if view is not None:
            view.close()
        print(f"Run metrics: {worker.run_dir / 'summary.json'}", flush=True)
    return 1 if worker.error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
