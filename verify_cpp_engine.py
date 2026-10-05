"""Repeatable comparison; saves aggregate numbers only, never camera images.

Run after building: .venv/bin/python verify_cpp_engine.py [--live]
--live needs exclusive access to the camera (close meganan first).
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

import _motion_native as cpp
from motion_features import MotionAnalyzer
from motion_presentation import MotionPresentation
from test_motion_features import pose


def equivalent(a, b, path="state", tolerance=1e-6):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for key in a:
            equivalent(a[key], b[key], path + "." + key, tolerance)
    elif isinstance(a, (np.ndarray, list, tuple)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, f"{path}[{index}]", tolerance)
    elif isinstance(a, (float, np.floating)):
        assert np.isclose(a, b, atol=tolerance, rtol=tolerance), (path, a, b)
    else:
        assert a == b, (path, a, b)


def features_benchmark():
    inputs = []
    for i in range(240):
        body = pose()
        body[:, 0] += .025 * np.sin(i / 12)
        angle = np.arange(478) * 2 * np.pi / 478
        mesh = np.column_stack((.5 + .1 * np.cos(angle) + .03 * np.sin(i / 15),
                                .4 + .2 * np.sin(angle), np.full(478, -.1)))
        inputs.append((body, i / 30, {"detected": True, "landmarks": mesh,
                                      "bbox": [.3, .2, .7, .6]}))
    results, timings = {}, {}
    for name, analyzer_type, presentation_type in (
            ("python", MotionAnalyzer, MotionPresentation),
            ("cpp", cpp.MotionAnalyzer, cpp.MotionPresentation)):
        batches = []
        for repeat in range(7):
            analyzer, presentation = analyzer_type(aspect_ratio=4/3), presentation_type()
            outputs = []
            start = time.perf_counter()
            for body, timestamp, face in inputs:
                outputs.append(presentation.update(analyzer.update(body, timestamp), timestamp, face))
            batches.append((time.perf_counter() - start) * 1e6 / len(inputs))
        results[name] = outputs
        timings[name + "_us_per_update"] = round(statistics.median(batches[1:]), 2)
    for a, b in zip(results["python"], results["cpp"]):
        equivalent(a, b)
    timings["speedup"] = round(timings["python_us_per_update"] / timings["cpp_us_per_update"], 2)
    timings["equal_output_frames"] = len(inputs)
    timings["scope"] = "body and 478-point face movement calculations, excludes model inference/GUI/API"
    return timings


def live_comparison(count):
    import cv2
    from motion_face import FaceTracker
    from motion_pose_mediapipe import MediaPipePose
    from motion_native import package_library, ROOT
    native = pose_model = face_model = camera = None
    try:
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(False)
        native = cpp.NativeLandmarkers(
            package_library("mediapipe", "1.0.1", "tasks/c/libmediapipe.so"),
            str(ROOT / "pose_probe_assets/pose_landmarker_lite.task"),
            str(ROOT / "pose_probe_assets/face_landmarker.task"))
        pose_model, face_model = MediaPipePose(), FaceTracker()
        from motion_privacy import disable_process_dumps
        disable_process_dumps()
        camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if not camera.isOpened():
            raise RuntimeError("Close meganan before the exclusive camera comparison")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        analyzers = [MotionAnalyzer(aspect_ratio=4/3), cpp.MotionAnalyzer(aspect_ratio=4/3)]
        presentations = [MotionPresentation(), cpp.MotionPresentation()]
        timings = {"python": [], "cpp": []}
        face_count = body_count = 0
        max_body_error = max_face_error = 0.
        for i in range(count):
            ok, frame = camera.read()
            assert ok, "Camera stopped"
            timestamp = time.monotonic()
            measured = {}
            for name in (("cpp", "python") if i % 2 else ("python", "cpp")):
                start = time.perf_counter()
                if name == "cpp":
                    measured[name] = native.infer_bgr(frame, timestamp)
                else:
                    measured[name] = (pose_model.infer(frame, timestamp),
                                      face_model.infer(frame, timestamp))
                if i >= 5:
                    timings[name].append((time.perf_counter() - start) * 1000)
            pa, fa = measured["python"]
            pb, fb = measured["cpp"]
            assert fa["detected"] == fb["detected"], "Face detection differs on identical input"
            error = float(np.max(np.abs(pa - pb)))
            max_body_error = max(max_body_error, error)
            assert error <= 1e-5, ("body landmarks", error)
            if fa["detected"]:
                face_count += 1
                error = float(np.max(np.abs(fa["landmarks"] - fb["landmarks"])))
                max_face_error = max(max_face_error, error)
                assert error <= 1e-5, ("face landmarks", error)
            outputs = [p.update(a.update(points, timestamp), timestamp, face)
                       for a, p, (points, face) in zip(analyzers, presentations, (measured["python"], measured["cpp"]))]
            equivalent(*outputs, tolerance=1e-5)
            body_count += int(outputs[0]["body_tracked"])
        return {"same_input_frames": count, "face_detected_frames": face_count,
                "body_tracked_frames": body_count, "max_body_absolute_error": max_body_error,
                "max_face_absolute_error": max_face_error, "state_and_event_parity": True,
                **{name + "_inference_median_ms": round(statistics.median(values), 3)
                   for name, values in timings.items()},
                "raw_images_saved": False,
                "scope": "identical in-memory camera frames through both model adapters and movement engines"}
    finally:
        if camera is not None:
            camera.release()
        for model in (native, pose_model, face_model):
            if model is not None:
                model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.frames < 6:
        parser.error("at least 6 frames are needed for warmup and measurement")
    report = {"features": features_benchmark()}
    if args.live:
        report["live"] = live_comparison(args.frames)
    target = Path("artifacts/cpp_engine_verification.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
