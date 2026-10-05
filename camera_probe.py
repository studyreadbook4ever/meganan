"""Measure a local webcam and CPU processing; never save image/video frames.

Run: .venv/bin/python camera_probe.py
The optional --output file contains timing and image statistics only.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from motion_privacy import disable_process_dumps


def benchmark(operation, iterations=60):
    for _ in range(8):
        operation()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": round(float(np.median(durations)), 3),
        "p95_ms": round(float(np.percentile(durations, 95)), 3),
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("camera_probe_results.json"))
    args = parser.parse_args()
    if args.frames < 30:
        parser.error("--frames must be at least 30")

    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    disable_process_dumps()
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.device}")

    def read():
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame read failed")
        return frame

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(15):
            read()

        started = time.perf_counter()
        deltas = []
        previous_gray = None
        for _ in range(args.frames):
            frame = read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                deltas.append(float(cv2.absdiff(gray, previous_gray).mean()))
            previous_gray = gray
        elapsed = time.perf_counter() - started
        h, w = frame.shape[:2]

        # Assumed demonstration parameters, NOT a calibration of this camera.
        camera_matrix = np.array([[500., 0., w / 2], [0., 500., h / 2], [0., 0., 1.]])
        distortion = np.array([-0.2, 0.06, 0., 0., 0.])
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix, distortion, None, camera_matrix, (w, h), cv2.CV_32FC1
        )
        homography = cv2.getPerspectiveTransform(
            np.float32([[w * .1, h * .1], [w * .9, h * .15],
                        [w * .95, h * .9], [w * .05, h * .9]]),
            np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]),
        )

        def edges(image):
            mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(cv2.GaussianBlur(mono, (5, 5), 0), 60, 120)

        def remap(image):
            return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        orb = cv2.ORB_create(nfeatures=500)
        operations = {
            "bilinear_remap_precomputed_map": lambda: remap(frame),
            "grayscale_blur_canny": lambda: edges(frame),
            "perspective_warp": lambda: cv2.warpPerspective(frame, homography, (w, h)),
            "orb_up_to_500_features": lambda: orb.detectAndCompute(gray, None),
        }
        timings = {name: benchmark(op) for name, op in operations.items()}
        keypoints, _ = orb.detectAndCompute(gray, None)

        # Drain buffered frames after the fixed-frame processing benchmarks.
        for _ in range(10):
            read()
        started = time.perf_counter()
        cpu_started = time.process_time()
        for _ in range(args.frames):
            edges(remap(read()))
        pipeline_elapsed = time.perf_counter() - started
        pipeline_cpu_seconds = time.process_time() - cpu_started

        result = {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "opencl_enabled": cv2.ocl.useOpenCL(),
            "device": args.device,
            "resolution": [w, h],
            "driver_reported_fps": cap.get(cv2.CAP_PROP_FPS),
            "capture_plus_grayscale_stats": {
                "frames": args.frames,
                "wall_seconds": round(elapsed, 3),
                "measured_fps": round(args.frames / elapsed, 2),
                "last_frame_gray_mean": round(float(gray.mean()), 2),
                "last_frame_gray_std": round(float(gray.std()), 2),
                "mean_abs_change_between_frames": round(float(np.mean(deltas)), 3),
                "last_frame_orb_keypoints": len(keypoints),
            },
            "fixed_frame_cpu_timings": timings,
            "live_capture_remap_gray_blur_canny_no_display": {
                "frames": args.frames,
                "wall_seconds": round(pipeline_elapsed, 3),
                "measured_fps": round(args.frames / pipeline_elapsed, 2),
                "process_cpu_seconds": round(pipeline_cpu_seconds, 3),
                "mean_cpu_core_equivalents": round(pipeline_cpu_seconds / pipeline_elapsed, 3),
            },
            "limits": [
                "No GUI/display/encoding; end-to-end preview latency not measured.",
                "Processing timings reuse one current camera frame; content affects cost.",
                "Remap uses arbitrary demonstration parameters, not measured camera calibration.",
                "No camera images or videos are written to disk or uploaded.",
                "Short measurement under current lighting and system load; sustained speed may differ.",
            ],
        }
    finally:
        cap.release()

    serialized = json.dumps(result, indent=2)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
