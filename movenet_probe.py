"""Measure MoveNet Lightning on the local webcam; save aggregate metrics only."""

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from motion_privacy import disable_process_dumps


def main():
    root = Path(__file__).resolve().parent
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    interpreter = Interpreter(
        model_path=str(root / "pose_probe_assets" / "movenet_lightning_int8_v4.tflite"),
        num_threads=2,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if tuple(inp["shape"]) != (1, 192, 192, 3) or inp["dtype"] != np.uint8:
        raise RuntimeError("Unexpected MoveNet input format")
    disable_process_dumps()
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    timings, counts, shoulder_scores = [], [], []
    measured_start = None
    try:
        for index in range(70):
            if index == 10:
                measured_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            started = time.perf_counter()
            height, width = frame.shape[:2]
            scale = min(192 / width, 192 / height)
            rw, rh = round(width * scale), round(height * scale)
            rgb = cv2.cvtColor(cv2.resize(frame, (rw, rh)), cv2.COLOR_BGR2RGB)
            left, top = (192 - rw) // 2, (192 - rh) // 2
            padded = cv2.copyMakeBorder(rgb, top, 192 - rh - top,
                                       left, 192 - rw - left,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            interpreter.set_tensor(inp["index"], padded[np.newaxis])
            interpreter.invoke()
            points = interpreter.get_tensor(out["index"])[0, 0]
            # Convert normalized model coordinates to source-image pixels.
            xy = (points[:, [1, 0]] * 192 - [left, top]) / scale
            reliable = ((points[:, 2] >= .3) & (xy[:, 0] >= 0) &
                        (xy[:, 0] < width) & (xy[:, 1] >= 0) &
                        (xy[:, 1] < height))
            finished = time.perf_counter()
            if index >= 10:
                timings.append((finished - started) * 1000)
                counts.append(int(reliable.sum()))
                shoulder_scores.append(bool(reliable[5] and reliable[6]))
        elapsed = time.perf_counter() - measured_start
    finally:
        cap.release()
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "MoveNet SinglePose Lightning int8 v4",
        "runtime": "LiteRT Interpreter, CPU, num_threads=2",
        "input_shape": list(map(int, inp["shape"])),
        "input_dtype": str(inp["dtype"]),
        "camera_resolution": [width, height],
        "measured_frames": len(timings),
        "processing_median_ms": round(float(np.median(timings)), 3),
        "processing_p95_ms": round(float(np.percentile(timings, 95)), 3),
        "capture_plus_processing_fps_no_display": round(len(timings) / elapsed, 2),
        "median_reliable_in_frame_points_of_17": float(np.median(counts)),
        "frames_with_both_shoulders_reliable": sum(shoulder_scores),
        "point_confidence_threshold": .3,
        "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "limits": [
            "Short single-scene feasibility measurement; not an accuracy evaluation.",
            "No display, temporal filter, crop tracker, or gesture classification.",
            "Model always predicts points; confidence counts do not prove correctness.",
            "Camera frames processed in memory only, never saved or uploaded.",
            "Only aggregate metrics saved; no landmark trajectories stored.",
        ],
    }
    serialized = json.dumps(result, indent=2)
    (root / "movenet_probe_results.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
