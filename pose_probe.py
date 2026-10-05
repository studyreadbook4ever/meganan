"""Short CPU-only person/pose feasibility probe. Saves aggregate metrics only.

Run with .venv/bin/python pose_probe.py. Models and their licensed wrappers are
from OpenCV Zoo; see pose_probe_assets/sources.json for pinned source hashes.
"""

import json
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np
from motion_privacy import disable_process_dumps

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "pose_probe_assets"
sys.path.insert(0, str(ASSETS / "person_detection_mediapipe"))
sys.path.insert(0, str(ASSETS / "pose_estimation_mediapipe"))
from mp_persondet import MPPersonDet
from mp_pose import MPPose


class OrderedPersonDet(MPPersonDet):
    """OpenCV 5 may enumerate ONNX output names in a different order."""

    def _postprocess(self, blobs, original_shape, pad_bias):
        boxes = next(b for b in blobs if b.shape == (1, 2254, 12))
        scores = next(b for b in blobs if b.shape == (1, 2254, 1))
        return super()._postprocess([boxes, scores], original_shape, pad_bias)


class OrderedPose(MPPose):
    def _postprocess(self, blobs, *args):
        shapes = [(1, 195), (1, 1), (1, 256, 256, 1),
                  (1, 64, 64, 39), (1, 117)]
        ordered = [next(b for b in blobs if b.shape == shape) for shape in shapes]
        return super()._postprocess(ordered, *args)


def summarize(values):
    if not values:
        return None
    return {
        "median": round(float(np.median(values)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "samples": len(values),
    }


def main():
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(False)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU
    detector = OrderedPersonDet(str(ASSETS / "person_detection_mediapipe" /
                              "person_detection_mediapipe_2023mar.onnx"),
                           backendId=backend, targetId=target)
    pose_model = OrderedPose(str(ASSETS / "pose_estimation_mediapipe" /
                           "pose_estimation_mediapipe_2023mar.onnx"),
                        backendId=backend, targetId=target)
    print("Models loaded; probing local camera without saving frames.", flush=True)
    disable_process_dumps()
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    det_ms, pose_ms, total_ms = [], [], []
    confidences, reliable_counts = [], []
    detected_frames = valid_pose_frames = 0
    try:
        for _ in range(10):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera warmup failed")
        measured_start = None
        for index in range(35):
            if index == 5:
                measured_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            start = time.perf_counter()
            people = detector.infer(frame)
            after_detection = time.perf_counter()
            pose = None
            if len(people):
                person = people[np.argmax(people[:, -1])].copy()
                pose = pose_model.infer(frame, person)
            finished = time.perf_counter()
            if index >= 5:
                det_ms.append((after_detection - start) * 1000)
                total_ms.append((finished - start) * 1000)
                if len(people):
                    detected_frames += 1
                    pose_ms.append((finished - after_detection) * 1000)
                if pose is not None:
                    valid_pose_frames += 1
                    landmarks = pose[1][:33]
                    confidences.append(float(pose[-1]))
                    # Ignore low-confidence and out-of-frame landmarks.
                    reliable = ((landmarks[:, 3] >= .5) &
                                (landmarks[:, 4] >= .5) &
                                (landmarks[:, 0] >= 0) &
                                (landmarks[:, 0] < frame.shape[1]) &
                                (landmarks[:, 1] >= 0) &
                                (landmarks[:, 1] < frame.shape[0]))
                    reliable_counts.append(int(reliable.sum()))
        elapsed = time.perf_counter() - measured_start
    finally:
        cap.release()

    result = {
        "opencv": cv2.__version__,
        "threads": cv2.getNumThreads(),
        "backend": "OpenCV DNN CPU, OpenCL disabled",
        "camera_resolution": list(frame.shape[1::-1]),
        "measured_frames": 30,
        "warmup_pipeline_frames": 5,
        "frames_with_person_detection": detected_frames,
        "frames_with_pose_above_threshold": valid_pose_frames,
        "person_detection_ms": summarize(det_ms),
        "pose_estimation_ms_when_person_detected": summarize(pose_ms),
        "combined_processing_ms": summarize(total_ms),
        "capture_plus_processing_fps_no_display": round(30 / elapsed, 2),
        "pose_confidence": summarize(confidences),
        "reliable_in_frame_landmark_count": summarize(reliable_counts),
        "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "limits": [
            "Single scene and short run, not a pose accuracy or gesture recognition evaluation.",
            "Highest-scoring person only; detector runs on every frame without temporal tracking.",
            "No GUI or display; latency and sustained performance not measured.",
            "Source camera frames remain in process memory only and are never saved/uploaded.",
            "Only aggregate metrics are saved, no images, video, or landmark trajectories.",
        ],
    }
    text = json.dumps(result, indent=2)
    (ROOT / "pose_probe_results.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
