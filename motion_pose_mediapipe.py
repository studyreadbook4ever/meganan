"""CPU pose inference returning COCO-17 landmarks, without saving camera frames.

The SDK processes one RGB frame in memory in VIDEO mode. The public output is
an array of normalized (x, y, confidence) landmarks; no source pixels escape
``infer``. See ``pose_probe_assets/pose_landmarker_lite.source.json`` for the
official model source and downloaded artifact checksum.

Official API: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
"""

from pathlib import Path
import time

import cv2
import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "pose_probe_assets"
    / "pose_landmarker_lite.task"
)

# The rest of the application uses the COCO-17 joint ordering used by MoveNet.
COCO_FROM_MEDIAPIPE = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


class MediaPipePose:
    """One-person, CPU-only MediaPipe Lite adapter.

    ``num_threads`` is accepted for interchangeability with other adapters;
    MediaPipe's public Tasks API manages its own CPU thread count.
    ``timestamp`` passed to ``infer`` is monotonic time in **seconds**, matching
    ``time.monotonic()``. Calls must be sequential; this object is not thread safe.
    """

    name = "MediaPipe Pose Landmarker Lite (CPU)"

    def __init__(self, num_threads=2, model_path=DEFAULT_MODEL_PATH):
        del num_threads  # There is no public thread-count option in Tasks API.
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        # A delayed import leaves other inference adapters usable independently.
        import mediapipe as mp

        self._mp = mp
        self._last_timestamp_ms = -1
        self._closed = False
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def infer(self, frame, timestamp=None):
        """Return shape (17, 3), ordered x/y/confidence; absent poses are zeros.

        Input is a uint8 BGR frame. Confidence is the smaller of landmark
        presence and visibility. Coordinates are normalized to the source
        width/height. Points outside the image have zero confidence and should
        not be drawn or used to classify an action.
        """
        if self._closed:
            raise RuntimeError("Pose adapter is closed")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3)")
        if frame.dtype != np.uint8 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("Expected a non-empty uint8 BGR image")

        seconds = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("timestamp must be finite, nonnegative monotonic seconds")
        # The SDK requires strictly increasing millisecond timestamps, including
        # successive frames received within the same millisecond.
        timestamp_ms = max(int(seconds * 1000), self._last_timestamp_ms + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        output = np.zeros((17, 3), dtype=np.float32)
        if not result.pose_landmarks:
            return output

        landmarks = result.pose_landmarks[0]
        for index, source_index in enumerate(COCO_FROM_MEDIAPIPE):
            point = landmarks[source_index]
            x, y = float(point.x), float(point.y)
            confidence = min(float(point.visibility or 0), float(point.presence or 0))
            if not np.isfinite((x, y, confidence)).all():
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1):
                confidence = 0.0
            output[index] = (x, y, np.clip(confidence, 0.0, 1.0))
        return output

    def close(self):
        """Release the native graph. Safe to call repeatedly."""
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
