"""CPU facial geometry tracking. Input frames are used in memory only.

API: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
The 478 returned x/y/z coordinates are geometry, not identity, emotions, or
per-point confidence. Blendshape and transformation-matrix outputs are disabled.
Source/checksum: pose_probe_assets/face_landmarker.source.json.
"""

from pathlib import Path
import time

import cv2
import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "pose_probe_assets" / "face_landmarker.task"

# Static (start, end) pairs from MediaPipe 1.0.1 FaceLandmarksConnections:
# official contour connections plus both irises and the nose. Generated from
# the installed SDK, whose connection definitions are Apache-2.0 licensed.
# Each two-index item is one line segment; consumers must not join unrelated
# segments. Static data avoids importing/initializing MediaPipe in GUI imports.
from motion_topology import FACE_CONTOURS


class FaceTracker:
    """Sequential one-face tracking; timestamps use monotonic seconds."""

    name = "MediaPipe Face Landmarker (CPU)"

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face model not found: {model_path}")
        import mediapipe as mp

        self._mp = mp
        self._last_timestamp_ms = -1
        self._closed = False
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _absent():
        return {
            "detected": False,
            "landmarks": np.empty((0, 3), dtype=np.float32),
            "contours": FACE_CONTOURS,
            "bbox": None,
        }

    def infer(self, frame, timestampSeconds=None):
        """Return detected, (N,3) x/y/z landmarks, contour edges, and bbox.

        x/y are normalized image coordinates. z is relative model depth, not
        distance in meters. Bbox is [xmin,ymin,xmax,ymax], clipped to the image.
        Points may fall beyond image edges and must be clipped when rendering.
        No per-landmark confidence is returned by this model.
        """
        if self._closed:
            raise RuntimeError("Face tracker is closed")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3)")
        if frame.dtype != np.uint8 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("Expected a non-empty uint8 BGR image")
        seconds = time.monotonic() if timestampSeconds is None else float(timestampSeconds)
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("timestamp must be finite, nonnegative monotonic seconds")
        timestamp_ms = max(int(seconds * 1000), self._last_timestamp_ms + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        if not result.face_landmarks:
            return self._absent()

        points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        if points.shape != (478, 3) or not np.isfinite(points).all():
            return self._absent()
        minimum = np.clip(points[:, :2].min(axis=0), 0.0, 1.0)
        maximum = np.clip(points[:, :2].max(axis=0), 0.0, 1.0)
        if np.any(maximum <= minimum):
            return self._absent()
        return {
            "detected": True,
            "landmarks": points,
            "contours": FACE_CONTOURS,
            "bbox": [float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])],
        }

    def close(self):
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
