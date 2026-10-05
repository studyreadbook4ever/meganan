"""Expose visible face movement without changing body inference or decisions.

This presentation adapter accepts either the existing COCO face landmarks or
a dedicated face mesh. It does not identify a person or infer emotion.
Face velocity uses normalized image coordinates per second; the existing body
``motion_speed`` retains its separate shoulder-widths-per-second meaning.
"""

from __future__ import annotations

import copy
import math

import numpy as np


class MotionPresentation:
    """Add face visibility and velocity to a detached body-state dictionary.

    ``tracked`` means body or face movement is available. ``body_tracked``
    preserves the original body gate so body gestures are never inferred from
    a face alone. ``face_center`` is a stable anatomical anchor: nose first,
    then the eye midpoint, then the ear midpoint. A dedicated face mesh uses
    its stable nose landmark 1. Switching anchors resets velocity. ``dx`` and
    ``dy`` are signed image-coordinate velocities per second; ``speed`` is
    their Euclidean length. Mesh z coordinates are relative, not metric depth.
    """

    def __init__(self, confidence_threshold: float = 0.3):
        if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be finite and in (0, 1]")
        self.threshold = confidence_threshold
        self._previous_time: float | None = None
        self._previous_center: np.ndarray | None = None
        self._previous_anchor: str | None = None
        self._previous_face_tracked = False

    def update(self, body_state: dict, timestamp: float, face: dict | None = None) -> dict:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite monotonic seconds")
        if self._previous_time is not None and timestamp <= self._previous_time:
            raise ValueError("timestamps must increase strictly")
        result = copy.deepcopy(body_state)
        points = np.asarray(body_state.get("points", np.zeros((17, 3))), dtype=float)
        if points.shape != (17, 3):
            raise ValueError("points must have shape (17, 3): x, y, confidence")
        pose_face = points[:5]
        valid = (np.isfinite(pose_face).all(axis=1)
                 & (pose_face[:, :2] >= 0).all(axis=1)
                 & (pose_face[:, :2] <= 1).all(axis=1)
                 & (pose_face[:, 2] >= self.threshold)
                 & (pose_face[:, 2] <= 1))
        eye_pair = bool(valid[1] and valid[2])
        ear_pair = bool(valid[3] and valid[4])
        # A spatially collapsed pair does not establish visible face geometry,
        # even if all confidence values are high.
        eye_span = float(np.linalg.norm(pose_face[1, :2] - pose_face[2, :2])) if eye_pair else 0.
        ear_span = float(np.linalg.norm(pose_face[3, :2] - pose_face[4, :2])) if ear_pair else 0.
        pose_face_tracked = bool(valid.sum() >= 3 and max(eye_span, ear_span) >= .012)
        mesh = None
        if isinstance(face, dict) and face.get("detected", False):
            try:
                candidate = np.asarray(face.get("landmarks"), dtype=float)
            except (ValueError, TypeError):
                candidate = np.zeros((0, 3))
            # Preserve mesh landmark indices: dropping individual invalid rows
            # would connect the wrong facial features. Reject malformed meshes.
            if (candidate.ndim == 2 and candidate.shape[1] == 3
                    and candidate.shape[0] in (468, 478) and np.isfinite(candidate).all()
                    and (candidate[1, :2] >= 0).all() and (candidate[1, :2] <= 1).all()):
                mesh = candidate
        face_tracked = mesh is not None or pose_face_tracked
        face_mode = "mesh" if mesh is not None else "pose" if pose_face_tracked else "none"
        body_tracked = bool(body_state.get("tracked", False))
        result["body_tracked"] = body_tracked
        result["face_tracked"] = face_tracked
        result["face_mode"] = face_mode
        result["tracked"] = body_tracked or face_tracked
        result["person_present"] = bool(body_state.get("person_present", body_tracked)
                                        or face_tracked)
        if face_tracked and not body_tracked:
            result["tracking_level"] = "FACE"
            result["status"] = "FACE_TRACKING"
        result["face_center"] = None
        result["face_motion"] = None
        result["face_landmarks"] = [] if mesh is None else mesh.tolist()
        result["face_bbox"] = None
        if mesh is not None:
            bbox = np.asarray(face.get("bbox"), dtype=float)
            if (bbox.shape != (4,) or not np.isfinite(bbox).all()
                    or not (bbox[0] <= bbox[2] and bbox[1] <= bbox[3])):
                bbox = np.r_[mesh[:, :2].min(axis=0), mesh[:, :2].max(axis=0)]
            result["face_bbox"] = np.clip(bbox, 0., 1.).tolist()
        events = list(result.get("events", []))
        if face_tracked != self._previous_face_tracked:
            events.append("FACE_ACQUIRED" if face_tracked else "FACE_LOST")
        result["events"] = events

        anchor = None
        center = None
        if face_tracked:
            confidence = None
            if mesh is not None:
                anchor = "mesh_nose"
                center = mesh[1, :2].copy()
            elif valid[0]:
                anchor, indices = "nose", [0]
            elif eye_pair:
                anchor, indices = "eyes", [1, 2]
            else:
                anchor, indices = "ears", [3, 4]
            if mesh is None:
                center = np.mean(pose_face[indices, :2], axis=0)
                confidence = float(np.min(pose_face[indices, 2]))
            result["face_center"] = {
                "x": float(center[0]), "y": float(center[1]),
                "confidence": confidence,
            }
            dt = None if self._previous_time is None else timestamp - self._previous_time
            if (self._previous_face_tracked and anchor == self._previous_anchor
                    and self._previous_center is not None and dt is not None
                    and .001 <= dt <= .25):
                velocity = (center - self._previous_center) / dt
                result["face_motion"] = {
                    "dx": float(velocity[0]), "dy": float(velocity[1]),
                    "speed": float(np.linalg.norm(velocity)),
                    "unit": "image_fraction_per_second",
                }
        self._previous_time = timestamp
        self._previous_face_tracked = face_tracked
        self._previous_center = center.copy() if center is not None else None
        self._previous_anchor = anchor
        return result
