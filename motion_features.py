"""Temporal, image-plane motion features for one COCO-17 pose.

No camera access or persistence occurs here. Input coordinates are normalized
``x, y, confidence``; left/right joints refer to the person's anatomical sides.
Lean LEFT/RIGHT refers to the unmirrored input image. Pass ``aspect_ratio=w/h``
for geometrically correct image-plane angles and shoulder-width-normalized speed.
These are 2D measurements, not anatomical 3D joint angles or physical velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class _Debounced:
    value: str = "UNKNOWN"
    candidate: str = "UNKNOWN"
    since: float | None = None

    def update(self, candidate: str, timestamp: float, delay: float) -> bool:
        """Unknown evidence invalidates output immediately; changes need dwell."""
        if candidate == "UNKNOWN":
            self.value = self.candidate = "UNKNOWN"
            self.since = None
            return False
        if candidate != self.candidate:
            self.candidate, self.since = candidate, timestamp
        if self.since is not None and timestamp - self.since + 1e-9 >= delay:
            if self.value != candidate:
                self.value = candidate
                return True
        return False


class MotionAnalyzer:
    """Analyze one pose stream; call ``update(None, now)`` on missing frames.

    ``tracked`` means this frame has both shoulders, a nondegenerate shoulder
    span, and at least two reliable elbows/wrists/hips. During missing evidence,
    status becomes OCCLUDED immediately and SEARCHING after ``loss_seconds``.
    ``person_present`` includes partial bodies; ``tracking_level`` distinguishes
    FULL, UPPER_BODY, PARTIAL, and NONE. Partial visible landmarks may be drawn,
    but motion decisions remain unavailable until ``tracked`` is true.
    Points and movement decisions never survive missing current evidence.

    Events are edge notifications (TRACKING_ACQUIRED, TRACKING_LOST,
    LEFT_ARM_UP/DOWN, RIGHT_ARM_UP/DOWN, LEAN_LEFT/CENTER/RIGHT), not per-frame
    assertions. They should not be interpreted as verified human activity labels.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        *,
        aspect_ratio: float = 1.0,
        smoothing_seconds: float = 0.07,
        debounce_seconds: float = 0.18,
        loss_seconds: float = 0.5,
    ) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in
               (smoothing_seconds, debounce_seconds, loss_seconds)):
            raise ValueError("time constants must be finite and nonnegative")
        self.threshold = confidence_threshold
        self.aspect_ratio = aspect_ratio
        self.smoothing_seconds = smoothing_seconds
        self.debounce_seconds = debounce_seconds
        self.loss_seconds = loss_seconds
        self._previous_points: np.ndarray | None = None
        self._previous_time: float | None = None
        self._previous_tracked = False
        self._last_seen: float | None = None
        self._tracking_session = False
        self._arms = {"left": _Debounced(), "right": _Debounced()}
        self._lean = _Debounced()
        self._speed: float | None = None

    def _metric(self, xy: np.ndarray) -> np.ndarray:
        return xy * np.array([self.aspect_ratio, 1.0])

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
        first, second = a - b, c - b
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-8:
            return None
        cosine = float(np.clip(np.dot(first, second) / denominator, -1, 1))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _arm_candidate(points: np.ndarray, valid: np.ndarray,
                       shoulder: int, elbow: int, wrist: int,
                       width: float, previous: str) -> str:
        if not valid[[shoulder, elbow, wrist]].all():
            return "UNKNOWN"
        # Positive means wrist above shoulder. Hysteresis keeps small jitter
        # around a threshold from repeatedly changing the settled state.
        elevation = (points[shoulder, 1] - points[wrist, 1]) / width
        if elevation >= 0.15:
            return "UP"
        if elevation <= -0.25:
            return "DOWN"
        if previous == "UP" and elevation > 0.05:
            return "UP"
        if previous == "DOWN" and elevation < -0.12:
            return "DOWN"
        return "UNKNOWN"

    def update(self, points: np.ndarray | None, timestamp: float) -> dict:
        """Return detached points plus current features; keep no frame images."""
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite monotonic seconds")
        if self._previous_time is not None and timestamp <= self._previous_time:
            raise ValueError("timestamps must increase strictly")
        current = np.zeros((17, 3), dtype=np.float64)
        if points is not None:
            source = np.asarray(points, dtype=np.float64)
            if source.shape != (17, 3):
                raise ValueError("points must have shape (17, 3): x, y, confidence")
            valid = (np.isfinite(source).all(axis=1)
                     & (source[:, :2] >= 0).all(axis=1)
                     & (source[:, :2] <= 1).all(axis=1)
                     & (source[:, 2] >= self.threshold)
                     & (source[:, 2] <= 1))
            current[valid] = source[valid]
        valid = current[:, 2] >= self.threshold
        dt = None if self._previous_time is None else timestamp - self._previous_time
        consecutive = (dt is not None and dt <= 0.25
                       and self._previous_points is not None)
        if consecutive and self.smoothing_seconds > 0:
            common = valid & (self._previous_points[:, 2] >= self.threshold)
            alpha = 1 - math.exp(-dt / self.smoothing_seconds)
            current[common, :2] = (alpha * current[common, :2]
                                      + (1 - alpha) * self._previous_points[common, :2])
        metric = self._metric(current[:, :2])
        shoulder_width = float(np.linalg.norm(metric[5] - metric[6]))
        tracked = bool(valid[[5, 6]].all() and valid[7:13].sum() >= 2
                       and shoulder_width >= 0.035)
        # A close-up can show a clear face/shoulder or one arm while failing the
        # two-shoulder motion gate. Preserve that useful framing feedback.
        partial_evidence = bool(
            (valid[:13].sum() >= 3 and valid[5:7].any() and valid[7:11].any())
            or (valid[:5].sum() >= 3 and valid[5:7].any())
            or (valid[5:7].all() and shoulder_width >= 0.035)
        )
        person_present = tracked or partial_evidence
        tracking_level = ("FULL" if tracked and valid[11:17].all()
                          else "UPPER_BODY" if tracked
                          else "PARTIAL" if person_present else "NONE")
        events: list[str] = []
        if tracked:
            if not self._tracking_session:
                self._tracking_session = True
                events.append("TRACKING_ACQUIRED")
            self._last_seen = timestamp
            status = "TRACKING"
        else:
            expired = (self._last_seen is None
                       or timestamp - self._last_seen >= self.loss_seconds)
            status = "SEARCHING" if expired else "OCCLUDED"
            if expired and self._tracking_session:
                events.append("TRACKING_LOST")
                self._tracking_session = False
            if person_present:
                status = "PARTIAL"

        angles: dict[str, float | None] = {}
        for side, shoulder, elbow, wrist in (("left", 5, 7, 9), ("right", 6, 8, 10)):
            state = self._arms[side]
            candidate = "UNKNOWN"
            angles[f"{side}_elbow"] = None
            if tracked:
                candidate = self._arm_candidate(current, valid, shoulder, elbow,
                                                wrist, shoulder_width, state.value)
                if valid[[shoulder, elbow, wrist]].all():
                    angles[f"{side}_elbow"] = self._angle(
                        metric[shoulder], metric[elbow], metric[wrist])
            if state.update(candidate, timestamp, self.debounce_seconds):
                events.append(f"{side.upper()}_ARM_{state.value}")

        lean_candidate = "UNKNOWN"
        if tracked and valid[[11, 12]].all():
            shoulder_midpoint = (metric[5] + metric[6]) / 2
            hip_midpoint = (metric[11] + metric[12]) / 2
            torso_height = hip_midpoint[1] - shoulder_midpoint[1]
            if torso_height >= 0.15 * shoulder_width:
                offset = (shoulder_midpoint[0] - hip_midpoint[0]) / shoulder_width
                if offset < -0.20 or (self._lean.value == "LEFT" and offset < -0.12):
                    lean_candidate = "LEFT"
                elif offset > 0.20 or (self._lean.value == "RIGHT" and offset > 0.12):
                    lean_candidate = "RIGHT"
                else:
                    lean_candidate = "CENTER"
        if self._lean.update(lean_candidate, timestamp, self.debounce_seconds):
            events.append(f"LEAN_{self._lean.value}")

        motion_speed = None
        if tracked and self._previous_tracked and consecutive and dt >= 0.001:
            common = valid & (self._previous_points[:, 2] >= self.threshold)
            common[:5] = False  # Face landmark jitter is not body movement.
            common[13:] = False  # This project measures upper-body movement.
            if common.sum() >= 4:
                delta = self._metric(current[common, :2]
                                     - self._previous_points[common, :2])
                raw_speed = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
                raw_speed /= shoulder_width * dt
                alpha = 1 - math.exp(-dt / 0.15)
                motion_speed = (raw_speed if self._speed is None
                                else alpha * raw_speed + (1 - alpha) * self._speed)
        self._speed = motion_speed
        self._previous_points = current.copy()
        self._previous_time = timestamp
        self._previous_tracked = tracked
        return {
            "points": current,
            "tracked": tracked,
            "person_present": person_present,
            "tracking_level": tracking_level,
            "status": status,
            "left_arm": self._arms["left"].value,
            "right_arm": self._arms["right"].value,
            "lean": self._lean.value,
            "angles": angles,
            "motion_speed": motion_speed,
            "events": events,
        }
