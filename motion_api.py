"""Local read-only HTTP API for numerical pose data; no image transport.

The capture loop owns a StateStore and publishes dictionaries. HTTP readers
never block the capture loop, and each SSE reader retains only the latest state.
This module uses only the Python standard library.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


from motion_topology import BODY_EDGES, FACE_CONTOURS, KEYPOINT_NAMES
SCHEMA_VERSION = "1.1"
TRACKING_LEVELS = ("FULL", "UPPER_BODY", "PARTIAL", "FACE", "NONE")
FACE_MODES = ("mesh", "pose", "none")


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _default_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "sequence": 0,
        "timestamp_unix_s": time.time(), "elapsed_s": 0.0,
        "tracked": False, "body_tracked": False, "face_tracked": False,
        "person_present": False,
        "tracking_level": "NONE", "status": "starting",
        "left_arm": "unknown", "right_arm": "unknown", "lean": "unknown",
        "angles": {}, "motion_speed": None,
        "points": [[0.0, 0.0, 0.0] for _ in KEYPOINT_NAMES],
        "keypoint_names": list(KEYPOINT_NAMES),
        "fps": 0.0, "fps_limit": 20.0, "inference_ms": None, "frame_age_ms": None,
        "events": [], "recent_events": [],
        "camera_fps": 0.0, "reliable_points": 0,
        "source_resolution": [0, 0], "model": "",
        "error": None, "camera_status": "starting",
        "face_center": None, "face_motion": None, "face_landmarks": [],
        "face_bbox": None, "face_mode": "none",
    }


def _validate_state(state: dict[str, Any]) -> None:
    """Validate the finite, image-free public contract before publication."""
    # JSON serialization rejects NaN/Infinity. These checks also reject bools
    # masquerading as numbers and arbitrary nested data in numerical fields.
    def numeric(value: Any) -> bool:
        return type(value) in (int, float)

    for name in ("timestamp_unix_s", "elapsed_s", "fps", "camera_fps"):
        if not numeric(state[name]):
            raise ValueError(f"{name} must be a finite number")
    for name in ("elapsed_s", "fps", "camera_fps"):
        if state[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not numeric(state["fps_limit"]) or not 1 <= state["fps_limit"] <= 30:
        raise ValueError("fps_limit must be a finite number in [1,30]")
    for name in ("motion_speed", "inference_ms", "frame_age_ms"):
        if state[name] is not None and not numeric(state[name]):
            raise ValueError(f"{name} must be a finite number or null")
    for name in ("status", "left_arm", "right_arm", "lean", "model", "camera_status"):
        if not isinstance(state[name], str):
            raise ValueError(f"{name} must be a string")
    if state["error"] is not None and not isinstance(state["error"], str):
        raise ValueError("error must be a string or null")
    for name in ("tracked", "body_tracked", "face_tracked"):
        if type(state[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if type(state["person_present"]) is not bool:
        raise ValueError("person_present must be a boolean")
    if state["tracking_level"] not in TRACKING_LEVELS:
        raise ValueError("tracking_level must be FULL, UPPER_BODY, PARTIAL, FACE, or NONE")
    if type(state["reliable_points"]) is not int or not 0 <= state["reliable_points"] <= 17:
        raise ValueError("reliable_points must be an integer in [0,17]")
    resolution = state["source_resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2 or any(
        type(value) is not int or value < 0 for value in resolution
    ):
        raise ValueError("source_resolution must be [width,height] nonnegative integers")
    points = state["points"]
    if not isinstance(points, list) or len(points) != 17:
        raise ValueError("points must contain exactly 17 [x,y,confidence] entries")
    for point in points:
        if not isinstance(point, list) or len(point) != 3 or not all(map(numeric, point)):
            raise ValueError("each point must be three finite numbers [x,y,confidence]")
        if not 0 <= point[2] <= 1:
            raise ValueError("point confidence must be in [0,1]")
    angles = state["angles"]
    if not isinstance(angles, dict) or any(
        not isinstance(name, str) or (value is not None and not numeric(value))
        for name, value in angles.items()
    ):
        raise ValueError("angles must map string names to finite numbers or null")
    events = state["events"]
    if not isinstance(events, list) or not all(isinstance(item, str) for item in events):
        raise ValueError("events must be a list of strings")
    if state["face_mode"] not in FACE_MODES:
        raise ValueError("face_mode must be mesh, pose, or none")
    center = state["face_center"]
    if center is not None:
        if not isinstance(center, dict) or set(center) != {"x", "y", "confidence"}:
            raise ValueError("face_center must be null or {x,y,confidence}")
        if not numeric(center["x"]) or not numeric(center["y"]):
            raise ValueError("face_center x/y must be finite numbers")
        confidence = center["confidence"]
        if confidence is not None and (not numeric(confidence) or not 0 <= confidence <= 1):
            raise ValueError("face_center confidence must be null or a number in [0,1]")
    movement = state["face_motion"]
    if movement is not None:
        if not isinstance(movement, dict) or set(movement) != {"dx", "dy", "speed", "unit"}:
            raise ValueError("face_motion must be null or {dx,dy,speed,unit}")
        if not all(numeric(movement[name]) for name in ("dx", "dy", "speed")):
            raise ValueError("face_motion dx/dy/speed must be finite numbers")
        if movement["speed"] < 0 or movement["unit"] != "image_fraction_per_second":
            raise ValueError("face_motion needs nonnegative speed and image_fraction_per_second unit")
    landmarks = state["face_landmarks"]
    if not isinstance(landmarks, list) or len(landmarks) not in (0, 468, 478):
        raise ValueError("face_landmarks must contain 0, 468, or 478 [x,y,z] entries")
    for point in landmarks:
        if not isinstance(point, list) or len(point) != 3 or not all(map(numeric, point)):
            raise ValueError("each face landmark must contain three finite numbers [x,y,z]")
    bbox = state["face_bbox"]
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(map(numeric, bbox)):
            raise ValueError("face_bbox must be null or four finite [xmin,ymin,xmax,ymax] numbers")
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError("face_bbox minimum coordinates must not exceed maximums")


class StateStore:
    """Thread-safe single-state mailbox.

    ``publish`` and ``snapshot`` return detached dictionaries. Sequence numbers
    are owned by the store, including when the publisher supplies a sequence.
    Non-JSON values, NaN, and infinity are rejected without replacing the state.
    A closed store keeps its final snapshot available to ordinary readers.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._json = _encode(_default_state())
        self._recent_events: list[dict[str, Any]] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("pose payload must be a dictionary")
        # Copy before entering the lock: callers can safely reuse their lists
        # after this method returns, and serializing cannot stall SSE readers.
        state = _default_state()
        # Explicit projection happens before serialization: even a raw ndarray
        # accidentally attached by a producer cannot enter the public payload.
        projected = {name: value for name, value in payload.items()
                     if name in state and name not in
                     ("sequence", "schema_version", "keypoint_names", "recent_events")}
        for name, allowed in (("face_center", {"x", "y", "confidence"}),
                              ("face_motion", {"dx", "dy", "speed", "unit"})):
            if isinstance(projected.get(name), dict):
                projected[name] = {key: value for key, value in projected[name].items()
                                   if key in allowed}
        copied = json.loads(_encode(projected))
        state.update(copied)
        state["schema_version"] = SCHEMA_VERSION
        state["keypoint_names"] = list(KEYPOINT_NAMES)
        _validate_state(state)
        with self._condition:
            if self._closed:
                raise RuntimeError("StateStore is closed")
            state["sequence"] = self._sequence + 1
            recent_events = (self._recent_events + [
                {"event": event, "timestamp_unix_s": state["timestamp_unix_s"],
                 "sequence": state["sequence"]}
                for event in state["events"]
            ])[-3:]
            state["recent_events"] = recent_events
            encoded = _encode(state)
            self._sequence += 1
            self._json = encoded
            self._recent_events = recent_events
            self._condition.notify_all()
        return json.loads(encoded)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            encoded = self._json
        return json.loads(encoded)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _wait_after(self, sequence: int, stop: threading.Event,
                    timeout: float = 1.0) -> tuple[int, str] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence or self._closed or stop.is_set(),
                timeout=timeout,
            )
            if stop.is_set() or self._closed:
                return None
            if self._sequence <= sequence:
                return None
            return self._sequence, self._json


def metadata_document() -> dict[str, Any]:
    """Geometry and visibility rules needed to reconstruct the motion GUI."""
    return {
        "schema_version": SCHEMA_VERSION,
        "keypoint_names": list(KEYPOINT_NAMES),
        "body_edges": [list(edge) for edge in BODY_EDGES],
        "face_edges": [list(edge) for edge in FACE_CONTOURS],
        "coordinates": {"x": "right", "y": "down", "space": "normalized_image",
                        "face_z": "relative_not_metric", "gui_mirrored": True},
        "point_visibility": {
            "body_min_confidence": 0.3, "body_confidence_operator": ">",
            "body_in_bounds": True, "face_in_bounds": True,
            "valid_xy_range": [0.0, 1.0],
            "body_requires_person_present": True,
            "body_excluded_tracking_levels": ["NONE"],
            "face_requires_face_tracked": True,
            "hide_body_indices_when_mesh_has_visible_points": [0, 1, 2, 3, 4],
        },
    }


def openapi_document() -> dict[str, Any]:
    number_or_null = {"type": ["number", "null"]}
    properties = {
        "schema_version": {"type": "string", "const": SCHEMA_VERSION},
        "sequence": {"type": "integer", "minimum": 0,
                     "description": "Monotonic frame revision for this process."},
        "timestamp_unix_s": {"type": "number",
                             "description": "Time of state sample, Unix seconds."},
        "elapsed_s": {"type": "number", "minimum": 0},
        "tracked": {"type": "boolean",
                    "description": "A reliable body or face is tracked. Since schema 1.1, face-only tracking also sets this true; use body_tracked for the earlier body reliability criterion."},
        "body_tracked": {"type": "boolean",
                         "description": "Body pose meets the reliability criterion required for body angles and arm actions."},
        "face_tracked": {"type": "boolean",
                         "description": "Face movement is tracked, using the source identified by face_mode."},
        "person_present": {"type": "boolean",
                           "description": "The tracker has evidence of a person, including a partially visible body."},
        "tracking_level": {"type": "string", "enum": list(TRACKING_LEVELS),
                           "description": "Current extent of reliable pose observation. Use with per-point confidence; PARTIAL can still provide usable visible joints."},
        "status": {"type": "string"},
        "left_arm": {"type": "string"},
        "right_arm": {"type": "string"},
        "lean": {"type": "string"},
        "angles": {"type": "object", "additionalProperties": number_or_null,
                   "description": "Named joint angles in degrees; null if unavailable."},
        "motion_speed": {**number_or_null,
                         "description": "Body movement in shoulder widths per second; null if unavailable. Face velocity uses image fractions per second instead."},
        "points": {
            "type": "array", "minItems": 17, "maxItems": 17,
            "description": "COCO17 in camera image coordinates, unmirrored. Each point is [x,y,confidence]; x/y normalized to image dimensions. Confidence 0 means unavailable. UI may mirror display separately.",
            "items": {
                "type": "array", "minItems": 3, "maxItems": 3,
                "prefixItems": [{"type": "number"}, {"type": "number"},
                                {"type": "number", "minimum": 0, "maximum": 1}],
                "items": False,
            },
        },
        "keypoint_names": {"type": "array", "const": list(KEYPOINT_NAMES),
                           "items": {"type": "string"}},
        "fps": {"type": "number", "minimum": 0},
        "fps_limit": {"type": "number", "minimum": 1, "maximum": 30,
                      "description": "Current configured tracking FPS cap, adjustable locally in the GUI. The read-only API cannot change it."},
        "inference_ms": {**number_or_null, "description": "Model inference duration in milliseconds."},
        "frame_age_ms": {**number_or_null, "description": "Age of source frame at publication, in milliseconds."},
        "events": {"type": "array", "items": {"type": "string"},
                   "description": "Best-effort events associated with this frame. SSE is latest-state delivery, not a durable event log."},
        "recent_events": {
            "type": "array", "maxItems": 3,
            "items": {"type": "object", "required": ["event", "timestamp_unix_s", "sequence"],
                      "additionalProperties": False,
                      "properties": {"event": {"type": "string"},
                                     "timestamp_unix_s": {"type": "number"},
                                     "sequence": {"type": "integer", "minimum": 1}}},
            "description": "Last three emitted action events, oldest to newest, retained in memory across frames without events. Generated by the store, never accepted from the producer. Clears on application restart; this bounded recent-history window is not a durable log.",
        },
        "camera_fps": {"type": "number", "minimum": 0},
        "reliable_points": {"type": "integer", "minimum": 0, "maximum": 17},
        "source_resolution": {"type": "array", "minItems": 2, "maxItems": 2,
                              "items": {"type": "integer", "minimum": 0},
                              "description": "Camera [width,height] in pixels; no image content."},
        "model": {"type": "string"},
        "error": {"type": ["string", "null"]},
        "camera_status": {"type": "string"},
        "face_center": {
            "type": ["object", "null"], "required": ["x", "y", "confidence"],
            "additionalProperties": False,
            "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                           "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1}},
            "description": "Face center in unmirrored normalized camera image coordinates. Confidence is null when the model does not expose a confidence score; it is never invented.",
        },
        "face_motion": {
            "type": ["object", "null"], "required": ["dx", "dy", "speed", "unit"],
            "additionalProperties": False,
            "properties": {"dx": {"type": "number"}, "dy": {"type": "number"},
                           "speed": {"type": "number", "minimum": 0},
                           "unit": {"type": "string", "const": "image_fraction_per_second"}},
            "description": "Face center velocity: signed x/y components and nonnegative magnitude, all in normalized image fractions per second. Null when unavailable. These are not physical distances or speeds.",
        },
        "face_landmarks": {
            "type": "array",
            "oneOf": [{"minItems": count, "maxItems": count} for count in (0, 468, 478)],
            "items": {"type": "array", "minItems": 3, "maxItems": 3,
                      "items": {"type": "number"}},
            "description": "Empty when mesh is unavailable; otherwise 468 or 478 MediaPipe face landmarks in native model index order, each [x,y,z]. x/y are normalized unmirrored image coordinates and can extend outside [0,1]. z is model-relative depth, not confidence and not an absolute physical measurement. No camera pixels are included.",
        },
        "face_bbox": {"type": ["array", "null"], "minItems": 4, "maxItems": 4,
                      "items": {"type": "number"},
                      "description": "Normalized unmirrored [xmin,ymin,xmax,ymax] face bounds, or null."},
        "face_mode": {"type": "string", "enum": list(FACE_MODES),
                      "description": "mesh: dedicated facial landmarks; pose: coarse face tracking from body pose points; none: face tracking unavailable."},
    }
    pose_schema = {"type": "object", "required": list(properties),
                   "properties": properties, "additionalProperties": False}
    json_pose = {"description": "Current motion state; no camera image data.",
                 "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PoseState"}}}}
    metadata = metadata_document()
    metadata_schema = {
        "type": "object", "required": list(metadata), "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "keypoint_names": {"type": "array", "const": metadata["keypoint_names"]},
            "body_edges": {"type": "array", "const": metadata["body_edges"],
                           "description": "Independent COCO17 line segments; render only when both endpoints are visible."},
            "face_edges": {"type": "array", "const": metadata["face_edges"],
                           "description": "Independent face contour/iris/nose segments. Skip edges whose indices exceed the available landmark count or whose endpoints are outside the image."},
            "coordinates": {"type": "object", "const": metadata["coordinates"]},
            "point_visibility": {"type": "object", "const": metadata["point_visibility"]},
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {"title": "meganan local API", "version": "1.1.0",
                 "description": "Read-only, local numerical motion data. No authentication; intended for loopback only. Raw frames are never exposed. No CORS permission is granted."},
        "servers": [{"url": "/"}],
        "paths": {
            "/api/v1/state": {"get": {"operationId": "getPoseState", "responses": {"200": json_pose}}},
            "/api/v1/metadata": {"get": {"operationId": "getMotionMetadata", "responses": {"200": {
                "description": "Shared body/face topology, coordinate convention, and visibility rules for rendering the same movement as the GUI.",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/MotionMetadata"}}},
            }}}},
            "/api/v1/events": {"get": {
                "operationId": "streamPoseState",
                "description": "Latest-state frame stream using Server-Sent Events. Each event has event: pose, id: sequence, data: PoseState JSON. Slow readers skip intermediate states, including one-shot values in their events arrays. recent_events retains the last three emitted events across frames, but no durable action-event delivery is provided. Last-Event-ID waits for a newer state, but missed frames are not replayed. Reconnect without that header after a server restart. Heartbeat comments occur every 15 seconds.",
                "parameters": [{"name": "Last-Event-ID", "in": "header", "required": False,
                                "schema": {"type": "integer", "minimum": 0}}],
                "responses": {"200": {"description": "Latest-state stream", "content": {"text/event-stream": {"schema": {"type": "string"}}}}},
            }},
            "/health": {"get": {"operationId": "health", "responses": {"200": {
                "description": "API process liveness. Inspect /api/v1/state for tracking and camera status.",
                "content": {"application/json": {"schema": {
                    "type": "object", "required": ["ok", "store_closed", "sequence", "tracked", "state_age_ms"],
                    "properties": {"ok": {"type": "boolean"}, "store_closed": {"type": "boolean"},
                                   "sequence": {"type": "integer"}, "tracked": {"type": "boolean"},
                                   "state_age_ms": {"type": "number"}},
                }}},
            }}}},
            "/openapi.json": {"get": {"operationId": "openapi", "responses": {"200": {"description": "This OpenAPI 3.1 specification."}}}},
            "/": {"get": {"operationId": "index", "responses": {"200": {"description": "API endpoint index and keypoint metadata."}}}},
        },
        "components": {"schemas": {"PoseState": pose_schema, "MotionMetadata": metadata_schema}},
    }


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True


class MotionAPIServer:
    """Start a local HTTP server without blocking the camera or GUI thread."""

    def __init__(self, store: StateStore, host: str = "127.0.0.1",
                 port: int = 8765) -> None:
        self.store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "meganan/1.1"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(2.0)

            def log_message(self, format: str, *args: Any) -> None:
                # A GUI app should not flood stderr with polling request logs.
                return

            def _json_response(self, status: int, payload: Any) -> None:
                encoded = _encode(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self) -> None:
                try:
                    path = urlsplit(self.path).path
                    if path == "/api/v1/events":
                        self._events()
                    elif path == "/api/v1/state":
                        self._json_response(200, owner.store.snapshot())
                    elif path == "/api/v1/metadata":
                        self._json_response(200, metadata_document())
                    elif path == "/health":
                        snapshot = owner.store.snapshot()
                        self._json_response(200, {
                            "ok": not owner._stop.is_set(),
                            "store_closed": owner.store.closed,
                            "sequence": snapshot["sequence"],
                            "tracked": snapshot["tracked"],
                            "state_age_ms": max(0.0, (time.time() - snapshot["timestamp_unix_s"]) * 1000),
                        })
                    elif path == "/openapi.json":
                        self._json_response(200, openapi_document())
                    elif path == "/":
                        self._json_response(200, {
                            "name": "meganan", "schema_version": SCHEMA_VERSION,
                            "endpoints": {"state": "/api/v1/state", "events": "/api/v1/events",
                                          "metadata": "/api/v1/metadata", "health": "/health",
                                          "openapi": "/openapi.json"},
                            "keypoint_names": list(KEYPOINT_NAMES),
                            "coordinates": "unmirrored normalized camera image x/y",
                            "raw_images": False,
                        })
                    else:
                        self._json_response(404, {"error": "not_found"})
                except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
                    self.close_connection = True

            def _events(self) -> None:
                try:
                    sequence = int(self.headers.get("Last-Event-ID", "-1"))
                except ValueError:
                    sequence = -1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.close_connection = True
                last_write = time.monotonic()
                while not owner._stop.is_set() and not owner.store.closed:
                    newest = owner.store._wait_after(sequence, owner._stop)
                    if newest is not None:
                        sequence, encoded = newest
                        self.wfile.write(f"event: pose\nid: {sequence}\ndata: {encoded}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last_write = time.monotonic()
                    elif time.monotonic() - last_write >= 15.0:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        last_write = time.monotonic()

        self._server = _HTTPServer((host, port), Handler)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> "MotionAPIServer":
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("MotionAPIServer is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._server.serve_forever,
                    kwargs={"poll_interval": 0.1},
                    name="motion-api", daemon=True,
                )
                self._thread.start()
        return self

    def close(self) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
            self.store._wake()
            if self._thread is not None:
                self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=3.0)

    def __enter__(self) -> "MotionAPIServer":
        return self.start()

    def __exit__(self, *_args: Any) -> None:
        self.close()
