"""Network and concurrency checks for the camera-independent local API."""

import http.client
import json
import socket
import threading
import time
import unittest

from motion_api import KEYPOINT_NAMES, MotionAPIServer, SCHEMA_VERSION, StateStore
from motion_topology import BODY_EDGES, FACE_CONTOURS


def event(response):
    values = {}
    while True:
        line = response.readline().decode("utf-8").rstrip("\r\n")
        if not line:
            return values
        name, value = line.split(":", 1)
        values[name] = value.lstrip()


class StateStoreTests(unittest.TestCase):
    def test_snapshot_is_detached_and_sequences_are_internal(self):
        store = StateStore()
        source = {"points": [[0.2, 0.3, 0.9]] * 17, "sequence": 1000}
        published = store.publish(source)
        source["points"][0][0] = 0.8
        published["points"][0][0] = 0.7
        snapshot = store.snapshot()
        self.assertEqual(snapshot["sequence"], 1)
        self.assertEqual(snapshot["points"][0][0], 0.2)
        self.assertEqual(store.publish({})["sequence"], 2)
        self.assertEqual(snapshot["keypoint_names"], list(KEYPOINT_NAMES))

    def test_nonfinite_value_cannot_replace_valid_state(self):
        store = StateStore()
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                store.publish({"inference_ms": invalid})
        self.assertEqual(store.snapshot()["sequence"], 0)

    def test_image_fields_are_removed_before_serialization(self):
        store = StateStore()
        payload = {"tracked": True, "frame": object(), "image": b"raw image",
                   "jpeg": "private", "base64": "private",
                   "unknown": {"raw_image": object()}}
        public = store.publish(payload)
        self.assertTrue(public["tracked"])
        for key in ("frame", "image", "jpeg", "base64", "unknown"):
            self.assertNotIn(key, public)
            self.assertNotIn(key, store.snapshot())

    def test_point_contract_rejects_malformed_or_nonfinite_coordinates(self):
        store = StateStore()
        invalid_points = [
            [[0, 0, 0]] * 16,
            [[0, 0]] * 17,
            [[0, 0, 1.01]] * 17,
            [[0, 0, -0.01]] * 17,
            [[float("nan"), 0, 0.9]] * 17,
            [[0, True, 0.9]] * 17,
            [["private image", 0, 0.9]] * 17,
        ]
        for points in invalid_points:
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    store.publish({"points": points})
        self.assertEqual(store.snapshot()["sequence"], 0)

    def test_nested_image_data_cannot_escape_through_motion_fields(self):
        store = StateStore()
        for payload in ({"angles": {"image": "private"}},
                        {"events": [{"frame": "private"}]},
                        {"motion_speed": {"image": "private"}}):
            with self.assertRaises(ValueError):
                store.publish(payload)

    def test_partial_visibility_is_preserved_and_level_is_validated(self):
        store = StateStore()
        self.assertFalse(store.snapshot()["person_present"])
        self.assertEqual(store.snapshot()["tracking_level"], "NONE")
        public = store.publish({"tracked": False, "person_present": True,
                                "tracking_level": "PARTIAL", "status": "PARTIAL"})
        self.assertFalse(public["tracked"])
        self.assertTrue(public["person_present"])
        self.assertEqual(public["tracking_level"], "PARTIAL")
        for invalid in ({"person_present": 1}, {"tracking_level": "face"},
                        {"tracking_level": None}, {"tracking_level": {"image": "private"}}):
            with self.assertRaises(ValueError):
                store.publish(invalid)
        self.assertEqual(store.snapshot()["sequence"], 1)

    def test_face_only_mesh_defaults_and_detached_coordinates(self):
        store = StateStore()
        defaults = store.snapshot()
        self.assertEqual(defaults["schema_version"], "1.1")
        self.assertFalse(defaults["body_tracked"])
        self.assertFalse(defaults["face_tracked"])
        self.assertIsNone(defaults["face_center"])
        self.assertIsNone(defaults["face_motion"])
        self.assertIsNone(defaults["face_bbox"])
        self.assertEqual(defaults["face_landmarks"], [])
        self.assertEqual(defaults["face_mode"], "none")
        landmarks = [[0.5, 0.4, -0.01] for _ in range(478)]
        face = {"tracked": True, "body_tracked": False, "face_tracked": True,
                "person_present": True, "tracking_level": "FACE", "status": "FACE_TRACKING",
                "face_center": {"x": 0.5, "y": 0.4, "confidence": None, "image": object()},
                "face_motion": {"dx": -0.2, "dy": 0.1, "speed": 0.2236,
                                "unit": "image_fraction_per_second", "frame": object()},
                "face_landmarks": landmarks, "face_bbox": [0.3, 0.2, 0.7, 0.6],
                "face_mode": "mesh"}
        public = store.publish(face)
        landmarks[0][0] = 0.1
        self.assertTrue(public["tracked"])
        self.assertFalse(public["body_tracked"])
        self.assertTrue(public["face_tracked"])
        self.assertEqual(public["tracking_level"], "FACE")
        self.assertEqual(len(public["face_landmarks"]), 478)
        self.assertEqual(public["face_landmarks"][0], [0.5, 0.4, -0.01])
        self.assertIsNone(public["face_center"]["confidence"])
        self.assertNotIn("image", public["face_center"])
        self.assertNotIn("frame", public["face_motion"])
        self.assertEqual(len(store.publish({"face_landmarks": [[-0.01, 1.01, -0.2]] * 468})["face_landmarks"]), 468)

    def test_face_payload_contract_rejects_invalid_shapes_and_values(self):
        store = StateStore()
        invalid = [
            {"body_tracked": 1}, {"face_tracked": "yes"}, {"face_mode": "image"},
            {"face_center": {"x": 0, "y": 0}},
            {"face_center": {"x": True, "y": 0, "confidence": None}},
            {"face_center": {"x": 0, "y": 0, "confidence": 1.1}},
            {"face_center": {"x": float("inf"), "y": 0, "confidence": None}},
            {"face_motion": {"dx": 0, "dy": 0, "speed": 0, "unit": "pixels"}},
            {"face_motion": {"dx": 0, "dy": 0, "speed": -1, "unit": "image_fraction_per_second"}},
            {"face_motion": {"dx": "image", "dy": 0, "speed": 0, "unit": "image_fraction_per_second"}},
            {"face_landmarks": [[0, 0, 0]] * 17},
            {"face_landmarks": [[0, 0]] * 478},
            {"face_landmarks": [[0, float("nan"), 0]] * 478},
            {"face_landmarks": [[0, 0, "private image"]] * 468},
            {"face_bbox": [0, 0, 1]}, {"face_bbox": [1, 0, 0, 1]},
            {"face_bbox": [0, 0, 1, float("inf")]},
        ]
        for payload in invalid:
            with self.subTest(keys=list(payload)):
                with self.assertRaises(ValueError):
                    store.publish(payload)
        self.assertEqual(store.snapshot()["sequence"], 0)

    def test_fps_limit_is_read_only_metadata_with_bounded_number(self):
        store = StateStore()
        self.assertEqual(store.snapshot()["fps_limit"], 20)
        self.assertEqual(store.publish({"fps_limit": 1})["fps_limit"], 1)
        self.assertEqual(store.publish({"fps_limit": 30})["fps_limit"], 30)
        for cap in (0, 31, True, float("inf"), float("nan"), "20"):
            with self.assertRaises(ValueError):
                store.publish({"fps_limit": cap})
        self.assertEqual(store.snapshot()["fps_limit"], 30)

    def test_recent_events_persist_are_bounded_and_cannot_be_supplied(self):
        store = StateStore()
        self.assertEqual(store.snapshot()["recent_events"], [])
        first = store.publish({"timestamp_unix_s": 100.0, "events": ["first", "second"],
                               "recent_events": [{"image": object()}]})
        self.assertEqual(first["recent_events"], [
            {"event": "first", "timestamp_unix_s": 100.0, "sequence": 1},
            {"event": "second", "timestamp_unix_s": 100.0, "sequence": 1},
        ])
        first["recent_events"][0]["event"] = "mutated outside store"
        quiet = store.publish({"timestamp_unix_s": 101.0, "events": []})
        self.assertEqual(quiet["recent_events"][0]["event"], "first")
        newest = store.publish({"timestamp_unix_s": 102.0, "events": ["third", "fourth"]})
        self.assertEqual([entry["event"] for entry in newest["recent_events"]],
                         ["second", "third", "fourth"])
        self.assertEqual([entry["sequence"] for entry in newest["recent_events"]], [1, 3, 3])
        self.assertEqual(store.snapshot()["recent_events"], newest["recent_events"])
        self.assertEqual(StateStore().snapshot()["recent_events"], [])

    def test_close_wakes_waiter_and_prevents_publish(self):
        store = StateStore()
        result = []
        waiting = threading.Thread(target=lambda: result.append(
            store._wait_after(0, threading.Event(), timeout=10)))
        waiting.start()
        store.close()
        waiting.join(timeout=1)
        self.assertFalse(waiting.is_alive())
        self.assertEqual(result, [None])
        with self.assertRaises(RuntimeError):
            store.publish({})
        self.assertEqual(store.snapshot()["sequence"], 0)

    def test_slow_reader_sees_latest_without_backlog(self):
        store = StateStore()
        for revision in range(100):
            store.publish({"elapsed_s": revision})
        sequence, encoded = store._wait_after(0, threading.Event())
        self.assertEqual(sequence, 100)
        self.assertEqual(json.loads(encoded)["elapsed_s"], 99)


class APITests(unittest.TestCase):
    def setUp(self):
        self.store = StateStore()
        self.api = MotionAPIServer(self.store, port=0).start()
        self.connections = []

    def tearDown(self):
        for connection in self.connections:
            connection.close()
        self.api.close()
        self.store.close()

    def connect(self):
        connection = http.client.HTTPConnection(*self.api.address, timeout=2)
        self.connections.append(connection)
        return connection

    def get(self, path):
        connection = self.connect()
        connection.request("GET", path)
        response = connection.getresponse()
        return response, json.loads(response.read())

    def test_state_health_and_openapi_describe_real_payload(self):
        self.store.publish({"tracked": True, "status": "tracking", "left_arm": "raised",
                            "image": b"must never be served"})
        response, state = self.get("/api/v1/state")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Cache-Control"), "no-store")
        self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        self.assertEqual(state["left_arm"], "raised")
        self.assertNotIn("image", state)
        _, health = self.get("/health")
        self.assertTrue(health["ok"])
        self.assertTrue(health["tracked"])
        self.assertEqual(health["sequence"], state["sequence"])
        _, spec = self.get("/openapi.json")
        schema = spec["components"]["schemas"]["PoseState"]
        self.assertTrue(set(schema["required"]).issubset(state))
        self.assertEqual(set(schema["properties"]), set(state))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["keypoint_names"]["const"], state["keypoint_names"])
        self.assertEqual(len(state["points"]), 17)
        self.assertIn("text/event-stream", spec["paths"]["/api/v1/events"]["get"]["responses"]["200"]["content"])
        response, error = self.get("/camera.jpg")
        self.assertEqual(response.status, 404)
        self.assertEqual(error["error"], "not_found")

    def test_sse_snapshot_new_pose_and_shutdown(self):
        connection = self.connect()
        connection.request("GET", "/api/v1/events")
        response = connection.getresponse()
        self.assertTrue(response.getheader("Content-Type").startswith("text/event-stream"))
        first = event(response)
        self.assertEqual(first["event"], "pose")
        self.assertEqual(first["id"], "0")
        self.store.publish({"tracked": True, "events": ["left_hand_raised"]})
        second = event(response)
        self.assertEqual(second["id"], "1")
        self.assertEqual(json.loads(second["data"])["events"], ["left_hand_raised"])
        started = time.monotonic()
        self.api.close()
        self.assertEqual(response.read(), b"")
        self.assertLess(time.monotonic() - started, 2)

    def test_sse_resume_skips_already_seen_state(self):
        self.store.publish({"elapsed_s": 1})
        connection = self.connect()
        connection.request("GET", "/api/v1/events", headers={"Last-Event-ID": "1"})
        response = connection.getresponse()
        self.store.publish({"elapsed_s": 2})
        state_event = event(response)
        self.assertEqual(state_event["id"], "2")
        self.assertEqual(json.loads(state_event["data"])["elapsed_s"], 2)

    def test_partial_visibility_is_available_in_http_and_sse(self):
        self.store.publish({"tracked": False, "person_present": True,
                            "tracking_level": "PARTIAL", "status": "PARTIAL"})
        _, state = self.get("/api/v1/state")
        self.assertFalse(state["tracked"])
        self.assertTrue(state["person_present"])
        self.assertEqual(state["tracking_level"], "PARTIAL")
        connection = self.connect()
        connection.request("GET", "/api/v1/events")
        response = connection.getresponse()
        streamed = json.loads(event(response)["data"])
        self.assertEqual(streamed, state)
        _, spec = self.get("/openapi.json")
        properties = spec["components"]["schemas"]["PoseState"]["properties"]
        self.assertEqual(properties["person_present"]["type"], "boolean")
        self.assertEqual(properties["tracking_level"]["enum"],
                         ["FULL", "UPPER_BODY", "PARTIAL", "FACE", "NONE"])

    def test_face_only_478_landmarks_flow_through_state_and_sse(self):
        self.store.publish({"tracked": True, "body_tracked": False,
                            "face_tracked": True, "person_present": True,
                            "tracking_level": "FACE", "status": "FACE_TRACKING",
                            "face_center": {"x": 0.5, "y": 0.4, "confidence": None},
                            "face_landmarks": [[0.5, 0.4, -0.03]] * 478,
                            "face_bbox": [0.3, 0.2, 0.7, 0.6], "face_mode": "mesh",
                            "fps_limit": 12.5})
        _, state = self.get("/api/v1/state")
        self.assertEqual(state["schema_version"], SCHEMA_VERSION)
        self.assertTrue(state["tracked"])
        self.assertFalse(state["body_tracked"])
        self.assertTrue(state["face_tracked"])
        self.assertEqual(state["fps_limit"], 12.5)
        connection = self.connect()
        connection.request("GET", "/api/v1/events")
        response = connection.getresponse()
        self.assertEqual(json.loads(event(response)["data"]), state)
        _, spec = self.get("/openapi.json")
        self.assertEqual(spec["info"]["version"], "1.1.0")
        properties = spec["components"]["schemas"]["PoseState"]["properties"]
        self.assertEqual(properties["schema_version"]["const"], "1.1")
        self.assertFalse(properties["face_center"]["additionalProperties"])
        self.assertFalse(properties["face_motion"]["additionalProperties"])
        self.assertEqual([entry["minItems"] for entry in properties["face_landmarks"]["oneOf"]], [0, 468, 478])
        _, index = self.get("/")
        self.assertEqual(index["schema_version"], "1.1")

    def test_metadata_exposes_shared_gui_topology_and_rendering_rules(self):
        response, metadata = self.get("/api/v1/metadata")
        self.assertEqual(response.status, 200)
        self.assertEqual(metadata["schema_version"], "1.1")
        self.assertEqual(metadata["keypoint_names"], list(KEYPOINT_NAMES))
        self.assertEqual(metadata["body_edges"], [list(edge) for edge in BODY_EDGES])
        self.assertEqual(metadata["face_edges"], [list(edge) for edge in FACE_CONTOURS])
        self.assertEqual(len(metadata["body_edges"]), 16)
        self.assertEqual(len(metadata["face_edges"]), 157)
        self.assertTrue(all(len(edge) == 2 and all(0 <= point < 478 for point in edge)
                            for edge in metadata["face_edges"]))
        self.assertTrue(metadata["coordinates"]["gui_mirrored"])
        self.assertEqual(metadata["coordinates"]["face_z"], "relative_not_metric")
        self.assertEqual(metadata["point_visibility"]["body_min_confidence"], 0.3)
        self.assertEqual(metadata["point_visibility"]["body_confidence_operator"], ">")
        self.assertTrue(metadata["point_visibility"]["body_requires_person_present"])
        self.assertEqual(metadata["point_visibility"]["body_excluded_tracking_levels"], ["NONE"])
        self.assertTrue(metadata["point_visibility"]["face_requires_face_tracked"])
        self.assertEqual(metadata["point_visibility"]["hide_body_indices_when_mesh_has_visible_points"],
                         [0, 1, 2, 3, 4])
        _, spec = self.get("/openapi.json")
        self.assertIn("/api/v1/metadata", spec["paths"])
        schema = spec["components"]["schemas"]["MotionMetadata"]
        self.assertEqual(set(schema["required"]), set(metadata))
        for key, value in metadata.items():
            self.assertEqual(schema["properties"][key]["const"], value)
        _, index = self.get("/")
        self.assertEqual(index["endpoints"]["metadata"], "/api/v1/metadata")

    def test_sse_retains_recent_event_after_event_frame_has_passed(self):
        self.store.publish({"events": ["left_arm_raised"], "timestamp_unix_s": 100.0})
        self.store.publish({"events": [], "timestamp_unix_s": 101.0})
        connection = self.connect()
        connection.request("GET", "/api/v1/events")
        response = connection.getresponse()
        state = json.loads(event(response)["data"])
        self.assertEqual(state["events"], [])
        self.assertEqual(state["recent_events"], [
            {"event": "left_arm_raised", "timestamp_unix_s": 100.0, "sequence": 1},
        ])

    def test_disconnected_sse_client_does_not_break_next_reader(self):
        peer = socket.create_connection(self.api.address, timeout=2)
        peer.sendall(b"GET /api/v1/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        peer.recv(2048)
        peer.shutdown(socket.SHUT_RDWR)
        peer.close()
        for _ in range(10):
            self.store.publish({"tracked": True})
        response, state = self.get("/api/v1/state")
        self.assertEqual(response.status, 200)
        self.assertEqual(state["sequence"], 10)

    def test_store_close_ends_stream_without_hanging(self):
        connection = self.connect()
        connection.request("GET", "/api/v1/events")
        response = connection.getresponse()
        event(response)
        self.store.close()
        self.assertEqual(response.read(), b"")
        _, health = self.get("/health")
        self.assertTrue(health["store_closed"])


if __name__ == "__main__":
    unittest.main()
