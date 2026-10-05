"""Camera-free face presentation tests using known landmarks and timestamps."""

import unittest

import numpy as np

from motion_presentation import MotionPresentation


def face_state(body=False):
    points = np.zeros((17, 3), dtype=float)
    points[:5] = [[.5, .3, .9], [.46, .27, .8], [.54, .27, .8],
                  [.4, .3, .7], [.6, .3, .7]]
    return {"points": points, "tracked": body, "person_present": body,
            "tracking_level": "UPPER_BODY" if body else "NONE",
            "status": "TRACKING" if body else "SEARCHING",
            "left_arm": "DOWN" if body else "UNKNOWN",
            "right_arm": "DOWN" if body else "UNKNOWN",
            "lean": "CENTER" if body else "UNKNOWN",
            "angles": {"left_elbow": 150. if body else None},
            "motion_speed": .5 if body else None, "events": []}


class MotionPresentationTests(unittest.TestCase):
    def test_face_only_is_tracked_without_claiming_body_actions(self):
        result = MotionPresentation().update(face_state(), 0)
        self.assertTrue(result["tracked"])
        self.assertTrue(result["person_present"])
        self.assertTrue(result["face_tracked"])
        self.assertFalse(result["body_tracked"])
        self.assertEqual(result["tracking_level"], "FACE")
        self.assertEqual(result["status"], "FACE_TRACKING")
        self.assertEqual(result["left_arm"], "UNKNOWN")
        self.assertIsNone(result["angles"]["left_elbow"])
        self.assertIsNone(result["motion_speed"])
        self.assertIsNone(result["face_motion"])
        self.assertEqual(result["events"], ["FACE_ACQUIRED"])

    def test_static_face_has_zero_motion_without_repeating_events(self):
        adapter = MotionPresentation()
        adapter.update(face_state(), 0)
        result = adapter.update(face_state(), .1)
        self.assertEqual(result["face_motion"], {
            "dx": 0., "dy": 0., "speed": 0., "unit": "image_fraction_per_second",
        })
        self.assertEqual(result["events"], [])

    def test_known_translation_has_correct_normalized_velocity(self):
        adapter = MotionPresentation()
        adapter.update(face_state(), 0)
        moved = face_state()
        moved["points"][:5, :2] += [.03, .04]
        result = adapter.update(moved, .1)
        self.assertAlmostEqual(result["face_motion"]["dx"], .3)
        self.assertAlmostEqual(result["face_motion"]["dy"], .4)
        self.assertAlmostEqual(result["face_motion"]["speed"], .5)

    def test_dropout_clears_face_immediately_and_reacquisition_resets_speed(self):
        adapter = MotionPresentation()
        adapter.update(face_state(), 0)
        empty = face_state()
        empty["points"][:] = 0
        result = adapter.update(empty, .1)
        self.assertFalse(result["tracked"])
        self.assertFalse(result["face_tracked"])
        self.assertIsNone(result["face_center"])
        self.assertIsNone(result["face_motion"])
        self.assertEqual(result["events"], ["FACE_LOST"])
        self.assertEqual(adapter.update(empty, .2)["events"], [])
        moved = face_state()
        moved["points"][:5, 0] += .2
        recovered = adapter.update(moved, .3)
        self.assertIsNone(recovered["face_motion"])
        self.assertEqual(recovered["events"], ["FACE_ACQUIRED"])

    def test_body_state_survives_face_addition_and_face_loss(self):
        adapter = MotionPresentation()
        body = face_state(body=True)
        body["events"] = ["LEFT_ARM_DOWN"]
        result = adapter.update(body, 0)
        self.assertTrue(result["body_tracked"])
        self.assertEqual(result["status"], "TRACKING")
        self.assertEqual(result["tracking_level"], "UPPER_BODY")
        self.assertEqual(result["motion_speed"], .5)
        self.assertEqual(result["events"], ["LEFT_ARM_DOWN", "FACE_ACQUIRED"])
        body["points"][:5] = 0
        result = adapter.update(body, .1)
        self.assertTrue(result["tracked"])
        self.assertFalse(result["face_tracked"])
        self.assertEqual(result["left_arm"], "DOWN")

    def test_anchor_change_and_sampling_gap_reset_velocity(self):
        adapter = MotionPresentation()
        adapter.update(face_state(), 0)
        eyes = face_state()
        eyes["points"][0, 2] = 0
        result = adapter.update(eyes, .1)
        self.assertTrue(result["face_tracked"])
        self.assertAlmostEqual(result["face_center"]["y"], .27)
        self.assertIsNone(result["face_motion"])
        self.assertIsNotNone(adapter.update(eyes, .2)["face_motion"])
        self.assertIsNone(adapter.update(eyes, .5)["face_motion"])
        self.assertIsNone(adapter.update(face_state(), .6)["face_motion"])

    def test_face_needs_three_valid_points_and_a_noncollapsed_pair(self):
        adapter = MotionPresentation()
        points = face_state()
        points["points"][0, 2] = 0
        points["points"][3:, 2] = 0
        self.assertFalse(adapter.update(points, 0)["face_tracked"])
        points = face_state()
        points["points"][:5, :2] = [.5, .3]
        self.assertFalse(adapter.update(points, .1)["face_tracked"])
        points = face_state()
        points["points"][[1, 3], 2] = 0
        self.assertFalse(adapter.update(points, .2)["face_tracked"])

    def test_invalid_face_points_are_not_used(self):
        adapter = MotionPresentation()
        points = face_state()
        points["points"][0, 0] = float("nan")
        points["points"][1, 0] = -1
        points["points"][3, 2] = .29
        result = adapter.update(points, 0)
        self.assertFalse(result["face_tracked"])
        self.assertIsNone(result["face_center"])

    def test_output_is_detached_and_does_not_mutate_body_input(self):
        adapter = MotionPresentation()
        source = face_state()
        original_points = source["points"].copy()
        output = adapter.update(source, 0)
        self.assertNotIn("face_tracked", source)
        self.assertFalse(source["tracked"])
        self.assertEqual(source["events"], [])
        output["points"][:] = 0
        output["angles"]["left_elbow"] = 45
        np.testing.assert_array_equal(source["points"], original_points)
        self.assertIsNone(source["angles"]["left_elbow"])
        self.assertEqual(adapter.update(source, .1)["face_motion"]["speed"], 0)

    def test_mesh_tracks_without_any_pose_landmarks_or_fabricated_confidence(self):
        adapter = MotionPresentation()
        state = face_state()
        state["points"][:] = 0
        mesh = np.tile([.5, .3, -.02], (478, 1))
        detection = {"detected": True, "landmarks": mesh, "bbox": [.3, .1, .7, .6]}
        result = adapter.update(state, 0, detection)
        self.assertTrue(result["tracked"])
        self.assertFalse(result["body_tracked"])
        self.assertEqual(result["face_mode"], "mesh")
        self.assertEqual(len(result["face_landmarks"]), 478)
        self.assertEqual(result["face_bbox"], [.3, .1, .7, .6])
        self.assertEqual(result["face_center"], {"x": .5, "y": .3, "confidence": None})
        result["face_landmarks"][1][0] = 0
        self.assertEqual(mesh[1, 0], .5)
        mesh[:, :2] += [.03, .04]
        moved = adapter.update(state, .1, detection)
        self.assertAlmostEqual(moved["face_motion"]["dx"], .3)
        self.assertAlmostEqual(moved["face_motion"]["dy"], .4)
        self.assertAlmostEqual(moved["face_motion"]["speed"], .5)

    def test_mesh_loss_or_switch_to_pose_resets_velocity_and_geometry(self):
        adapter = MotionPresentation()
        mesh = np.tile([.5, .3, 0.], (468, 1))
        adapter.update(face_state(), 0, {"detected": True, "landmarks": mesh})
        fallback = adapter.update(face_state(), .1, {"detected": False})
        self.assertTrue(fallback["face_tracked"])
        self.assertEqual(fallback["face_mode"], "pose")
        self.assertEqual(fallback["face_landmarks"], [])
        self.assertIsNone(fallback["face_bbox"])
        self.assertIsNone(fallback["face_motion"])
        empty = face_state()
        empty["points"][:] = 0
        absent = adapter.update(empty, .2, {"detected": False})
        self.assertFalse(absent["face_tracked"])
        self.assertEqual(absent["face_mode"], "none")
        self.assertEqual(absent["events"], ["FACE_LOST"])

    def test_malformed_mesh_falls_back_without_publishing_invalid_geometry(self):
        adapter = MotionPresentation()
        mesh = np.tile([.5, .3, 0.], (478, 1))
        mesh[50, 2] = float("nan")
        result = adapter.update(face_state(), 0, {"detected": True, "landmarks": mesh})
        self.assertEqual(result["face_mode"], "pose")
        self.assertEqual(result["face_landmarks"], [])
        short = adapter.update(face_state(), .1, {"detected": True, "landmarks": mesh[:5]})
        self.assertEqual(short["face_mode"], "pose")


if __name__ == "__main__":
    unittest.main()
