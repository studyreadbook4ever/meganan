"""Synthetic geometry/time tests; these do not establish model accuracy."""

import unittest

import numpy as np

from motion_features import MotionAnalyzer


def pose():
    points = np.zeros((17, 3), dtype=float)
    for index, xy in {
        5: (.35, .30), 6: (.65, .30),
        7: (.30, .48), 8: (.70, .48),
        9: (.28, .65), 10: (.72, .65),
        11: (.38, .72), 12: (.62, .72),
    }.items():
        points[index] = [*xy, .9]
    return points


class MotionAnalyzerTests(unittest.TestCase):
    def test_static_pose_settles_without_repeating_events(self):
        analyzer = MotionAnalyzer()
        first = analyzer.update(pose(), 0)
        self.assertIn("TRACKING_ACQUIRED", first["events"])
        self.assertEqual(first["left_arm"], "UNKNOWN")
        settled = analyzer.update(pose(), .2)
        self.assertEqual(settled["left_arm"], "DOWN")
        self.assertEqual(settled["lean"], "CENTER")
        self.assertEqual(settled["motion_speed"], 0)
        again = analyzer.update(pose(), .3)
        self.assertEqual(again["events"], [])

    def test_one_frame_raise_does_not_count_as_arm_raise(self):
        analyzer = MotionAnalyzer(smoothing_seconds=0)
        analyzer.update(pose(), 0)
        analyzer.update(pose(), .2)
        raised = pose()
        raised[9, 1] = .1
        self.assertNotIn("LEFT_ARM_UP", analyzer.update(raised, .25)["events"])
        self.assertEqual(analyzer.update(pose(), .30)["left_arm"], "DOWN")
        analyzer.update(raised, .4)
        result = analyzer.update(raised, .6)
        self.assertIn("LEFT_ARM_UP", result["events"])
        self.assertEqual(result["left_arm"], "UP")

    def test_occlusion_removes_stale_measurements_immediately(self):
        analyzer = MotionAnalyzer()
        analyzer.update(pose(), 0)
        analyzer.update(pose(), .2)
        partial = pose()
        partial[9, 2] = .29
        partial[11:13, 2] = 0
        result = analyzer.update(partial, .3)
        self.assertTrue(result["tracked"])
        self.assertEqual(result["points"][9, 2], 0)
        self.assertIsNone(result["angles"]["left_elbow"])
        self.assertEqual(result["left_arm"], "UNKNOWN")
        self.assertEqual(result["lean"], "UNKNOWN")

    def test_loss_is_finite_and_reacquisition_has_no_speed_spike(self):
        analyzer = MotionAnalyzer()
        analyzer.update(pose(), 0)
        analyzer.update(pose(), .1)
        missing = analyzer.update(None, .2)
        self.assertFalse(missing["tracked"])
        self.assertEqual(missing["status"], "OCCLUDED")
        self.assertFalse(missing["points"].any())
        self.assertIsNone(missing["motion_speed"])
        lost = analyzer.update(None, .7)
        self.assertEqual(lost["status"], "SEARCHING")
        self.assertEqual(lost["events"], ["TRACKING_LOST"])
        self.assertEqual(analyzer.update(None, .8)["events"], [])
        shifted = pose()
        shifted[5:13, 0] += .2
        recovered = analyzer.update(shifted, .9)
        self.assertTrue(recovered["tracked"])
        self.assertIsNone(recovered["motion_speed"])
        self.assertIn("TRACKING_ACQUIRED", recovered["events"])
        np.testing.assert_array_equal(recovered["points"], shifted)

    def test_invalid_coordinates_never_become_visible(self):
        analyzer = MotionAnalyzer()
        points = pose()
        points[9] = [1.1, .4, .9]
        points[10] = [np.nan, .4, .9]
        points[11] = [.4, .7, float("inf")]
        result = analyzer.update(points, 0)
        self.assertTrue(np.isfinite(result["points"]).all())
        self.assertTrue((result["points"][[9, 10, 11]] == 0).all())

    def test_angle_corrects_for_image_aspect_ratio(self):
        analyzer = MotionAnalyzer(aspect_ratio=2, smoothing_seconds=0)
        points = pose()
        points[5, :2] = [.3, .3]
        points[7, :2] = [.3, .5]
        points[9, :2] = [.4, .7]
        result = analyzer.update(points, 0)
        self.assertAlmostEqual(result["angles"]["left_elbow"], 135)

    def test_known_translation_has_shoulder_normalized_speed(self):
        analyzer = MotionAnalyzer(smoothing_seconds=0)
        analyzer.update(pose(), 0)
        translated = pose()
        translated[5:13, 0] += .03
        result = analyzer.update(translated, .1)
        self.assertAlmostEqual(result["motion_speed"], 1)

    def test_lean_requires_hips_and_uses_image_direction(self):
        analyzer = MotionAnalyzer(smoothing_seconds=0)
        points = pose()
        points[5:11, 0] -= .10
        analyzer.update(points, 0)
        result = analyzer.update(points, .2)
        self.assertEqual(result["lean"], "LEFT")
        points[11:13, 2] = 0
        self.assertEqual(analyzer.update(points, .3)["lean"], "UNKNOWN")

    def test_degenerate_geometry_and_bad_timestamps(self):
        analyzer = MotionAnalyzer()
        points = pose()
        points[5] = points[6]
        result = analyzer.update(points, 0)
        self.assertFalse(result["tracked"])
        self.assertIsNone(result["angles"]["left_elbow"])
        with self.assertRaises(ValueError):
            analyzer.update(points, 0)
        with self.assertRaises(ValueError):
            analyzer.update(points, float("nan"))

    def test_returned_points_do_not_alias_internal_history(self):
        analyzer = MotionAnalyzer(smoothing_seconds=0)
        result = analyzer.update(pose(), 0)
        result["points"][:] = 0
        self.assertEqual(analyzer.update(pose(), .1)["motion_speed"], 0)

    def test_partial_face_and_shoulders_stay_visible_without_motion_claim(self):
        analyzer = MotionAnalyzer()
        points = np.zeros((17, 3), dtype=float)
        points[:3] = [[.5, .2, .9], [.45, .18, .9], [.55, .18, .9]]
        points[5:7] = [[.35, .96, .9], [.65, .96, .9]]
        result = analyzer.update(points, 0)
        self.assertTrue(result["person_present"])
        self.assertFalse(result["tracked"])
        self.assertEqual(result["tracking_level"], "PARTIAL")
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["left_arm"], "UNKNOWN")
        self.assertIsNone(result["motion_speed"])
        self.assertEqual(result["events"], [])
        np.testing.assert_array_equal(result["points"], points)

    def test_partial_requires_supporting_body_evidence(self):
        analyzer = MotionAnalyzer()
        points = np.zeros((17, 3), dtype=float)
        points[0] = [.5, .2, .9]
        points[5] = [.35, .5, .9]
        self.assertFalse(analyzer.update(points, 0)["person_present"])
        points[7] = [.25, .6, .9]
        partial = analyzer.update(points, .1)
        self.assertTrue(partial["person_present"])
        self.assertEqual(partial["tracking_level"], "PARTIAL")

    def test_full_body_and_upper_body_classification(self):
        analyzer = MotionAnalyzer()
        points = pose()
        self.assertEqual(analyzer.update(points, 0)["tracking_level"], "UPPER_BODY")
        points[13:17] = [[.4, .85, .9], [.6, .85, .9],
                         [.4, .98, .9], [.6, .98, .9]]
        self.assertEqual(analyzer.update(points, .1)["tracking_level"], "FULL")
        absent = analyzer.update(None, .2)
        self.assertEqual(absent["tracking_level"], "NONE")
        self.assertFalse(absent["person_present"])

    def test_partial_presence_does_not_keep_full_tracking_alive(self):
        analyzer = MotionAnalyzer()
        analyzer.update(pose(), 0)
        cropped = pose()
        cropped[7:, 2] = 0
        partial = analyzer.update(cropped, .6)
        self.assertTrue(partial["person_present"])
        self.assertFalse(partial["tracked"])
        self.assertEqual(partial["status"], "PARTIAL")
        self.assertIn("TRACKING_LOST", partial["events"])


if __name__ == "__main__":
    unittest.main()
