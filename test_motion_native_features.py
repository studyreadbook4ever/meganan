"""Behavioral parity of native temporal features, without camera/model inference.

The unchanged Python implementations are independent reference oracles. Native
versions also run every existing feature and presentation regression test.
"""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

import _motion_native as native
import test_motion_features as analyzer_tests
import test_motion_presentation as presentation_tests
from motion_features import MotionAnalyzer as PythonAnalyzer
from motion_presentation import MotionPresentation as PythonPresentation


class NativeAnalyzerCompatibility(analyzer_tests.MotionAnalyzerTests):
    def setUp(self):
        self.patch = patch.object(analyzer_tests, "MotionAnalyzer", native.MotionAnalyzer)
        self.patch.start()
        self.addCleanup(self.patch.stop)


class NativePresentationCompatibility(presentation_tests.MotionPresentationTests):
    def setUp(self):
        self.patch = patch.object(presentation_tests, "MotionPresentation", native.MotionPresentation)
        self.patch.start()
        self.addCleanup(self.patch.stop)


class NativeFeatureParity(unittest.TestCase):
    def assert_equivalent(self, actual, expected, path="result"):
        """Check the complete schema and numerical outputs, including missing data."""
        if isinstance(expected, np.ndarray):
            self.assertIsInstance(actual, np.ndarray, path)
            self.assertEqual(actual.dtype, expected.dtype, path)
            self.assertEqual(actual.shape, expected.shape, path)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11,
                                       equal_nan=True, err_msg=path)
        elif isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(actual.keys(), expected.keys(), path)
            for key in expected:
                self.assert_equivalent(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            self.assertIsInstance(actual, type(expected), path)
            self.assertEqual(len(actual), len(expected), path)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_equivalent(left, right, f"{path}[{index}]")
        elif isinstance(expected, float):
            self.assertIsInstance(actual, float, path)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual), path)
            else:
                self.assertTrue(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
                                f"{path}: {actual!r} != {expected!r}")
        else:
            self.assertIsInstance(actual, type(expected), path)
            self.assertEqual(actual, expected, path)

    def test_seeded_analyzer_sequences_with_occlusion_and_config_changes(self):
        rng = np.random.default_rng(687103)
        for kwargs in ({}, {"smoothing_seconds": 0, "debounce_seconds": 0,
                           "loss_seconds": 0}, {"confidence_threshold": .65,
                           "aspect_ratio": 16 / 9, "smoothing_seconds": .4,
                           "debounce_seconds": .06, "loss_seconds": 1.2}):
            python = PythonAnalyzer(**kwargs)
            compiled = native.MotionAnalyzer(**kwargs)
            timestamp = 0.
            for index in range(500):
                timestamp += rng.choice([.0005, .001, .016, .07, .18, .249, .251, .7])
                points = analyzer_tests.pose()
                points[:5] = presentation_tests.face_state()["points"][:5]
                points[:, :2] += rng.normal(0, .025, (17, 2))
                points[:, 2] = rng.choice([0., .29, .3, .65, .9, 1.], 17,
                                          p=[.05, .05, .05, .05, .75, .05])
                if index % 11 == 0:
                    points[:, 2] = .9
                    points[13:17, :2] = [[.4, .8], [.6, .8], [.4, .97], [.6, .97]]
                if index % 7 == 0:
                    points[9:11, 1] = .1
                if index % 17 == 0:
                    points[rng.integers(17), rng.integers(3)] = rng.choice(
                        [np.nan, np.inf, -np.inf, -1., 1.1])
                if index % 31 == 0:
                    points[5] = points[6]
                if index % 47 == 0:
                    points = None
                if index % 89 == 0:
                    python.aspect_ratio = compiled.aspect_ratio = rng.choice([.5, 1., 4 / 3, 2.])
                with self.subTest(config=kwargs, frame=index):
                    self.assert_equivalent(compiled.update(points, timestamp),
                                           python.update(points, timestamp))

    def test_seeded_face_sequences_with_mesh_and_pose_anchor_changes(self):
        rng = np.random.default_rng(80539)
        python, compiled = PythonPresentation(), native.MotionPresentation()
        timestamp = 0.
        for index in range(360):
            timestamp += rng.choice([.0005, .001, .02, .1, .25, .251, .8])
            body = presentation_tests.face_state(body=index % 4 == 0)
            body["points"][:5, :2] += rng.normal(0, .01, (5, 2))
            if index % 7 in (1, 2):
                body["points"][0, 2] = 0
            if index % 7 == 3:
                body["points"][[0, 1], 2] = 0
            if index % 7 == 4:
                body["points"][:] = 0
            body["events"] = ["LEFT_ARM_DOWN"] if index % 17 == 0 else []
            face = None
            if index % 5 in (0, 1, 2):
                mesh = rng.normal([.5, .3, 0.], [.05, .07, .03], (468 if index % 2 else 478, 3))
                face = {"detected": True, "landmarks": mesh}
                if index % 3 == 0:
                    face["bbox"] = [-.1, .2, 1.1, .8]
                if index % 19 == 0:
                    mesh[83, 1] = np.nan
                if index % 23 == 0:
                    mesh[1, 0] = -1
            with self.subTest(frame=index):
                self.assert_equivalent(compiled.update(body, timestamp, face),
                                       python.update(body, timestamp, face))

    def test_hysteresis_debounce_and_gap_boundaries(self):
        python = PythonAnalyzer(smoothing_seconds=0, aspect_ratio=1)
        compiled = native.MotionAnalyzer(smoothing_seconds=0, aspect_ratio=1)
        timestamp = 0.
        # Positive wrist elevation is above the shoulder. Exercise all threshold
        # boundaries from both settled states and interrupt candidate dwell.
        for elevation in [-.3, -.25, -.12, -.120001, 0., .15, .05, .050001,
                          .14999, .15001, -.25001, -.1, -.3]:
            for elapsed in [.0005, .179, .001, .25, .250001]:
                timestamp += elapsed
                points = analyzer_tests.pose()
                points[9, 1] = points[5, 1] - elevation * .3
                self.assert_equivalent(compiled.update(points, timestamp),
                                       python.update(points, timestamp))
        for elapsed in [.001, .499, .000001, .5]:
            timestamp += elapsed
            self.assert_equivalent(compiled.update(None, timestamp), python.update(None, timestamp))

    def test_mesh_shapes_bbox_clipping_and_malformed_fallback(self):
        mesh = np.tile([.5, .3, -.02], (478, 1))
        mesh[0, :2] = [-.2, 1.2]  # Only the nose anchor must be inside the image.
        cases = [None, {}, {"detected": False}, {"detected": True},
                 {"detected": True, "landmarks": "invalid"},
                 {"detected": True, "landmarks": [[1], [1, 2]]}]
        for value in [None, [], np.zeros((0, 3)), mesh[:467], mesh[:468], mesh,
                      mesh.T, np.full((478, 3), np.inf), mesh.tolist()]:
            cases.append({"detected": True, "landmarks": value})
        for bbox in [None, [], [.8, .6, .2, .1], [np.nan, 0, 1, 1],
                     [-5, -2, 5, 2], [.2, .1, .2, .1], [[.2, .1, .8, .6]]]:
            cases.append({"detected": True, "landmarks": mesh, "bbox": bbox})
        for index, face in enumerate(cases):
            with self.subTest(case=index):
                self.assert_equivalent(native.MotionPresentation().update(presentation_tests.face_state(), 0, face),
                                       PythonPresentation().update(presentation_tests.face_state(), 0, face))

    def test_array_conversion_strides_and_source_detachment(self):
        for points in [analyzer_tests.pose().astype(np.float32),
                       np.asfortranarray(analyzer_tests.pose()), analyzer_tests.pose()[::-1],
                       analyzer_tests.pose().tolist()]:
            source = copy.deepcopy(points)
            actual = native.MotionAnalyzer().update(points, 0)
            expected = PythonAnalyzer().update(points, 0)
            self.assert_equivalent(actual, expected)
            actual["points"][:] = 0
            np.testing.assert_equal(points, source)
            body = presentation_tests.face_state()
            body["points"] = points
            self.assert_equivalent(native.MotionPresentation().update(body, 0),
                                   PythonPresentation().update(body, 0))

    def test_invalid_constructor_parameters_and_public_settings(self):
        for kind in ("analyzer", "presentation"):
            original = PythonAnalyzer if kind == "analyzer" else PythonPresentation
            port = native.MotionAnalyzer if kind == "analyzer" else native.MotionPresentation
            for value in [0., -.1, 1.01, np.nan, np.inf, -np.inf]:
                for implementation in (original, port):
                    with self.subTest(kind=kind, value=value, implementation=implementation):
                        with self.assertRaises(ValueError):
                            implementation(value)
        for key in ("aspect_ratio", "smoothing_seconds", "debounce_seconds", "loss_seconds"):
            invalid_values = [np.nan, np.inf, -np.inf, -1.]
            if key == "aspect_ratio":
                invalid_values.append(0.)
            for value in invalid_values:
                for implementation in (PythonAnalyzer, native.MotionAnalyzer):
                    with self.assertRaises(ValueError):
                        implementation(**{key: value})
        compiled = native.MotionAnalyzer()
        for key, value in {"threshold": .6, "aspect_ratio": 4 / 3,
                           "smoothing_seconds": .2, "debounce_seconds": .3,
                           "loss_seconds": .8}.items():
            setattr(compiled, key, value)
            self.assertEqual(getattr(compiled, key), value)

    def test_unaligned_contiguous_views_are_copied_before_native_reads(self):
        def unaligned(values):
            source = np.asarray(values, dtype=np.float64)
            storage = bytearray(source.nbytes + 1)
            result = np.ndarray(source.shape, dtype=np.float64, buffer=storage, offset=1)
            result[:] = source
            self.assertTrue(result.flags.c_contiguous)
            self.assertFalse(result.flags.aligned)
            return result

        points = unaligned(analyzer_tests.pose())
        self.assert_equivalent(native.MotionAnalyzer().update(points, 0),
                               PythonAnalyzer().update(points, 0))
        body = presentation_tests.face_state()
        body["points"] = points
        mesh = unaligned(np.tile([.5, .3, -.02], (478, 1)))
        face = {"detected": True, "landmarks": mesh,
                "bbox": unaligned([.2, .1, .8, .6])}
        self.assert_equivalent(native.MotionPresentation().update(body, 0, face),
                               PythonPresentation().update(body, 0, face))

    def test_exception_parity_and_rejected_frames_do_not_advance_time(self):
        for original, port, argument in [(PythonAnalyzer, native.MotionAnalyzer, analyzer_tests.pose()),
                                         (PythonPresentation, native.MotionPresentation,
                                          presentation_tests.face_state())]:
            python, compiled = original(), port()
            self.assert_equivalent(compiled.update(argument, "1"), python.update(argument, "1"))
            for timestamp in [1, .5, np.nan, np.inf, -np.inf, "bad", None]:
                errors = []
                for implementation in (python, compiled):
                    try:
                        implementation.update(argument, timestamp)
                    except Exception as error:
                        errors.append(type(error))
                self.assertEqual(len(errors), 2)
                self.assertIs(errors[0], errors[1])
            self.assert_equivalent(compiled.update(argument, 1.1), python.update(argument, 1.1))
        for malformed in [[], np.zeros((17, 2)), np.zeros((1, 17, 3)), "bad", [[1], [2, 3]]]:
            for original, port in [(PythonAnalyzer, native.MotionAnalyzer),
                                   (PythonPresentation, native.MotionPresentation)]:
                python, compiled = original(), port()
                invalid = malformed if original is PythonAnalyzer else {"points": malformed}
                errors = []
                for implementation in (python, compiled):
                    try:
                        implementation.update(invalid, 1)
                    except Exception as error:
                        errors.append(type(error))
                self.assertEqual(len(errors), 2)
                self.assertIs(errors[0], errors[1])
                valid = analyzer_tests.pose() if original is PythonAnalyzer else presentation_tests.face_state()
                self.assert_equivalent(compiled.update(valid, 1), python.update(valid, 1))

    def test_detached_unknown_fields_preserve_aliases_and_cycles(self):
        body = presentation_tests.face_state()
        shared = {"nested": [1, 2, {"tuple": (3, 4)}]}
        body["extension"] = {"a": shared, "b": shared}
        body["self"] = body
        body["event_alias"] = body["events"]
        result = native.MotionPresentation().update(body, 0)
        self.assertIs(result["self"], result)
        self.assertIs(result["extension"]["a"], result["extension"]["b"])
        self.assertIsNot(result["extension"]["a"], shared)
        self.assertIsNot(result["event_alias"], result["events"])
        self.assertEqual(result["event_alias"], [])
        result["extension"]["a"]["nested"].append("changed")
        self.assertEqual(len(shared["nested"]), 3)

    def test_bbox_conversion_failure_does_not_commit_presentation_state(self):
        mesh = np.tile([.5, .3, 0.], (478, 1))
        face = {"detected": True, "landmarks": mesh, "bbox": "not a number"}
        python, compiled = PythonPresentation(), native.MotionPresentation()
        for implementation in (python, compiled):
            with self.assertRaises(ValueError):
                implementation.update(presentation_tests.face_state(), 0, face)
        face["bbox"] = None
        self.assert_equivalent(compiled.update(presentation_tests.face_state(), 0, face),
                               python.update(presentation_tests.face_state(), 0, face))


if __name__ == "__main__":
    unittest.main()
