import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from Utils.control import LateralController
from Utils.fcw import ForwardCollisionWarning
from Utils.lane_det import PolynomialLaneDetector
from Utils.object_tracker import ObjectTracker, compute_iou
from Utils.roi import PerspectiveTransformer
from Utils.smoothing import KalmanLaneTracker


BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR / "config.json", "r", encoding="utf-8") as _f:
    CONFIG = json.load(_f)


class TestPerspectiveTransformer(unittest.TestCase):
    def test_warp_and_unwarp_shapes_and_inverse(self):
        w, h = 960, 540
        tf = PerspectiveTransformer(
            w,
            h,
            CONFIG["perspective"]["src_points_norm"],
            CONFIG["perspective"]["dst_points_norm"],
        )
        # M * Minv should be close to identity
        identity = tf.M @ tf.Minv
        identity /= identity[2, 2]
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-4)

        img = np.zeros((h, w), dtype=np.uint8)
        warped = tf.warp(img)
        unwarped = tf.unwarp(warped)
        self.assertEqual(warped.shape, (h, w))
        self.assertEqual(unwarped.shape, (h, w))


class TestPolynomialLaneDetector(unittest.TestCase):
    def _make_synthetic_bev(self, left_fit, right_fit, w=960, h=540, thickness=14):
        bev = np.zeros((h, w), dtype=np.uint8)
        ys = np.linspace(0, h - 1, h)
        lx = np.polyval(left_fit, ys)
        rx = np.polyval(right_fit, ys)

        pts_l = np.array([np.transpose(np.vstack([lx, ys]))], dtype=np.int32)
        pts_r = np.array([np.transpose(np.vstack([rx, ys]))], dtype=np.int32)
        cv2.polylines(bev, pts_l, False, 255, thickness)
        cv2.polylines(bev, pts_r, False, 255, thickness)
        return bev

    def test_detects_curved_lanes_cold_and_warm(self):
        detector = PolynomialLaneDetector(CONFIG["detection"])
        true_left = np.array([2.5e-4, -0.18, 260.0])
        true_right = np.array([2.5e-4, -0.18, 720.0])
        bev = self._make_synthetic_bev(true_left, true_right)

        # Cold start (sliding window)
        res_cold = detector.detect(bev)
        self.assertEqual(res_cold["method"], "sliding_window")
        self.assertIsNotNone(res_cold["left_fit"])
        self.assertIsNotNone(res_cold["right_fit"])
        self.assertGreater(res_cold["left_conf"], 0.5)
        self.assertGreater(res_cold["right_conf"], 0.5)

        # Check accuracy at bottom and mid frame
        for y in (270.0, 539.0):
            self.assertAlmostEqual(
                np.polyval(res_cold["left_fit"], y),
                np.polyval(true_left, y),
                delta=8.0,
            )
            self.assertAlmostEqual(
                np.polyval(res_cold["right_fit"], y),
                np.polyval(true_right, y),
                delta=8.0,
            )

        # Warm start (search around poly)
        res_warm = detector.detect(
            bev, prev_left=res_cold["left_fit"], prev_right=res_cold["right_fit"]
        )
        self.assertEqual(res_warm["method"], "poly_search")
        self.assertIsNotNone(res_warm["left_fit"])
        self.assertIsNotNone(res_warm["right_fit"])

    def test_rejects_empty_frame(self):
        detector = PolynomialLaneDetector(CONFIG["detection"])
        empty = np.zeros((540, 960), dtype=np.uint8)
        res = detector.detect(empty)
        self.assertIsNone(res["left_fit"])
        self.assertIsNone(res["right_fit"])
        self.assertEqual(res["left_conf"], 0.0)


class TestKalmanLaneTracker(unittest.TestCase):
    def test_smoothing_and_occlusion_reset(self):
        tracker_cfg = {
            "process_noise": 1e-4,
            "measurement_noise": 1e-2,
            "max_missed_frames": 3,
        }
        tracker = KalmanLaneTracker(tracker_cfg)
        base_left = np.array([0.0, 0.0, 250.0])
        base_right = np.array([0.0, 0.0, 710.0])

        # Initial update locks immediately
        out = tracker.update({
            "left_fit": base_left,
            "right_fit": base_right,
            "left_conf": 1.0,
            "right_conf": 1.0,
        })
        np.testing.assert_allclose(out["left_fit"], base_left)
        self.assertEqual(out["confidence"], 1.0)

        # Noisy spike should be damped
        noisy_left = np.array([0.0, 0.0, 300.0])
        out2 = tracker.update({
            "left_fit": noisy_left,
            "right_fit": base_right,
            "left_conf": 0.5,
            "right_conf": 1.0,
        })
        self.assertGreater(out2["left_fit"][2], 250.0)
        self.assertLess(out2["left_fit"][2], 280.0)

        # Missing frames decay confidence and eventually reset
        for _ in range(4):
            out_miss = tracker.update({
                "left_fit": None,
                "right_fit": None,
                "left_conf": 0.0,
                "right_conf": 0.0,
            })
        self.assertIsNone(out_miss["left_fit"])
        self.assertIsNone(out_miss["right_fit"])
        self.assertEqual(out_miss["confidence"], 0.0)


class TestLateralController(unittest.TestCase):
    def setUp(self):
        self.ctrl = LateralController(CONFIG["metrics"], CONFIG["control"])
        self.w = 960
        self.h = 540

    def test_centered_straight_lane(self):
        # Symmetric around x = 480 (width 460 px)
        left = np.array([0.0, 0.0, 250.0])
        right = np.array([0.0, 0.0, 710.0])
        out = self.ctrl.compute(self.w, self.h, left, right, confidence=1.0)
        self.assertEqual(out["state"], "LKA ACTIVE")
        self.assertEqual(out["command"], "Go Straight")
        self.assertAlmostEqual(out["offset_m"], 0.0, places=3)
        self.assertAlmostEqual(out["steering_deg"], 0.0, places=2)

    def test_ldw_right_triggers_left_steer(self):
        # Lane shifted left => ego vehicle is to the RIGHT of lane center
        left = np.array([0.0, 0.0, 120.0])
        right = np.array([0.0, 0.0, 580.0])
        out = self.ctrl.compute(self.w, self.h, left, right, confidence=1.0)
        self.assertEqual(out["state"], "LDW RIGHT")
        self.assertGreater(out["offset_m"], 0.45)
        self.assertLess(out["steering_deg"], 0.0)
        self.assertEqual(out["command"], "Steer Left")

    def test_ldw_left_triggers_right_steer(self):
        # Lane shifted right => ego vehicle is to the LEFT of lane center
        left = np.array([0.0, 0.0, 380.0])
        right = np.array([0.0, 0.0, 840.0])
        out = self.ctrl.compute(self.w, self.h, left, right, confidence=1.0)
        self.assertEqual(out["state"], "LDW LEFT")
        self.assertLess(out["offset_m"], -0.45)
        self.assertGreater(out["steering_deg"], 0.0)
        self.assertEqual(out["command"], "Steer Right")

    def test_low_confidence_enters_standby(self):
        left = np.array([0.0, 0.0, 250.0])
        right = np.array([0.0, 0.0, 710.0])
        out = self.ctrl.compute(self.w, self.h, left, right, confidence=0.1)
        self.assertEqual(out["state"], "STANDBY")
        self.assertEqual(out["command"], "Lane Lost")
        self.assertEqual(out["steering_deg"], 0.0)


class TestObjectTrackerAndFCW(unittest.TestCase):
    def setUp(self):
        self.w = 960
        self.h = 540
        self.tf = PerspectiveTransformer(
            self.w,
            self.h,
            CONFIG["perspective"]["src_points_norm"],
            CONFIG["perspective"]["dst_points_norm"],
        )
        self.fcw = ForwardCollisionWarning(CONFIG["fcw"])

    def test_tracker_persists_ids_and_expires(self):
        tracker = ObjectTracker(iou_threshold=0.3, max_age=2)
        det1 = [{"bbox": [400, 300, 500, 380], "class_id": 2, "class_name": "car", "confidence": 0.9}]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 1)
        tid = tracks1[0]["track_id"]

        # Slight motion keeps same track ID
        det2 = [{"bbox": [405, 302, 505, 382], "class_id": 2, "class_name": "car", "confidence": 0.92}]
        tracks2 = tracker.update(det2)
        self.assertEqual(tracks2[0]["track_id"], tid)

        # Expire after max_age + 1 missed frames
        for _ in range(3):
            tracker.update([])
        tracks3 = tracker.update(det1)
        self.assertNotEqual(tracks3[0]["track_id"], tid)

    def test_fcw_ego_lane_and_warning_states(self):
        # Straight ego lane in BEV around [240, 720]
        left_fit = np.array([0.0, 0.0, 240.0])
        right_fit = np.array([0.0, 0.0, 720.0])

        # Project BEV center (480, 300) back to camera space to place a close car in ego lane
        pt_bev = np.array([480.0, 300.0, 1.0])
        cam = self.tf.Minv @ pt_bev
        cx, y2 = cam[0] / cam[2], cam[1] / cam[2]

        # Car height = 100 px => dist = 1.5 * 720 / 100 = 10.8 m (<= 12 m => FCW WARNING)
        tracker = ObjectTracker()
        close_car = [{
            "bbox": [cx - 50, y2 - 100, cx + 50, y2],
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.9,
        }]
        tracked = tracker.update(close_car)
        res = self.fcw.evaluate(tracked, self.tf, left_fit, right_fit, self.w, self.h, dt=0.033)
        self.assertEqual(res["state"], "FCW WARNING")
        self.assertIsNotNone(res["lead_object"])
        self.assertTrue(res["lead_object"]["in_ego_lane"])
        self.assertLessEqual(res["lead_distance_m"], 12.0)

        # Far right oncoming car outside ego lane => CLEAR
        tracker2 = ObjectTracker()
        oncoming_car = [{
            "bbox": [820, 340, 920, 440],
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.85,
        }]
        tracked2 = tracker2.update(oncoming_car)
        res2 = self.fcw.evaluate(tracked2, self.tf, left_fit, right_fit, self.w, self.h, dt=0.033)
        self.assertEqual(res2["state"], "CLEAR")
        self.assertIsNone(res2["lead_object"])

    def test_iou_bounds(self):
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


if __name__ == "__main__":
    unittest.main()
