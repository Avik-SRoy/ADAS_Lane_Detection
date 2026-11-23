import numpy as np


class LaneSmoother:
    """Legacy EMA lane smoother retained for backward compatibility."""

    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.left_lane = None
        self.right_lane = None

    def smooth(self, new_left, new_right):
        if new_left is not None:
            new_left = np.array(new_left)
            if self.left_lane is None:
                self.left_lane = new_left
            else:
                self.left_lane = (
                    self.alpha * self.left_lane + (1 - self.alpha) * new_left
                )

        if new_right is not None:
            new_right = np.array(new_right)
            if self.right_lane is None:
                self.right_lane = new_right
            else:
                self.right_lane = (
                    self.alpha * self.right_lane + (1 - self.alpha) * new_right
                )

        return self.left_lane, self.right_lane


class _PolyKalman1D:
    """Per-lane 3-state Kalman filter tracking 2nd-order polynomial [a, b, c]."""

    # Relative scale factors for [a (y^2), b (y), c (const)] coefficients
    _SCALE = np.array([1e-6, 1e-2, 100.0], dtype=np.float64)

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.q_base = process_noise * (self._SCALE ** 2)
        self.r_base = measurement_noise * (self._SCALE ** 2)
        self.x = None
        self.P = None
        self.missed = 0

    def update(self, measurement, confidence=1.0):
        """Predict and optionally update with a new [a, b, c] polynomial fit."""
        if measurement is not None:
            z = np.asarray(measurement, dtype=np.float64)
            if self.x is None:
                self.x = z.copy()
                self.P = self.r_base.copy()
                self.missed = 0
                return self.x

            # Predict step (random-walk process model)
            P_pred = self.P + self.q_base

            # Scale measurement noise inversely with detection confidence
            conf_clamped = max(0.15, float(confidence))
            R = self.r_base / conf_clamped

            # Diagonal Kalman gain and state update
            K = P_pred / (P_pred + R)
            self.x = self.x + K * (z - self.x)
            self.P = (1.0 - K) * P_pred
            self.missed = 0
            return self.x

        # Measurement missing: coast on prior estimate and grow uncertainty
        if self.x is not None:
            self.P = self.P + self.q_base * 2.0
            self.missed += 1
        return self.x

    def reset(self):
        self.x = None
        self.P = None
        self.missed = 0


class KalmanLaneTracker:
    """Tracks left and right lane polynomials with confidence and occlusion handling."""

    def __init__(self, cfg):
        q = cfg["process_noise"]
        r = cfg["measurement_noise"]
        self.max_missed = cfg["max_missed_frames"]
        self.left_kf = _PolyKalman1D(q, r)
        self.right_kf = _PolyKalman1D(q, r)
        self.confidence = 0.0

    def update(self, detection):
        """Update Kalman trackers from PolynomialLaneDetector output."""
        left_fit = self.left_kf.update(
            detection["left_fit"], detection["left_conf"]
        )
        right_fit = self.right_kf.update(
            detection["right_fit"], detection["right_conf"]
        )

        if self.left_kf.missed > self.max_missed:
            self.left_kf.reset()
            left_fit = None
        if self.right_kf.missed > self.max_missed:
            self.right_kf.reset()
            right_fit = None

        # Compute overall tracking confidence factoring in coasting decay
        left_score = (
            max(0.0, 1.0 - self.left_kf.missed / self.max_missed)
            if left_fit is not None
            else 0.0
        )
        right_score = (
            max(0.0, 1.0 - self.right_kf.missed / self.max_missed)
            if right_fit is not None
            else 0.0
        )
        self.confidence = 0.5 * (left_score + right_score)

        return {
            "left_fit": left_fit,
            "right_fit": right_fit,
            "confidence": self.confidence,
            "left_missed": self.left_kf.missed,
            "right_missed": self.right_kf.missed,
        }
