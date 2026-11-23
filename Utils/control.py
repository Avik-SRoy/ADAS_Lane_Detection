import numpy as np


def compute_lane_center(frame, left_lane, right_lane):
    """Legacy straight-line lane center helper retained for compatibility."""
    if left_lane is None or right_lane is None:
        return None

    height = frame.shape[0]
    y = height
    left_x = int((y - left_lane[1]) / left_lane[0])
    right_x = int((y - right_lane[1]) / right_lane[0])

    lane_center = (left_x + right_x) // 2
    frame_center = frame.shape[1] // 2
    return lane_center, frame_center


def get_steering_direction(lane_center, frame_center, threshold=50):
    """Legacy discrete steering helper retained for compatibility."""
    deviation = lane_center - frame_center
    if deviation > threshold:
        return "Turn Right"
    elif deviation < -threshold:
        return "Turn Left"
    else:
        return "Go Straight"


class LateralController:
    """Computes metric lane geometry, LDW state, and continuous steering angle."""

    def __init__(self, metrics_cfg, control_cfg):
        self.lane_width_m = metrics_cfg["lane_width_m"]
        self.ym_per_pix = metrics_cfg["ym_per_pix"]
        self.default_xm_per_pix = metrics_cfg["xm_per_pix"]

        self.ldw_threshold_m = control_cfg["ldw_threshold_m"]
        self.kp_lateral = control_cfg["kp_lateral"]
        self.kp_heading = control_cfg["kp_heading"]
        self.max_steer_deg = control_cfg["max_steer_deg"]
        self.straight_deadband_deg = control_cfg["straight_deadband_deg"]

    def compute(self, width, height, left_fit, right_fit, confidence):
        """Compute lateral offset (m), curvature radius (m), steering angle, and ADAS state."""
        if left_fit is None or right_fit is None or confidence < 0.25:
            return {
                "state": "STANDBY",
                "offset_m": None,
                "curvature_m": None,
                "steering_deg": 0.0,
                "command": "Lane Lost",
            }

        y_eval = float(height - 1)
        left_x = float(np.polyval(left_fit, y_eval))
        right_x = float(np.polyval(right_fit, y_eval))
        lane_width_px = right_x - left_x

        if lane_width_px <= 10:
            return {
                "state": "STANDBY",
                "offset_m": None,
                "curvature_m": None,
                "steering_deg": 0.0,
                "command": "Lane Lost",
            }

        # Self-calibrate meters-per-pixel horizontally using nominal lane width
        xm_per_pix = np.clip(
            self.lane_width_m / lane_width_px,
            self.default_xm_per_pix * 0.5,
            self.default_xm_per_pix * 1.8,
        )

        # Positive offset_m => ego vehicle is to the RIGHT of lane center
        lane_center_px = 0.5 * (left_x + right_x)
        vehicle_center_px = 0.5 * width
        offset_m = (vehicle_center_px - lane_center_px) * xm_per_pix

        # Curvature radius in meters for left and right lanes
        left_r = self._curvature_radius_m(left_fit, y_eval, xm_per_pix)
        right_r = self._curvature_radius_m(right_fit, y_eval, xm_per_pix)
        curvature_m = 0.5 * (left_r + right_r)

        # Heading angle of lane ahead relative to ego axis (rad)
        # In image coords y grows downward, so dx/dy < 0 means lane heads right.
        left_dx_dy = 2.0 * left_fit[0] * y_eval + left_fit[1]
        right_dx_dy = 2.0 * right_fit[0] * y_eval + right_fit[1]
        avg_dx_dy = 0.5 * (left_dx_dy + right_dx_dy)
        lane_heading_rad = -np.arctan((avg_dx_dy * xm_per_pix) / self.ym_per_pix)

        # Proportional lateral + heading controller (positive = steer right)
        raw_steer_deg = (
            -self.kp_lateral * offset_m + self.kp_heading * lane_heading_rad
        )
        steering_deg = float(
            np.clip(raw_steer_deg, -self.max_steer_deg, self.max_steer_deg)
        )

        if steering_deg > self.straight_deadband_deg:
            command = "Steer Right"
        elif steering_deg < -self.straight_deadband_deg:
            command = "Steer Left"
        else:
            command = "Go Straight"

        # Lane Departure Warning state machine
        if offset_m > self.ldw_threshold_m:
            state = "LDW RIGHT"
        elif offset_m < -self.ldw_threshold_m:
            state = "LDW LEFT"
        else:
            state = "LKA ACTIVE"

        return {
            "state": state,
            "offset_m": float(offset_m),
            "curvature_m": float(curvature_m),
            "steering_deg": steering_deg,
            "command": command,
        }

    def _curvature_radius_m(self, fit, y_eval, xm_per_pix):
        a_m = fit[0] * xm_per_pix / (self.ym_per_pix ** 2)
        b_m = fit[1] * xm_per_pix / self.ym_per_pix
        y_m = y_eval * self.ym_per_pix
        denom = max(abs(2.0 * a_m), 1e-6)
        return float(((1.0 + (2.0 * a_m * y_m + b_m) ** 2) ** 1.5) / denom)
