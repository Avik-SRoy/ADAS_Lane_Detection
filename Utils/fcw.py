import numpy as np


class ForwardCollisionWarning:
    """Ego-lane spatial filtering, monocular distance estimation, TTC & FCW FSM."""

    def __init__(self, cfg):
        self.focal_length_px = float(cfg["focal_length_px"])
        self.known_heights_m = dict(cfg["known_heights_m"])
        self.ego_lane_margin_px = float(cfg["ego_lane_margin_px"])
        self.caution_distance_m = float(cfg["caution_distance_m"])
        self.warning_distance_m = float(cfg["warning_distance_m"])
        self.caution_ttc_s = float(cfg["caution_ttc_s"])
        self.warning_ttc_s = float(cfg["warning_ttc_s"])
        self.min_closing_speed_mps = float(cfg["min_closing_speed_mps"])

    def estimate_distance_m(self, bbox, class_name: str) -> float:
        """Estimate longitudinal distance using pinhole camera height model."""
        box_h = max(float(bbox[3] - bbox[1]), 1.0)
        real_h = float(self.known_heights_m.get(class_name, 1.5))
        dist = (real_h * self.focal_length_px) / box_h
        return float(np.clip(dist, 1.5, 150.0))

    def is_in_ego_lane(self, bbox, transformer, left_fit, right_fit, width, height) -> bool:
        """Check whether an object's ground contact point lies inside the ego lane."""
        if transformer is None or left_fit is None or right_fit is None:
            return False

        cx = 0.5 * (bbox[0] + bbox[2])
        y2 = float(bbox[3])

        # Project bottom-center ground contact point to Bird's-Eye View
        pt = np.array([cx, y2, 1.0], dtype=np.float64)
        warped = transformer.M @ pt
        if abs(warped[2]) < 1e-6:
            return False

        bx = warped[0] / warped[2]
        by = warped[1] / warped[2]

        # Clamp vertical evaluation to valid BEV range for far objects near horizon
        by_eval = float(np.clip(by, 0.0, height - 1.0))
        if by < -height * 0.35 or by > height * 1.1:
            return False

        left_x = float(np.polyval(left_fit, by_eval))
        right_x = float(np.polyval(right_fit, by_eval))
        if right_x <= left_x:
            return False

        return (left_x - self.ego_lane_margin_px) <= bx <= (right_x + self.ego_lane_margin_px)

    def evaluate(self, tracked_objects, transformer, left_fit, right_fit, width, height, dt: float):
        """Update per-object distance/TTC and compute overall FCW state."""
        dt = max(float(dt), 1e-3)
        lead_obj = None

        for obj in tracked_objects:
            raw_dist = self.estimate_distance_m(obj["bbox"], obj["class_name"])
            prev_dist = obj.get("distance_m")

            if prev_dist is None:
                smooth_dist = raw_dist
                closing_speed = 0.0
            else:
                smooth_dist = 0.35 * raw_dist + 0.65 * prev_dist
                inst_closing = (prev_dist - smooth_dist) / dt
                # Clamp unrealistic single-frame spikes and smooth velocity
                inst_closing = float(np.clip(inst_closing, -30.0, 30.0))
                prev_speed = float(obj.get("closing_speed_mps", 0.0))
                closing_speed = 0.30 * inst_closing + 0.70 * prev_speed

            obj["distance_m"] = float(smooth_dist)
            obj["closing_speed_mps"] = float(closing_speed)

            if closing_speed >= self.min_closing_speed_mps:
                obj["ttc_s"] = float(smooth_dist / closing_speed)
            else:
                obj["ttc_s"] = None

            in_lane = self.is_in_ego_lane(
                obj["bbox"], transformer, left_fit, right_fit, width, height
            )
            obj["in_ego_lane"] = in_lane

            if in_lane:
                if lead_obj is None or obj["distance_m"] < lead_obj["distance_m"]:
                    lead_obj = obj

        # Forward Collision Warning state machine
        state = "CLEAR"
        lead_dist = None
        lead_ttc = None

        if lead_obj is not None:
            lead_dist = lead_obj["distance_m"]
            lead_ttc = lead_obj["ttc_s"]
            if lead_dist <= self.warning_distance_m or (
                lead_ttc is not None and lead_ttc <= self.warning_ttc_s
            ):
                state = "FCW WARNING"
            elif lead_dist <= self.caution_distance_m or (
                lead_ttc is not None and lead_ttc <= self.caution_ttc_s
            ):
                state = "CAUTION"

        return {
            "state": state,
            "lead_object": lead_obj,
            "lead_distance_m": lead_dist,
            "lead_ttc_s": lead_ttc,
            "objects": tracked_objects,
        }
