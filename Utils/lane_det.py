import cv2
import numpy as np


def detect_lines(image):
    """Legacy Hough line detector retained for backward compatibility."""
    return cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150,
    )


def average_slope_intercept(frame, lines):
    """Legacy straight-line averaging retained for backward compatibility."""
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_lane = np.mean(left_lines, axis=0) if left_lines else None
    right_lane = np.mean(right_lines, axis=0) if right_lines else None
    return left_lane, right_lane


class PolynomialLaneDetector:
    """Sliding-window & search-around-poly 2nd-order lane detector in BEV space."""

    def __init__(self, cfg):
        self.n_windows = cfg["n_windows"]
        self.margin = cfg["margin"]
        self.minpix = cfg["minpix"]
        self.min_total_pixels = cfg["min_total_pixels"]
        self.min_lane_width_px = cfg["min_lane_width_px"]
        self.max_lane_width_px = cfg["max_lane_width_px"]
        self.max_slope_diff = cfg["max_slope_diff"]

    def detect(self, binary_warped, prev_left=None, prev_right=None):
        """Detect left and right 2nd-order lane polynomials x = a*y^2 + b*y + c.

        Returns:
            dict with keys:
                left_fit: np.ndarray(3,) or None
                right_fit: np.ndarray(3,) or None
                left_conf: float in [0, 1]
                right_conf: float in [0, 1]
                method: 'poly_search' or 'sliding_window'
        """
        height, width = binary_warped.shape[:2]
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if len(nonzeroy) == 0:
            return {
                "left_fit": None,
                "right_fit": None,
                "left_conf": 0.0,
                "right_conf": 0.0,
                "method": "none",
            }

        # Try fast search around previous polynomials first
        if prev_left is not None and prev_right is not None:
            left_inds, right_inds = self._search_around_poly(
                nonzerox, nonzeroy, prev_left, prev_right
            )
            result = self._fit_and_validate(
                nonzerox, nonzeroy, left_inds, right_inds, height, "poly_search"
            )
            if result["left_fit"] is not None and result["right_fit"] is not None:
                return result

        # Fallback to full sliding-window histogram search
        left_inds, right_inds = self._sliding_window_search(
            binary_warped, nonzerox, nonzeroy
        )
        return self._fit_and_validate(
            nonzerox, nonzeroy, left_inds, right_inds, height, "sliding_window"
        )

    def _sliding_window_search(self, binary_warped, nonzerox, nonzeroy):
        height, width = binary_warped.shape[:2]
        histogram = np.sum(binary_warped[height // 2 :, :], axis=0)
        midpoint = width // 2
        left_bound = int(width * 0.10)
        right_bound = int(width * 0.90)

        leftx_base = int(np.argmax(histogram[left_bound:midpoint])) + left_bound
        rightx_base = int(np.argmax(histogram[midpoint:right_bound])) + midpoint

        window_height = height // self.n_windows
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.n_windows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        return np.concatenate(left_lane_inds), np.concatenate(right_lane_inds)

    def _search_around_poly(self, nonzerox, nonzeroy, prev_left, prev_right):
        left_center = (
            prev_left[0] * (nonzeroy ** 2) + prev_left[1] * nonzeroy + prev_left[2]
        )
        right_center = (
            prev_right[0] * (nonzeroy ** 2) + prev_right[1] * nonzeroy + prev_right[2]
        )

        left_inds = (
            (nonzerox > left_center - self.margin)
            & (nonzerox < left_center + self.margin)
        ).nonzero()[0]
        right_inds = (
            (nonzerox > right_center - self.margin)
            & (nonzerox < right_center + self.margin)
        ).nonzero()[0]
        return left_inds, right_inds

    def _fit_and_validate(self, nonzerox, nonzeroy, left_inds, right_inds, height, method):
        left_fit = None
        right_fit = None
        left_conf = 0.0
        right_conf = 0.0

        if len(left_inds) >= self.min_total_pixels:
            leftx = nonzerox[left_inds]
            lefty = nonzeroy[left_inds]
            if np.ptp(lefty) > height * 0.25:
                left_fit = np.polyfit(lefty, leftx, 2)
                left_conf = min(1.0, len(left_inds) / 2500.0)

        if len(right_inds) >= self.min_total_pixels:
            rightx = nonzerox[right_inds]
            righty = nonzeroy[right_inds]
            if np.ptp(righty) > height * 0.25:
                right_fit = np.polyfit(righty, rightx, 2)
                right_conf = min(1.0, len(right_inds) / 2500.0)

        # Geometric sanity check when both lanes are present
        if left_fit is not None and right_fit is not None:
            y_samples = np.array([height * 0.25, height * 0.6, height - 1.0])
            left_xs = np.polyval(left_fit, y_samples)
            right_xs = np.polyval(right_fit, y_samples)
            widths = right_xs - left_xs

            # Check lane width bounds and approximate parallelism
            left_slope = 2 * left_fit[0] * (height * 0.6) + left_fit[1]
            right_slope = 2 * right_fit[0] * (height * 0.6) + right_fit[1]
            slope_diff = abs(left_slope - right_slope)

            width_ok = np.all(
                (widths >= self.min_lane_width_px) & (widths <= self.max_lane_width_px)
            )
            if not width_ok or slope_diff > self.max_slope_diff:
                # Keep the higher-confidence lane and reject the weaker outlier
                if left_conf >= right_conf:
                    right_fit = None
                    right_conf = 0.0
                else:
                    left_fit = None
                    left_conf = 0.0

        return {
            "left_fit": left_fit,
            "right_fit": right_fit,
            "left_conf": float(left_conf),
            "right_conf": float(right_conf),
            "method": method,
        }
