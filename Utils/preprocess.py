import cv2
import numpy as np


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def gaussian_blur(frame, kernel_size=5):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def canny_edge(frame, low_threshold=50, high_threshold=150):
    return cv2.Canny(frame, low_threshold, high_threshold)


def extract_lane_binary(frame, cfg):
    """Create a binary mask isolating white and yellow/orange lane markings.

    Combines HLS white-lane thresholding, HSV yellow/orange thresholding, and
    Sobel-X gradient magnitude on the lightness channel for illumination
    robustness across shadows and wet roads.
    """
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # 1. White lane markings (high lightness, low saturation)
    white_mask = (
        (l_channel >= cfg["white_hls_l_min"]) &
        (s_channel <= cfg["white_hls_s_max"])
    )

    # 2. Yellow / orange lane markings (HSV range)
    lower_yellow = np.array([
        cfg["yellow_hsv_h_min"],
        cfg["yellow_hsv_s_min"],
        cfg["yellow_hsv_v_min"],
    ], dtype=np.uint8)
    upper_yellow = np.array([
        cfg["yellow_hsv_h_max"],
        255,
        255,
    ], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow) > 0

    # 3. Directional Sobel-X gradient on blurred lightness channel
    blurred_l = cv2.GaussianBlur(l_channel, (5, 5), 0)
    sobelx = cv2.Sobel(blurred_l, cv2.CV_64F, 1, 0, ksize=cfg["sobel_kernel"])
    abs_sobelx = np.abs(sobelx)
    max_val = np.max(abs_sobelx)
    if max_val > 0:
        scaled_sobel = np.uint8(255 * abs_sobelx / max_val)
    else:
        scaled_sobel = np.zeros_like(l_channel)

    grad_mask = (
        (scaled_sobel >= cfg["sobel_x_min"]) &
        (scaled_sobel <= cfg["sobel_x_max"]) &
        (l_channel >= 120)
    )

    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    combined[white_mask | yellow_mask | grad_mask] = 255

    # Remove isolated speckle noise while keeping thin lane lines intact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined
