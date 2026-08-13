import cv2
import numpy as np

def detect_lines(image):
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )
    return lines


def average_slope_intercept(frame, lines):
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

        # Ignore horizontal noise
        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_lane = np.mean(left_lines, axis=0) if left_lines else None
    right_lane = np.mean(right_lines, axis=0) if right_lines else None

    return left_lane, right_lane

