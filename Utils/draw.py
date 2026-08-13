import cv2
import numpy as np

def make_coordinates(frame, line_params):
    slope, intercept = line_params
    height = frame.shape[0]

    y1 = height
    y2 = int(height * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def draw_lane_lines(frame, left_lane, right_lane):
    line_image = np.zeros_like(frame)

    if left_lane is not None:
        left_line = make_coordinates(frame, left_lane)
        cv2.line(line_image,
                 (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]),
                 (255, 0, 0), 5)

    if right_lane is not None:
        right_line = make_coordinates(frame, right_lane)
        cv2.line(line_image,
                 (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]),
                 (0, 0, 255), 5)

    return line_image


def fill_lane_area(frame, left_lane, right_lane):
    if left_lane is None or right_lane is None:
        return frame

    height = frame.shape[0]

    # Generate y values
    y_vals = np.linspace(int(height*0.6), height, 50)

    # Left lane curve
    left_x = (y_vals - left_lane[1]) / left_lane[0]

    # Right lane curve
    right_x = (y_vals - right_lane[1]) / right_lane[0]

    # Convert to int
    left_points = np.array([np.transpose(np.vstack([left_x, y_vals]))], dtype=np.int32)
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_x, y_vals])))], dtype=np.int32)

    # Combine into one polygon
    points = np.hstack((left_points, right_points))

    # Create overlay
    overlay = frame.copy()

    # Fill polygon
    cv2.fillPoly(overlay, points, (0, 255, 0))

    # Blend with original
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)