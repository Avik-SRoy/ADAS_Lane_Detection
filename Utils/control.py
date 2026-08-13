import numpy as np

def compute_lane_center(frame, left_lane, right_lane):
    if left_lane is None or right_lane is None:
        return None

    height = frame.shape[0]
    y = height

    # Convert line equations to x positions at bottom
    left_x = int((y - left_lane[1]) / left_lane[0])
    right_x = int((y - right_lane[1]) / right_lane[0])

    lane_center = (left_x + right_x) // 2
    frame_center = frame.shape[1] // 2

    return lane_center, frame_center


def get_steering_direction(lane_center, frame_center, threshold=50):
    deviation = lane_center - frame_center

    if deviation > threshold:
        return "Turn Right"
    elif deviation < -threshold:
        return "Turn Left"
    else:
        return "Go Straight"