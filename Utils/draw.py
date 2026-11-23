import cv2
import numpy as np


def make_coordinates(frame, line_params):
    """Legacy straight-line coordinate helper retained for compatibility."""
    slope, intercept = line_params
    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def draw_lane_lines(frame, left_lane, right_lane):
    """Legacy straight-line renderer retained for compatibility."""
    line_image = np.zeros_like(frame)
    if left_lane is not None:
        left_line = make_coordinates(frame, left_lane)
        cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 5)
    if right_lane is not None:
        right_line = make_coordinates(frame, right_lane)
        cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 5)
    return line_image


def fill_lane_area(frame, left_lane, right_lane):
    """Legacy straight-line fill helper retained for compatibility."""
    if left_lane is None or right_lane is None:
        return frame
    height = frame.shape[0]
    y_vals = np.linspace(int(height * 0.6), height, 50)
    left_x = (y_vals - left_lane[1]) / left_lane[0]
    right_x = (y_vals - right_lane[1]) / right_lane[0]
    left_points = np.array([np.transpose(np.vstack([left_x, y_vals]))], dtype=np.int32)
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_x, y_vals])))], dtype=np.int32)
    points = np.hstack((left_points, right_points))
    overlay = frame.copy()
    cv2.fillPoly(overlay, points, (0, 255, 0))
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


def render_lane_overlay(frame, transformer, left_fit, right_fit, adas_state="LKA ACTIVE"):
    """Draw curved lane polygon, boundaries, and centerline in BEV and unwarp onto frame."""
    height, width = frame.shape[:2]
    bev_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    if left_fit is None or right_fit is None:
        return frame.copy(), bev_overlay

    ploty = np.linspace(0, height - 1, height)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts_poly = np.hstack((pts_left, pts_right))

    # Color corridor green normally, amber/red on Lane Departure Warning
    if adas_state.startswith("LDW"):
        fill_color = (0, 140, 255)
    else:
        fill_color = (0, 220, 80)

    cv2.fillPoly(bev_overlay, pts_poly, fill_color)

    # Draw crisp lane boundary curves in BEV
    cv2.polylines(bev_overlay, pts_left, False, (255, 120, 0), 10)
    cv2.polylines(bev_overlay, pts_right, False, (0, 80, 255), 10)

    unwarped = transformer.unwarp(bev_overlay)
    blended = cv2.addWeighted(frame, 1.0, unwarped, 0.38, 0.0)
    return blended, bev_overlay


def render_hud(frame, control_out, confidence, fps, bev_binary=None, bev_overlay=None):
    """Render automotive-style telemetry HUD and optional mini Bird's-Eye View inset."""
    out = frame.copy()
    height, width = out.shape[:2]

    # Semi-transparent HUD panel (top-left)
    panel_w, panel_h = 330, 155
    overlay = out.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (20, 22, 26), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    cv2.rectangle(out, (12, 12), (12 + panel_w, 12 + panel_h), (90, 95, 105), 1)

    state = control_out["state"]
    if state == "LKA ACTIVE":
        badge_color = (40, 200, 80)
    elif state.startswith("LDW"):
        badge_color = (0, 140, 255)
    else:
        badge_color = (120, 120, 130)

    cv2.rectangle(out, (22, 22), (165, 48), badge_color, -1)
    cv2.putText(out, state, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)

    fps_text = f"FPS: {fps:4.1f}"
    cv2.putText(out, fps_text, (235, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 225, 230), 1, cv2.LINE_AA)

    offset_m = control_out["offset_m"]
    curv_m = control_out["curvature_m"]
    steer_deg = control_out["steering_deg"]
    cmd = control_out["command"]

    offset_str = f"{offset_m:+.2f} m" if offset_m is not None else "N/A"
    if curv_m is None:
        curv_str = "N/A"
    elif curv_m > 2500:
        curv_str = "Straight"
    else:
        curv_str = f"{curv_m:.0f} m"

    cv2.putText(out, f"Lateral Offset : {offset_str}", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, f"Curve Radius   : {curv_str}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, f"Steering       : {cmd} ({steer_deg:+.1f} deg)", (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    # Confidence bar
    cv2.putText(out, "Confidence:", (24, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 205, 210), 1, cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 125, 136, 195, 14
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 70), -1)
    fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
    bar_col = (50, 210, 90) if confidence >= 0.6 else (0, 180, 255)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (130, 135, 145), 1)

    # Picture-in-picture BEV debug inset (top-right)
    if bev_binary is not None and bev_overlay is not None:
        inset_w, inset_h = 220, 155
        bev_rgb = cv2.cvtColor(bev_binary, cv2.COLOR_GRAY2BGR)
        bev_vis = cv2.addWeighted(bev_rgb, 0.65, bev_overlay, 0.75, 0)
        bev_small = cv2.resize(bev_vis, (inset_w, inset_h))

        x0 = width - inset_w - 12
        y0 = 12
        out[y0 : y0 + inset_h, x0 : x0 + inset_w] = bev_small
        cv2.rectangle(out, (x0, y0), (x0 + inset_w, y0 + inset_h), (90, 95, 105), 1)
        cv2.putText(out, "Bird's-Eye View", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

    return out
