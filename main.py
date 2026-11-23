import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2

from Utils.control import LateralController
from Utils.draw import render_hud, render_lane_overlay
from Utils.lane_det import PolynomialLaneDetector
from Utils.preprocess import extract_lane_binary
from Utils.roi import PerspectiveTransformer
from Utils.smoothing import KalmanLaneTracker


BASE_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADAS Phase 1 Lane Perception & Lateral Control Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "config.json"),
        help="Path to JSON configuration file",
    )
    parser.add_argument("--input", type=str, default=None, help="Override input video path")
    parser.add_argument("--output", type=str, default=None, help="Override output MP4 path")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video to configured/specified output path",
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default=None,
        help="Optional path to write per-frame CSV telemetry",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without OpenCV GUI windows",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    video_cfg = cfg["video"]
    input_path = resolve_path(args.input or video_cfg["input_path"])
    output_path = resolve_path(args.output or video_cfg["output_path"])
    width = int(video_cfg["width"])
    height = int(video_cfg["height"])

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.save_video or args.output is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))

    telemetry_file = None
    telemetry_writer = None
    if args.telemetry:
        tel_path = resolve_path(args.telemetry)
        tel_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_file = open(tel_path, "w", newline="", encoding="utf-8")
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow([
            "frame", "method", "state", "confidence",
            "offset_m", "curvature_m", "steering_deg", "command", "latency_ms",
        ])

    transformer = PerspectiveTransformer(
        width,
        height,
        cfg["perspective"]["src_points_norm"],
        cfg["perspective"]["dst_points_norm"],
    )
    detector = PolynomialLaneDetector(cfg["detection"])
    tracker = KalmanLaneTracker(cfg["tracker"])
    controller = LateralController(cfg["metrics"], cfg["control"])

    frame_idx = 0
    active_frames = 0
    ldw_frames = 0
    latencies_ms = []
    smoothed_fps = source_fps

    prev_left = None
    prev_right = None

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))

            # 1. Color (HLS white + HSV yellow/orange) & Sobel-X binary mask
            binary = extract_lane_binary(frame, cfg["preprocessing"])

            # 2. Inverse Perspective Mapping to Bird's-Eye View
            bev_binary = transformer.warp(binary)

            # 3. 2nd-order polynomial lane detection
            det = detector.detect(bev_binary, prev_left=prev_left, prev_right=prev_right)

            # 4. Kalman filter tracking & occlusion handling
            tracked = tracker.update(det)
            prev_left = tracked["left_fit"]
            prev_right = tracked["right_fit"]

            # 5. Metric geometry, LDW state & lateral steering control
            ctrl = controller.compute(
                width, height, prev_left, prev_right, tracked["confidence"]
            )

            # 6. Render curved corridor + ADAS HUD + mini BEV inset
            lane_frame, bev_overlay = render_lane_overlay(
                frame, transformer, prev_left, prev_right, ctrl["state"]
            )

            dt = max(time.perf_counter() - t0, 1e-6)
            latency_ms = dt * 1000.0
            latencies_ms.append(latency_ms)
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / dt)

            hud_frame = render_hud(
                lane_frame,
                ctrl,
                tracked["confidence"],
                smoothed_fps,
                bev_binary=bev_binary,
                bev_overlay=bev_overlay,
            )

            if ctrl["state"] == "LKA ACTIVE":
                active_frames += 1
            elif ctrl["state"].startswith("LDW"):
                ldw_frames += 1

            if writer is not None:
                writer.write(hud_frame)

            if telemetry_writer is not None:
                telemetry_writer.writerow([
                    frame_idx,
                    det["method"],
                    ctrl["state"],
                    f"{tracked['confidence']:.3f}",
                    "" if ctrl["offset_m"] is None else f"{ctrl['offset_m']:.3f}",
                    "" if ctrl["curvature_m"] is None else f"{ctrl['curvature_m']:.1f}",
                    f"{ctrl['steering_deg']:.2f}",
                    ctrl["command"],
                    f"{latency_ms:.2f}",
                ])

            if not args.no_display:
                cv2.imshow("ADAS Lane Perception & Control", hud_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if telemetry_file is not None:
            telemetry_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if frame_idx > 0:
        avg_ms = sum(latencies_ms) / len(latencies_ms)
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        track_rate = 100.0 * (active_frames + ldw_frames) / frame_idx
        print(
            f"Processed {frame_idx} frames | Tracked: {track_rate:.1f}% "
            f"(LKA: {active_frames}, LDW: {ldw_frames}) | "
            f"Avg latency: {avg_ms:.1f} ms ({avg_fps:.1f} FPS)"
        )
        if args.save_video or args.output is not None:
            print(f"Saved output video to: {output_path}")


if __name__ == "__main__":
    main()
