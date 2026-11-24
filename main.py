import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from Utils.control import LateralController
from Utils.draw import render_hud, render_lane_overlay, render_objects
from Utils.fcw import ForwardCollisionWarning
from Utils.lane_det import PolynomialLaneDetector
from Utils.object_det import ObjectDetector
from Utils.object_tracker import ObjectTracker
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
        description="ADAS Lane Perception, YOLOv8 Object Detection & FCW Pipeline"
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
        "--disable-objects",
        action="store_true",
        help="Disable YOLOv8 object detection and FCW (lane-only mode)",
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
    frame_dt = 1.0 / source_fps

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
            "offset_m", "curvature_m", "steering_deg", "command",
            "object_count", "fcw_state", "lead_distance_m", "lead_ttc_s",
            "latency_ms",
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

    obj_cfg = cfg.get("objects", {})
    objects_enabled = (not args.disable_objects) and bool(obj_cfg.get("enabled", True))
    obj_detector = None
    obj_tracker = None
    fcw = None
    detect_every_n = max(1, int(obj_cfg.get("detect_every_n_frames", 1)))

    if objects_enabled:
        obj_detector = ObjectDetector(obj_cfg, base_dir=BASE_DIR)
        obj_tracker = ObjectTracker(
            iou_threshold=obj_cfg.get("tracker_iou_threshold", 0.25),
            max_age=obj_cfg.get("tracker_max_age", 10),
        )
        fcw = ForwardCollisionWarning(cfg["fcw"])

    frame_idx = 0
    active_frames = 0
    ldw_frames = 0
    fcw_warn_frames = 0
    fcw_caution_frames = 0
    total_objects_seen = 0
    latencies_ms = []
    smoothed_fps = source_fps

    prev_left = None
    prev_right = None
    last_raw_dets = []

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

            # 6. YOLOv8 object detection, tracking & Forward Collision Warning
            fcw_out = None
            if objects_enabled:
                if frame_idx % detect_every_n == 0:
                    last_raw_dets = obj_detector.detect(frame)
                tracked_objs = obj_tracker.update(last_raw_dets)
                fcw_out = fcw.evaluate(
                    tracked_objs, transformer, prev_left, prev_right, width, height, frame_dt
                )
                total_objects_seen += len(tracked_objs)
                if fcw_out["state"] == "FCW WARNING":
                    fcw_warn_frames += 1
                elif fcw_out["state"] == "CAUTION":
                    fcw_caution_frames += 1

            # 7. Render curved corridor + object boxes + ADAS HUD + mini BEV inset
            lane_frame, bev_overlay = render_lane_overlay(
                frame, transformer, prev_left, prev_right, ctrl["state"]
            )
            if fcw_out is not None:
                lane_frame = render_objects(lane_frame, fcw_out)

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
                fcw_out=fcw_out,
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
                    0 if fcw_out is None else len(fcw_out["objects"]),
                    "DISABLED" if fcw_out is None else fcw_out["state"],
                    "" if (fcw_out is None or fcw_out["lead_distance_m"] is None) else f"{fcw_out['lead_distance_m']:.2f}",
                    "" if (fcw_out is None or fcw_out["lead_ttc_s"] is None) else f"{fcw_out['lead_ttc_s']:.2f}",
                    f"{latency_ms:.2f}",
                ])

            if not args.no_display:
                cv2.imshow("ADAS Lane Perception & FCW", hud_frame)
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
        fcw_summary = (
            f" | FCW(warn={fcw_warn_frames}, caution={fcw_caution_frames}, avg_objs={total_objects_seen / frame_idx:.2f})"
            if objects_enabled
            else " | Objects: disabled"
        )
        print(
            f"Processed {frame_idx} frames | Tracked: {track_rate:.1f}% "
            f"(LKA: {active_frames}, LDW: {ldw_frames}){fcw_summary} | "
            f"Avg latency: {avg_ms:.1f} ms ({avg_fps:.1f} FPS)"
        )
        if args.save_video or args.output is not None:
            print(f"Saved output video to: {output_path}")


if __name__ == "__main__":
    main()
