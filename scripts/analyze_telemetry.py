import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent


def load_telemetry(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame"]),
                "method": row["method"],
                "state": row["state"],
                "confidence": float(row["confidence"]),
                "offset_m": float(row["offset_m"]) if row["offset_m"] else np.nan,
                "curvature_m": float(row["curvature_m"]) if row["curvature_m"] else np.nan,
                "steering_deg": float(row["steering_deg"]),
                "command": row["command"],
                "object_count": int(row["object_count"]) if row.get("object_count") else 0,
                "fcw_state": row.get("fcw_state", "DISABLED"),
                "lead_distance_m": float(row["lead_distance_m"]) if row.get("lead_distance_m") else np.nan,
                "lead_ttc_s": float(row["lead_ttc_s"]) if row.get("lead_ttc_s") else np.nan,
                "latency_ms": float(row["latency_ms"]),
            })
    return rows


def _draw_series_panel(canvas, x0, y0, w, h, title, values, y_min, y_max, color, ref_lines=None, unit=""):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (32, 35, 42), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 86, 98), 1)
    cv2.putText(canvas, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 240, 245), 1, cv2.LINE_AA)

    pad_l, pad_r, pad_t, pad_b = 58, 18, 36, 22
    px0, py0 = x0 + pad_l, y0 + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), (22, 24, 29), -1)

    def val_to_y(v):
        t = (v - y_min) / max(y_max - y_min, 1e-6)
        t = np.clip(t, 0.0, 1.0)
        return int(py0 + ph - t * ph)

    # Horizontal grid & labels
    for frac in (0.0, 0.5, 1.0):
        v = y_min + frac * (y_max - y_min)
        gy = val_to_y(v)
        cv2.line(canvas, (px0, gy), (px0 + pw, gy), (48, 52, 60), 1)
        cv2.putText(canvas, f"{v:.1f}{unit}", (x0 + 6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 172, 182), 1, cv2.LINE_AA)

    if ref_lines:
        for ref_val, ref_col in ref_lines:
            ry = val_to_y(ref_val)
            cv2.line(canvas, (px0, ry), (px0 + pw, ry), ref_col, 1, cv2.LINE_AA)

    n = len(values)
    if n < 2:
        return

    pts = []
    for i, v in enumerate(values):
        if np.isnan(v):
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            pts = []
            continue
        px = int(px0 + (i / (n - 1)) * pw)
        py = val_to_y(v)
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def render_telemetry_chart(rows, output_path: Path):
    w, h = 1100, 740
    canvas = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)

    cv2.putText(
        canvas,
        f"ADAS Telemetry Report ({len(rows)} frames)",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (240, 244, 248),
        2,
        cv2.LINE_AA,
    )

    offsets = [r["offset_m"] for r in rows]
    steers = [r["steering_deg"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]

    _draw_series_panel(
        canvas, 20, 52, w - 40, 205,
        "Lateral Offset (m)  [dashed amber = +/-0.45 m LDW threshold]",
        offsets, -0.8, 0.8, (80, 220, 120),
        ref_lines=[(0.0, (100, 105, 115)), (0.45, (0, 140, 255)), (-0.45, (0, 140, 255))],
    )
    _draw_series_panel(
        canvas, 20, 274, w - 40, 205,
        "Steering Command (deg)  [+ Right / - Left]",
        steers, -20.0, 20.0, (255, 185, 65),
        ref_lines=[(0.0, (100, 105, 115))],
    )
    max_lat = max(50.0, float(np.nanpercentile(latencies, 99)) * 1.15)
    _draw_series_panel(
        canvas, 20, 496, w - 40, 205,
        "Per-Frame Processing Latency (ms)  [red = 40 ms / 25 FPS budget]",
        latencies, 0.0, max_lat, (230, 120, 255),
        ref_lines=[(40.0, (70, 70, 230))],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Analyze ADAS CSV telemetry and plot diagnostics.")
    parser.add_argument("--csv", type=str, default=str(BASE_DIR / "Output" / "telemetry.csv"))
    parser.add_argument("--plot", type=str, default=str(BASE_DIR / "Output" / "telemetry_report.png"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    plot_path = Path(args.plot)
    if not plot_path.is_absolute():
        plot_path = BASE_DIR / plot_path

    rows = load_telemetry(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    states = Counter(r["state"] for r in rows)
    methods = Counter(r["method"] for r in rows)
    offsets = np.array([r["offset_m"] for r in rows], dtype=np.float64)
    steers = np.array([r["steering_deg"] for r in rows], dtype=np.float64)
    curvs = np.array([r["curvature_m"] for r in rows], dtype=np.float64)
    lats = np.array([r["latency_ms"] for r in rows], dtype=np.float64)

    tracked = sum(v for k, v in states.items() if k != "STANDBY")
    print("=== ADAS Telemetry Summary ===")
    print(f"Frames         : {len(rows)} (Tracked: {100.0 * tracked / len(rows):.1f}%)")
    print(f"States         : {dict(states)}")
    print(f"Detect Methods : {dict(methods)}")
    print(f"Latency (ms)   : mean={np.mean(lats):.1f}, p95={np.percentile(lats, 95):.1f}, max={np.max(lats):.1f} ({1000.0 / np.mean(lats):.1f} avg FPS)")
    print(f"Lateral Offset : mean={np.nanmean(offsets):+.3f} m, std={np.nanstd(offsets):.3f} m, max|off|={np.nanmax(np.abs(offsets)):.3f} m")
    print(f"Steering (deg) : mean={np.mean(steers):+.2f}, max|steer|={np.max(np.abs(steers)):.2f}")
    print(f"Curvature (m)  : min={np.nanmin(curvs):.0f} m, median={np.nanmedian(curvs):.0f} m")

    fcw_states = Counter(r["fcw_state"] for r in rows)
    if set(fcw_states.keys()) != {"DISABLED"}:
        lead_dists = np.array([r["lead_distance_m"] for r in rows], dtype=np.float64)
        obj_counts = np.array([r["object_count"] for r in rows], dtype=np.float64)
        valid_dists = lead_dists[~np.isnan(lead_dists)]
        dist_str = f"min={np.min(valid_dists):.1f} m, median={np.median(valid_dists):.1f} m" if len(valid_dists) else "no ego lead"
        print(f"FCW States     : {dict(fcw_states)} | avg_objs={np.mean(obj_counts):.2f} | Lead dist: {dist_str}")

    render_telemetry_chart(rows, plot_path)
    print(f"Saved chart to : {plot_path}")


if __name__ == "__main__":
    main()
