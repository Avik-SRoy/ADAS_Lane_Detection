import numpy as np


def compute_iou(box_a, box_b) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0.0 else 0.0


class ObjectTracker:
    """Lightweight IoU-based multi-object tracker with smoothed boxes & velocities."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 10, box_alpha: float = 0.65):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.box_alpha = float(box_alpha)
        self._next_id = 1
        self._tracks = {}

    def update(self, detections):
        """Match detections to active tracks and return currently visible tracked objects."""
        track_ids = list(self._tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        if track_ids and detections:
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                t_box = self._tracks[tid]["bbox"]
                for di, det in enumerate(detections):
                    if self._tracks[tid]["class_name"] == det["class_name"]:
                        iou_matrix[ti, di] = compute_iou(t_box, det["bbox"])

            # Greedy highest-IoU matching
            while True:
                idx = int(np.argmax(iou_matrix))
                ti, di = divmod(idx, len(detections))
                best_iou = float(iou_matrix[ti, di])
                if best_iou < self.iou_threshold:
                    break
                tid = track_ids[ti]
                det = detections[di]

                tr = self._tracks[tid]
                old_box = np.array(tr["bbox"], dtype=np.float64)
                new_box = np.array(det["bbox"], dtype=np.float64)
                smoothed_box = self.box_alpha * new_box + (1.0 - self.box_alpha) * old_box

                tr["bbox"] = smoothed_box.tolist()
                tr["confidence"] = det["confidence"]
                tr["missed"] = 0
                tr["hits"] += 1

                matched_tracks.add(tid)
                matched_dets.add(di)
                iou_matrix[ti, :] = -1.0
                iou_matrix[:, di] = -1.0

        # Age unmatched tracks and prune expired ones
        for tid in list(self._tracks.keys()):
            if tid not in matched_tracks:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_age:
                    del self._tracks[tid]

        # Spawn new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "track_id": tid,
                "bbox": list(det["bbox"]),
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "missed": 0,
                "hits": 1,
                "distance_m": None,
                "closing_speed_mps": 0.0,
            }

        # Return only tracks seen in the current frame (or very recently coasted)
        return [
            tr for tr in self._tracks.values()
            if tr["missed"] == 0
        ]
