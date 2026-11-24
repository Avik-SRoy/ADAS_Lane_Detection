from pathlib import Path


ROAD_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class ObjectDetector:
    """Real-time road object detector powered by Ultralytics YOLOv8."""

    def __init__(self, cfg, base_dir: Path | None = None):
        from ultralytics import YOLO

        model_path = Path(cfg["model_path"])
        if not model_path.is_absolute() and base_dir is not None:
            model_path = base_dir / model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(model_path))
        self.conf_threshold = float(cfg.get("conf_threshold", 0.35))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.45))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.target_classes = list(cfg.get("target_classes", [0, 1, 2, 3, 5, 7]))

    def detect(self, frame):
        """Run YOLOv8 inference and return road-relevant bounding boxes.

        Returns:
            list[dict]: Each entry has keys:
                bbox: [x1, y1, x2, y2] (float)
                class_id: int
                class_name: str
                confidence: float
        """
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = results[0].names

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            cls_name = ROAD_CLASS_NAMES.get(int(cls_id), str(names.get(int(cls_id), cls_id)))
            detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "class_id": int(cls_id),
                "class_name": cls_name,
                "confidence": float(conf),
            })
        return detections
