import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import config
from modules.colors import get_color_for_label

# Primary segregation: Metal vs Other waste (before it reaches plastic-type conveyor)


class WastePrimaryProcessor:
    """Conveyor 1: Primary Segregation — Metal vs Other Waste.
    Model is OPTIONAL. If 'models/waste_primary.pt' isn't present yet,
    this module shows a clear on-screen notice instead of crashing, so the
    rest of the dashboard keeps working while this model is being trained.
    """

    def __init__(self):
        self.model = None
        print(f"[WASTE-PRIMARY] Looking for model at: {config.WASTE_PRIMARY_MODEL}")
        if os.path.exists(config.WASTE_PRIMARY_MODEL):
            self.model = YOLO(config.WASTE_PRIMARY_MODEL)
            print(f"[WASTE-PRIMARY] Model LOADED OK. Classes: {self.model.names}")
            self._warmup()
        else:
            print("[WASTE-PRIMARY] Model not found — module will show placeholder until trained model is added")

        self.frame_count = 0
        self.session_counts = {}
        self.last_seen = {}
        self.log_path = os.path.join(config.CAPTURES_DIR, "waste_primary_log.csv")
        self._ensure_log()

    def _warmup(self):
        """Dummy inference at load time so the first real frame doesn't stutter
        (CUDA context init / cuDNN kernel selection happens here, not on-camera)."""
        dummy = np.zeros((config.INFER_IMGSZ, config.INFER_IMGSZ, 3), dtype=np.uint8)
        with torch.inference_mode():
            self.model(dummy, device=config.DEVICE, half=config.USE_HALF_PRECISION,
                        imgsz=config.INFER_IMGSZ, verbose=False)
        print("[WASTE-PRIMARY] Model warm-up done.")

    def _ensure_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Class", "Confidence"])

    def _log_detection(self, label, conf):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, round(conf, 2)])

    def status(self):
        return {"model_loaded": self.model is not None}

    def process(self, frame):
        if self.model is None:
            cv2.putText(frame, "Primary segregation model not trained yet", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
            cv2.putText(frame, "Add models/waste_primary.pt to activate", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            return frame

        # NOTE: frame-skip ab centrally app.py ke capture worker me hoti hai
        self.frame_count += 1

        with torch.inference_mode():
            results = self.model(frame, conf=config.WASTE_PRIMARY_CONF, verbose=False,
                                  device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                  imgsz=config.INFER_IMGSZ)
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = get_color_for_label(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                key = label
                now = time.time()
                if now - self.last_seen.get(key, 0) > 1.5:
                    self.session_counts[key] = self.session_counts.get(key, 0) + 1
                    self._log_detection(label, conf)
                self.last_seen[key] = now
        return frame

    def get_counts(self):
        return dict(self.session_counts)

    def reset_counts(self):
        self.session_counts = {}
        self.last_seen = {}
