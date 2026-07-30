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


class WasteProcessor:
    """Secondary Plastic Classification pipeline.

    Real detection-gated pipeline (NOT a timer/frame-skip):
        waste_seg2.pt (segregation / plastic-object detector)
            -> for EACH detected plastic object, crop it
            -> final_7types.pt runs ONLY on that crop (RIC type: PET/HDPE/...)

    final_7types.pt is never invoked on a frame where waste_seg2.pt found
    nothing — this is what "MUST run ONLY when plastic detected" means.
    """

    def __init__(self):
        self.model_material = None   # waste_seg2.pt — segregation / gate model
        self.model_type = None       # final_7types.pt — gated RIC-type classifier
        self._load_models()
        self.frame_count = 0
        self.session_counts = {}   # class_name -> count (session-wide, unique-ish per detection frame)
        self.last_seen = {}
        self.log_path = os.path.join(config.CAPTURES_DIR, "waste_secondary_log.csv")
        self._ensure_log()

    def _ensure_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Class", "Confidence"])

    def _log_detection(self, label, conf):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, round(conf, 2)])

    def _load_models(self):
        print(f"[WASTE] Looking for segregation model at: {config.WASTE_MATERIAL_MODEL}")
        if os.path.exists(config.WASTE_MATERIAL_MODEL):
            self.model_material = YOLO(config.WASTE_MATERIAL_MODEL)
            print(f"[WASTE] Segregation model (waste_seg2) LOADED OK. Classes: {self.model_material.names}")
        else:
            print(f"[WASTE] Segregation model NOT FOUND at {config.WASTE_MATERIAL_MODEL}")

        type_model_path = getattr(config, "WASTE_TYPE_MODEL_PATH", None)
        print(f"[WASTE] Looking for RIC-type model at: {type_model_path}")
        if type_model_path and os.path.exists(type_model_path):
            self.model_type = YOLO(type_model_path)
            print(f"[WASTE] RIC-type model (final_7types) LOADED OK. Classes: {self.model_type.names}")
        else:
            print(f"[WASTE] Type model not found (optional) — segregation model only")
            self.model_type = None

        self._warmup()

    def _warmup(self):
        """Ek dummy inference load-time par hi chala do (blank frame par).
        Ultralytics/CUDA ka pehla call hamesha sabse slow hota hai (CUDA
        context init, cuDNN kernel selection, memory allocation) — isi wajah
        se pehli baar 'Start' dabane par stream 1-2 second ruk-ruk ke chalti
        hai ('attack-attack'/stutter). Warm-up isse startup me hi absorb kar
        leta hai, taaki live stream shuru hote hi smooth ho.
        """
        dummy = np.zeros((config.INFER_IMGSZ, config.INFER_IMGSZ, 3), dtype=np.uint8)
        with torch.inference_mode():
            if self.model_material is not None:
                self.model_material(dummy, device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                     imgsz=config.INFER_IMGSZ, verbose=False)
            if self.model_type is not None:
                self.model_type(dummy, device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                 imgsz=config.INFER_IMGSZ, verbose=False)
        print("[WASTE] Model warm-up done — first live frame will not stutter.")

    def status(self):
        return {
            "material_model_loaded": self.model_material is not None,
            "type_model_loaded": self.model_type is not None,
        }

    def _draw_box(self, frame, x1, y1, x2, y2, label, conf, source_tag):
        color = get_color_for_label(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        key = f"{source_tag}:{label}"
        now = time.time()
        if now - self.last_seen.get(key, 0) > 1.5:  # naive de-dup, ~1.5s window
            self.session_counts[key] = self.session_counts.get(key, 0) + 1
            self._log_detection(label, conf)
        self.last_seen[key] = now

    def process(self, frame):
        # NOTE: frame-skip ab centrally app.py ke capture worker me hoti hai —
        # jab yeh process() call hota hai, hamesha full inference chalao.
        self.frame_count += 1
        h, w = frame.shape[:2]

        if self.model_material is None:
            cv2.putText(frame, "Segregation model not loaded", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            return frame

        with torch.inference_mode():
            results = self.model_material(frame, conf=config.WASTE_CONF, verbose=False,
                                           device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                           imgsz=config.INFER_IMGSZ)

        plastic_boxes = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plastic_boxes.append((x1, y1, x2, y2, label, conf))
                self._draw_box(frame, x1, y1, x2, y2, label, conf, "seg")

        # ---- GATED: final_7types.pt runs ONLY if waste_seg2.pt found something ----
        if self.model_type is not None and config.WASTE_TYPE_GATE_ENABLED and plastic_boxes:
            pad = config.WASTE_TYPE_CROP_PADDING
            for (x1, y1, x2, y2, seg_label, seg_conf) in plastic_boxes:
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                with torch.inference_mode():
                    type_results = self.model_type(crop, conf=config.WASTE_CONF, verbose=False,
                                                     device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                                     imgsz=config.INFER_IMGSZ)
                # Best RIC-type prediction for this crop
                best_label, best_conf = None, 0.0
                for r in type_results:
                    names = r.names
                    for box in r.boxes:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf = c
                            best_label = names.get(int(box.cls[0]), str(int(box.cls[0])))
                if best_label:
                    tag = f"{best_label}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    ty = min(h - 5, y2 + th + 8)
                    color = get_color_for_label(best_label)
                    cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, ty), color, -1)
                    cv2.putText(frame, tag, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                    key = f"type:{best_label}"
                    now = time.time()
                    if now - self.last_seen.get(key, 0) > 1.5:
                        self.session_counts[key] = self.session_counts.get(key, 0) + 1
                        self._log_detection(best_label, best_conf)
                    self.last_seen[key] = now

        cv2.putText(frame, "SECONDARY PLASTIC CLASSIFICATION - LIVE", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 140), 2)
        return frame

    def get_counts(self):
        # Aggregate by class label (strip source tag)
        agg = {}
        for key, val in self.session_counts.items():
            label = key.split(":", 1)[1]
            agg[label] = agg.get(label, 0) + val
        return agg

    def reset_counts(self):
        self.session_counts = {}
        self.last_seen = {}
