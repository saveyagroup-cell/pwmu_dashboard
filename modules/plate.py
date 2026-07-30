"""
ANPR — YOLOv8 plate detection + EasyOCR, upgraded with multi-frame
track-based OCR voting (adapted from the standalone high-accuracy ANPR
script): each physical plate is tracked across frames (IOU matching), OCR
runs every frame it's visible, and a plate number is only "confirmed" once
multiple independent frames agree — this avoids one-off OCR mistakes
(like 0/O confusion) from being saved as fact.

NOTE: unlike the standalone script, this does NOT keep its own CSV/crops
folder system — it reuses this app's existing Supabase + CSV log flow
(_save_detection), so there's one single source of truth instead of two.
"""
import os
import re
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import config
from modules import supabase_client

_reader = None  # lazy-loaded easyocr reader (slow to init)
PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        # GPU explicitly pass karo — warna EasyOCR kabhi-kabhi CPU par gir
        # jaata hai jo OCR pass ko sabse slow step bana deta hai (yeh module
        # ka sabse bhaari hissa hai: YOLO detect + OCR har visible plate par).
        _reader = easyocr.Reader(['en'], gpu=(config.DEVICE == "cuda"))
        print(f"[PLATE] EasyOCR reader loaded (gpu={config.DEVICE == 'cuda'})")
    return _reader


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def _clean_hsrp_noise(text):
    """HSRP hologram sometimes gets OCR'd as a stray 'IND' prefix/suffix."""
    if text.startswith("IND") and len(text) > 3:
        text = text[3:]
    if text.endswith("IND") and len(text) > 3:
        text = text[:-3]
    return text


def _is_valid_plate_shape(x1, y1, x2, y2):
    width, height = x2 - x1, y2 - y1
    if height <= 0 or width <= 0:
        return False
    ratio = width / height
    return (config.PLATE_MIN_ASPECT_RATIO <= ratio <= config.PLATE_MAX_ASPECT_RATIO) \
        and (width * height >= config.PLATE_MIN_BOX_AREA)


class PlateProcessor:
    def __init__(self):
        self.model = YOLO(config.PLATE_MODEL) if os.path.exists(config.PLATE_MODEL) else None
        if self.model is None:
            print(f"[PLATE] Model not found at {config.PLATE_MODEL}")
        else:
            print(f"[PLATE] Model loaded from {config.PLATE_MODEL}")
            print(f"[PLATE] Model classes: {self.model.names}")
            self._warmup()

        self.frame_count = 0
        self.last_saved = {}
        self.recent_plates = []  # for UI display: list of {plate, conf, time, url}
        self.saved_plate_texts = []  # for fuzzy duplicate-detection across the session

        # Multi-frame tracking state: {track_id: {...}}
        self.tracks = {}
        self._next_track_id = 0

        import csv
        self._csv = csv
        self.log_path = os.path.join(config.CAPTURES_DIR, "detected_plates.csv")
        self._ensure_log()

    def _ensure_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                self._csv.writer(f).writerow(["Date", "Time", "Plate Number", "Confidence",
                                               "Vehicle_Type", "Direction", "Source", "ImageURL"])

    def _warmup(self):
        """Dummy YOLO pass at load time — avoids the first-click stutter caused
        by CUDA/cuDNN warm-up happening on-camera instead of at startup."""
        dummy = np.zeros((config.INFER_IMGSZ, config.INFER_IMGSZ, 3), dtype=np.uint8)
        with torch.inference_mode():
            self.model.predict(source=dummy, conf=config.PLATE_CONF, verbose=False,
                                device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                imgsz=config.INFER_IMGSZ)
        print("[PLATE] Model warm-up done.")

    # ---------------------------------------------------------------
    # OCR
    # ---------------------------------------------------------------
    @staticmethod
    def _preprocess_variants(crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = 3.5 if crop.shape[0] < 40 else 2.5
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        v1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        v2 = clahe.apply(gray)
        return gray, v1, v2

    def _ocr_pass(self, variant):
        results = _get_reader().readtext(variant, allowlist=PLATE_CHARS)
        out = []
        for (_, text, conf) in results:
            text = re.sub(r"[^A-Z0-9]", "", text.upper())
            text = _clean_hsrp_noise(text)
            if 4 <= len(text) <= 10 and conf >= config.PLATE_MIN_OCR_CONF:
                out.append((text, conf))
        return out

    def _read_plate_candidates(self, crop):
        """Fast path: try 2 cheap variants; only run a 3rd (Otsu) if they disagree."""
        if crop.size == 0:
            return []
        gray, v1, v2 = self._preprocess_variants(crop)
        candidates = self._ocr_pass(v1) + self._ocr_pass(v2)

        texts_found = set(t for t, c in candidates)
        if len(texts_found) <= 1:
            return candidates  # already consistent — skip the expensive 3rd pass

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, v3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates += self._ocr_pass(v3)
        return candidates

    # ---------------------------------------------------------------
    # Multi-frame confirmation (this is the accuracy upgrade)
    # ---------------------------------------------------------------
    @staticmethod
    def _char_level_vote(votes, min_frames):
        if len(votes) < min_frames:
            return "", 0.0, False

        length_groups = Counter(len(t) for t, c in votes)
        target_len, len_count = length_groups.most_common(1)[0]
        if len_count / len(votes) < config.PLATE_MIN_VOTE_AGREEMENT:
            return "", 0.0, False

        same_len_votes = [(t, c) for t, c in votes if len(t) == target_len]
        final_chars = []
        for pos in range(target_len):
            char_weights = {}
            for text, conf in same_len_votes:
                ch = text[pos]
                char_weights[ch] = char_weights.get(ch, 0) + conf
            total_weight = sum(char_weights.values())
            best_char = max(char_weights, key=char_weights.get)
            agreement = char_weights[best_char] / total_weight if total_weight > 0 else 0
            if agreement < config.PLATE_MIN_VOTE_AGREEMENT:
                return "", 0.0, False
            final_chars.append(best_char)

        final_text = "".join(final_chars)
        avg_conf = sum(c for t, c in same_len_votes) / len(same_len_votes)
        return final_text, avg_conf, True

    def _try_confirm(self, track):
        """2-level strategy: needs agreement across MULTIPLE independent
        frames either way — no single-frame shortcut, so a one-off OCR
        mistake never gets saved as a confirmed plate."""
        votes = track["votes"]
        if not votes:
            return "", 0.0, False

        if len(votes) >= config.PLATE_MIN_VOTE_FRAMES_EARLY:
            recent = votes[-config.PLATE_MIN_VOTE_FRAMES_EARLY:]
            texts = set(t for t, c in recent)
            if len(texts) == 1:
                text = list(texts)[0]
                avg_conf = sum(c for t, c in recent) / len(recent)
                if avg_conf >= config.PLATE_EARLY_CONFIRM_MIN_CONF:
                    return text, avg_conf, True

        return self._char_level_vote(votes, config.PLATE_MAX_VOTE_FRAMES)

    def _match_track(self, box):
        best_id, best_iou = None, config.PLATE_IOU_MATCH_THRESH
        for tid, t in self.tracks.items():
            score = _iou(box, t["box"])
            if score > best_iou:
                best_id, best_iou = tid, score
        if best_id is not None:
            return best_id
        tid = self._next_track_id
        self._next_track_id += 1
        self.tracks[tid] = {"box": box, "votes": [], "last_seen": time.time(),
                             "first_seen": time.time(), "saved_text": None,
                             "final_text": "", "last_crop": None, "unconfirmed_saved": False}
        return tid

    def _cleanup_tracks(self):
        now = time.time()
        stale = [tid for tid, t in self.tracks.items() if now - t["last_seen"] > config.PLATE_TRACK_TIMEOUT]
        for tid in stale:
            t = self.tracks[tid]
            if not t["saved_text"] and not t["unconfirmed_saved"] and t["last_crop"] is not None:
                self._save_unconfirmed(t)
            del self.tracks[tid]

    def _is_duplicate(self, text):
        for existing in self.saved_plate_texts:
            max_len = max(len(text), len(existing))
            if max_len and (1 - _levenshtein(text, existing) / max_len) >= config.PLATE_SIMILARITY_THRESHOLD:
                return True
        return False

    def _save_unconfirmed(self, track):
        best_guess, best_conf = "", 0.0
        if track["votes"]:
            best_guess, best_conf = max(track["votes"], key=lambda v: v[1])
        track["unconfirmed_saved"] = True
        self._save_detection(best_guess or "UNREADABLE", best_conf, track["last_crop"], confirmed=False)

    # ---------------------------------------------------------------
    # Save / log (unchanged app-wide flow: local CSV + Supabase)
    # ---------------------------------------------------------------
    def _save_detection(self, plate_text, ocr_conf, crop, confirmed, vehicle_type="—", direction="—", source="auto"):
        now = datetime.now()
        now_ts = now.timestamp()

        dedupe_key = plate_text if confirmed else "unconfirmed"
        if source == "auto" and dedupe_key in self.last_saved and \
                (now_ts - self.last_saved[dedupe_key]) < config.PLATE_SAVE_COOLDOWN:
            return
        if confirmed and self._is_duplicate(plate_text):
            print(f"[PLATE] Skipping duplicate: {plate_text}")
            return
        self.last_saved[dedupe_key] = now_ts

        image_url = None
        if crop is not None and crop.size > 0:
            local_path = os.path.join(config.CAPTURES_DIR, f"plate_{int(now_ts*1000)}.jpg")
            cv2.imwrite(local_path, crop)
            print(f"[PLATE] Image saved locally: {local_path} (confirmed={confirmed}, text='{plate_text}')")

            image_url = supabase_client.upload_image(local_path, "plates", bucket=config.SUPABASE_ANPR_BUCKET)
            if image_url:
                print(f"[PLATE] Image uploaded to Supabase: {image_url}")
            elif supabase_client.is_enabled():
                print("[PLATE] Supabase upload failed — image kept locally only")

        with open(self.log_path, "a", newline="") as f:
            self._csv.writer(f).writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                                           plate_text, round(ocr_conf, 2), vehicle_type, direction,
                                           source, image_url or ""])

        supabase_client.insert_row("plate_detections", {
            "plate_number": plate_text,
            "confidence": round(float(ocr_conf), 2),
            "image_url": image_url,
            "detected_at": now.isoformat(),
        })

        if confirmed:
            self.saved_plate_texts.append(plate_text)

        entry = {"plate": plate_text, "conf": round(ocr_conf, 2), "time": now.strftime("%H:%M:%S"),
                  "url": image_url, "vehicle_type": vehicle_type, "direction": direction, "source": source}
        self.recent_plates.insert(0, entry)
        self.recent_plates = self.recent_plates[:15]

    def manual_entry(self, plate_text, vehicle_type="—", direction="—"):
        plate_text = "".join(c for c in plate_text if c.isalnum()).upper() or "UNKNOWN"
        self._save_detection(plate_text, 1.0, None, confirmed=True,
                              vehicle_type=vehicle_type, direction=direction, source="manual")

    # ---------------------------------------------------------------
    # Main per-frame entry point
    # ---------------------------------------------------------------
    def process(self, frame, vehicle_context=None):
        if self.model is None:
            cv2.putText(frame, "number_plate.pt NOT FOUND in /models", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        vtype = vehicle_context["vehicle_type"] if vehicle_context else "—"
        direction = vehicle_context["direction"] if vehicle_context else "—"

        self.frame_count += 1
        h, w = frame.shape[:2]
        with torch.inference_mode():
            results = self.model.predict(source=frame, conf=config.PLATE_CONF, verbose=False,
                                          device=config.DEVICE, half=config.USE_HALF_PRECISION,
                                          imgsz=config.INFER_IMGSZ)

        total_boxes = sum(len(r.boxes) for r in results)
        if self.frame_count % 30 == 0:
            print(f"[PLATE] frame {self.frame_count}: {total_boxes} box(es), "
                  f"{len(self.tracks)} active track(s)")

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if not _is_valid_plate_shape(x1, y1, x2, y2):
                    continue  # false positive shape (headlight/sticker etc.)

                tid = self._match_track((x1, y1, x2, y2))
                track = self.tracks[tid]
                track["box"] = (x1, y1, x2, y2)
                track["last_seen"] = time.time()

                if track["saved_text"]:
                    # Already confirmed — just draw the known-good result, no more OCR needed
                    self._draw_box(frame, x1, y1, x2, y2, track["final_text"], tid, confirmed=True)
                    continue

                crop = frame[y1:y2, x1:x2]
                track["last_crop"] = crop.copy()

                candidates = self._read_plate_candidates(crop)
                if candidates:
                    track["votes"].extend(candidates)

                final_text, final_conf, is_confident = self._try_confirm(track)

                if is_confident and len(final_text) >= config.PLATE_MIN_LEN:
                    track["saved_text"] = final_text
                    track["final_text"] = final_text
                    self._draw_box(frame, x1, y1, x2, y2, final_text, tid, confirmed=True)
                    self._save_detection(final_text, final_conf, crop, confirmed=True,
                                          vehicle_type=vtype, direction=direction)
                else:
                    self._draw_box(frame, x1, y1, x2, y2, "", tid, confirmed=False)
                    reading_duration = time.time() - track["first_seen"]
                    if reading_duration >= config.PLATE_READING_TIMEOUT and not track["unconfirmed_saved"]:
                        track["unconfirmed_saved"] = True
                        self._save_detection("UNREADABLE", final_conf, crop, confirmed=False,
                                              vehicle_type=vtype, direction=direction)

        self._cleanup_tracks()
        return frame

    @staticmethod
    def _draw_box(frame, x1, y1, x2, y2, text, track_id, confirmed):
        color = (0, 255, 0) if confirmed else (0, 165, 255)  # green=confirmed, orange=reading/unreadable
        label = f"#{track_id} {text}" if confirmed else f"#{track_id} reading..."
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1 - th - 10, 0)
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def get_recent(self):
        return self.recent_plates
