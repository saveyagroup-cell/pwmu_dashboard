"""
PWMU Unified AI Control-Room Dashboard - Config
Sabhi model paths, thresholds aur Supabase settings yahan se control hote hain.
Environment variables .env file se load hote hain (python-dotenv).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- INFERENCE DEVICE (GPU auto-detect) ----------------
# Yeh sabse bada speed lever hai: agar isse GPU nahi mil raha (ya explicitly
# specify nahi kiya gaya), Ultralytics/EasyOCR CPU par gir jaate hain jo
# VS code me manually chalaye gaye script se 5-10x slow ho sakta hai.
# Yahan ek hi jagah GPU detect karke sab modules isi DEVICE ko use karte hain.
try:
    import torch
    _cuda_ok = torch.cuda.is_available()
except Exception:
    _cuda_ok = False

DEVICE = os.environ.get("INFERENCE_DEVICE", "cuda" if _cuda_ok else "cpu")
USE_HALF_PRECISION = DEVICE.startswith("cuda")  # fp16 -> ~2x faster on GPU, negligible accuracy loss
# Streaming ke liye chhota imgsz = zyada FPS. 640 ultralytics ka default hai;
# agar lag ho raha hai to 480 ya 416 try karo (.env: INFER_IMGSZ=480)
INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ", 640))

if _cuda_ok:
    try:
        torch.backends.cudnn.benchmark = True  # fixed input size (imgsz) -> cuDNN picks fastest kernels after warm-up
    except Exception:
        pass
    print(f"[CONFIG] GPU detected -> running inference on {DEVICE} (half={USE_HALF_PRECISION})")
else:
    print("[CONFIG] No GPU detected -> running inference on CPU. "
          "This IS the main reason live-stream inference feels slower here than a "
          "one-off script in VS Code (VS Code run pe bhi agar GPU nahi tha to same speed hogi). "
          "Agar aapke paas NVIDIA GPU hai, CUDA-enabled torch install karo "
          "(pip install torch --index-url https://download.pytorch.org/whl/cu121) "
          "aur is app ko restart karo.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

# ---------------- FLASK / SESSION ----------------
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-this-in-production")

# ---------------- MODEL PATHS ----------------
# Teeno trained .pt files "models/" folder me already daale ja chuke hain.
# Real pipeline (jaisa aapne specify kiya):
#   Camera -> waste_seg2.pt (plastic OBJECT detection: bag/bottle/container/cup/straw/utensil)
#               -> agar plastic object detect hua -> final_7types.pt (RIC type: PET/HDPE/PVC/LDPE/PP/PS/OTHERS)
#          -> number_plate.pt (ANPR, camera par independently, waste pipeline ko block nahi karta)
WASTE_SEG_MODEL = os.path.join(MODELS_DIR, "waste_seg2.pt")        # Primary Waste Segregation page + gate-model for secondary page
WASTE_MATERIAL_MODEL = WASTE_SEG_MODEL                              # alias (WasteProcessor isi naam se use karta hai)
WASTE_PRIMARY_MODEL = WASTE_SEG_MODEL                                # alias (WastePrimaryProcessor isi naam se use karta hai)
WASTE_TYPE_MODEL_PATH = os.path.join(MODELS_DIR, "final_7types.pt")  # Secondary/gated: plastic RIC type classifier
PLATE_MODEL = os.path.join(MODELS_DIR, "number_plate.pt")           # Number plate detector (independent pipeline)
VEHICLE_MODEL = "yolov8s.pt"      # COCO pretrained -> ultralytics auto-download, upload ki zaroorat nahi
THIEF_MODEL = "yolov8s.pt"        # Same pretrained model, person + bag classes use hote hain

# ---------------- DETECTION SETTINGS ----------------
WASTE_CONF = 0.40
WASTE_PRIMARY_CONF = 0.40
VEHICLE_CONF = 0.30
PLATE_CONF = 0.35
THIEF_CONF = 0.40

VEHICLE_CLASSES = [2, 3, 5, 7]     # car, motorbike, bus, truck (COCO ids)
BAG_CLASSES = [24, 26, 28]         # backpack, handbag, suitcase (COCO ids)

VEHICLE_LINE_Y_RATIO = 0.6         # frame height ka 60% par IN/OUT line

# Loitering / thief detection
LOITER_SECONDS_TEST = 5            # test mode me jaldi trigger
LOITER_SECONDS_NIGHT = 300         # real mode me 5 min
ALERT_COOLDOWN_SEC = 12
TEST_MODE = True                   # False karo real deployment ke liye (raat 10pm-6am hi active hoga)

# OCR filters
PLATE_MIN_LEN = 4
PLATE_MIN_OCR_CONF = 0.40
PLATE_SAVE_COOLDOWN = 3.0

# ---- Advanced ANPR: multi-frame tracking + OCR voting (accuracy upgrade) ----
# Shape filter — rejects false positives like headlights/stickers
PLATE_MIN_ASPECT_RATIO = 1.3
PLATE_MAX_ASPECT_RATIO = 6.5
PLATE_MIN_BOX_AREA = 80

# Tracking (IOU-based, no extra library needed)
PLATE_TRACK_TIMEOUT = 2.0        # track expires if not seen for this long (seconds)
PLATE_IOU_MATCH_THRESH = 0.3
PLATE_READING_TIMEOUT = 3.0      # stuck in "reading..." this long -> save as unconfirmed, don't wait forever

# Multi-frame OCR voting/confirmation
PLATE_MIN_VOTE_FRAMES_EARLY = 3    # N consecutive frames with EXACT same text -> early confirm
PLATE_EARLY_CONFIRM_MIN_CONF = 0.6
PLATE_MAX_VOTE_FRAMES = 8          # strict fallback: character-level weighted vote after this many frames
PLATE_MIN_VOTE_AGREEMENT = 0.7

# Duplicate-plate detection (fuzzy match against already-saved plates this session)
PLATE_SIMILARITY_THRESHOLD = 0.82

# ---------------- OUTPUT / RECORDING ----------------
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------- ESTIMATED PLASTIC RESALE PRICES (₹ per kg) ----------------
# Yeh APPROXIMATE market rates hain — asli revenue ke liye weighing scale integration
# chahiye hoga. Abhi count-based rough estimate ke liye (avg 25g/item assume kiya hai).
PLASTIC_PRICE_PER_KG = {
    "PET": 18, "HDPE": 22, "PVC": 8, "LDPE": 14,
    "POLYPROPYLENE": 20, "POLYSTYRENE": 6, "OTHERS": 5,
}
ASSUMED_AVG_ITEM_WEIGHT_KG = 0.025  # 25 grams/item, rough estimate

# ---------------- CAMERA ----------------
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

# ---------------- SUPABASE ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
# Auth (sign up / sign in) MUST use the anon key — this is what Supabase Auth expects.
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
# Storage/DB writes: service_role is recommended (bypasses Storage RLS), but
# falls back to SUPABASE_KEY / anon key if that's all you've set.
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or os.environ.get("SUPABASE_KEY", "") or SUPABASE_ANON_KEY
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "pwmu-captures")
SUPABASE_ANPR_BUCKET = os.environ.get("SUPABASE_ANPR_BUCKET", "anpr-detections")
SUPABASE_SECURITY_BUCKET = os.environ.get("SUPABASE_SECURITY_BUCKET", "security-detections")

# ---------------- TELEGRAM (optional, thief alerts) ----------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Processing performance: har Nth frame par hi inference chalao (webcam ko smooth rakhne ke liye)
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 2))
# Plate module sabse heavy hai (YOLO + EasyOCR dono) — isko zyada aggressively skip karo
PLATE_FRAME_SKIP = int(os.environ.get("PLATE_FRAME_SKIP", 5))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 75))

# Adaptive frame skip: agar ek module ka actual inference time target se zyada
# ho jaaye, capture worker khud skip badha deta hai (aur jab fast ho jaaye to
# wapas kam kar deta hai) — taaki stream kabhi freeze na ho, chahe CPU/GPU load
# kuch bhi ho. 0 = adaptive OFF (sirf fixed FRAME_SKIP use hoga).
ADAPTIVE_SKIP = os.environ.get("ADAPTIVE_SKIP", "1") == "1"
ADAPTIVE_SKIP_TARGET_MS = int(os.environ.get("ADAPTIVE_SKIP_TARGET_MS", 80))  # ~12 FPS inference target
ADAPTIVE_SKIP_MAX = int(os.environ.get("ADAPTIVE_SKIP_MAX", 8))
FPS_LOG_INTERVAL_SEC = int(os.environ.get("FPS_LOG_INTERVAL_SEC", 5))

# ---- Secondary Plastic Classification gating (waste_seg2.pt -> final_7types.pt) ----
# final_7types.pt (RIC type classifier) MUST run ONLY when waste_seg2.pt ne
# frame me koi plastic object detect kiya ho. Yeh koi timer/skip nahi hai —
# purely detection-driven condition, jaisa aapne pipeline me specify kiya.
WASTE_TYPE_GATE_ENABLED = True
# Har detected plastic-object box ka crop nikaal kar us CROP par hi
# final_7types.pt chalao (poore frame par nahi) — chhota crop = tez inference,
# aur zyada accurate RIC-type prediction bhi (model sirf us item ko dekhta hai).
WASTE_TYPE_CROP_PADDING = 10  # px, box ke around thoda extra context
