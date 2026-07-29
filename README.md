# PWMU Unified AI Control-Room Dashboard

**Team EchoByte** · AI-powered monitoring dashboard for Plastic Waste Management Units (PWMUs) in rural Chhattisgarh

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Supabase](https://img.shields.io/badge/Backend-Supabase-3ECF8E)
![License](https://img.shields.io/badge/status-active--development-yellow)

---

## Project Overview

PWMU Control-Room is a Flask-based computer-vision dashboard that turns a single camera feed (live webcam or uploaded video) into real-time operational data for a Plastic Waste Management Unit. It runs four independent AI modules and streams the annotated video plus live counters to a browser dashboard, while writing every event to Supabase for history, reporting, and audit.

**Core modules:**

| Module | What it does | Model |
|---|---|---|
| Gate & Security (Vehicle Counter + ANPR) | Line-crossing IN/OUT vehicle counting + number-plate detection and OCR | `yolov8s.pt` + `number_plate.pt` |
| Primary Segregation | Metal vs. Other waste classification (Conveyor 1) | `waste_primary.pt` |
| Secondary Plastic Classification | Plastic RIC-type detection — PET / HDPE / PVC / LDPE / PP / PS / OTHERS (Conveyor 2) | `final_7types.pt` |
| PWMU Shed Security | Zone-based loitering & unattended-object detection, with Telegram + Supabase alerts | `yolov8s.pt` |

**Key features**
- Live MJPEG camera streaming, or drag-and-drop video upload, per module
- Supabase Auth login/signup (email + password) with operator profile (name, PWM unit, district)
- Real-time KPI cards, audit log, and hourly charts on the Command Center
- CSV and PDF report export (daily / weekly / monthly / all-time)
- Estimated plastic resale revenue based on detected item counts
- Multi-language UI (English / Hindi / Chhattisgarhi)
- Institutional branding: NIT Raipur · Government of Chhattisgarh · UNICEF

**Tech stack:** Python, Flask, OpenCV, Ultralytics YOLOv8, EasyOCR, Supabase (Postgres + Auth + Storage), ReportLab, Matplotlib.

---

## Repository Structure

```
pwmu_dashboard/
├── app.py                    # Flask app entrypoint — routes, camera threads, API
├── config.py                 # Model paths, thresholds, Supabase/Telegram settings
├── modules/                  # One file per detection module (waste, vehicle, plate, thief, gate)
│   ├── auth.py                # Supabase Auth (sign up / sign in / profile)
│   ├── supabase_client.py      # Storage upload + table insert/read helper
│   ├── reports.py / pdf_report.py
│   └── ...
├── models/                   # YOLO .pt weight files (see Installation Guide)
├── templates/                 # Jinja2 HTML pages
├── static/                    # css / js / images
├── captures/                  # Local CSV logs + fallback image saves
├── outputs/                   # Recorded annotated session videos
├── supabase_schema.sql        # DB tables + storage buckets + RLS policies
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation Guide

### Prerequisites
- Python 3.10 or 3.11
- pip
- A webcam (for live mode) — optional, video upload works without one
- A free [Supabase](https://supabase.com) project (for auth, history, and image storage)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/saveyagroup-cell/PWMU-System.git
cd pwmu_dashboard

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Model weights** — place your trained `.pt` files in `models/` using these exact names (or update the paths in `config.py`):

```
models/
├── final_7types.pt      # Secondary plastic RIC-type model
├── waste_primary.pt     # Primary metal-vs-other model
└── number_plate.pt      # Number-plate detector
```

`yolov8s.pt` (used for vehicle counting and loitering detection) does **not** need to be added manually — Ultralytics downloads it automatically on first run. For fully offline use, download it once and point `config.py` at the local file.

**Set up Supabase:**
1. Open your Supabase project → **SQL Editor** → paste and run the full contents of `supabase_schema.sql`. This creates the `profiles`, `plate_detections`, `thief_alerts`, and `waste_sessions` tables, plus RLS policies.
2. Confirm three **Storage buckets** exist (created automatically by the script, or add manually under Storage → New Bucket, each set to **Public: ON**):
   - `pwmu-captures`
   - `anpr-detections`
   - `security-detections`

**Run locally:**

```bash
python app.py
```

Open **http://localhost:5000** in your browser, sign up, and start any module from the Home page.

> The app also runs without Supabase configured — auth, cloud image storage, and cross-restart history won't work, but local detection, live streaming, and CSV logging still do.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your real values. **Never commit `.env` to GitHub** (already in `.gitignore`).

```env
# ---- Flask ----
# A random, long secret string used to sign session cookies — change this in production.
FLASK_SECRET_KEY=change-this-to-a-random-secret-string

# ---- Supabase ----
SUPABASE_URL=https://xxxxxxxx.supabase.co

# Used for Auth (Sign Up / Sign In) — Project Settings > API > anon/public key
SUPABASE_ANON_KEY=your-supabase-anon-key

# Used for Storage/DB writes — service_role is recommended (bypasses Storage RLS).
# Safe to use here because it is only ever read server-side by Flask, never sent to the browser.
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

SUPABASE_BUCKET=pwmu-captures
SUPABASE_ANPR_BUCKET=anpr-detections
SUPABASE_SECURITY_BUCKET=security-detections

# ---- Telegram (optional — for thief/security alerts) ----
# If you ever hardcoded a bot token during testing, revoke it immediately
# (BotFather -> /revoke) and generate a new one before deploying.
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ---- Camera ----
CAMERA_INDEX=0
FRAME_SKIP=2
PLATE_FRAME_SKIP=5
JPEG_QUALITY=75
```

| Variable | Required | Purpose |
|---|---|---|
| `FLASK_SECRET_KEY` | Yes | Signs the session cookie |
| `SUPABASE_URL` | Yes (for cloud features) | Supabase project endpoint |
| `SUPABASE_ANON_KEY` | Yes (for cloud features) | Auth sign-up / sign-in |
| `SUPABASE_SERVICE_ROLE_KEY` | Recommended | Storage uploads + table writes (bypasses RLS) |
| `SUPABASE_BUCKET` / `_ANPR_BUCKET` / `_SECURITY_BUCKET` | Yes (for cloud features) | Storage bucket names |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional | Sends a Telegram message on security alerts |
| `CAMERA_INDEX` | No (default `0`) | Local webcam device index |
| `FRAME_SKIP` / `PLATE_FRAME_SKIP` | No | Inference throttling for smoother live streams |
| `JPEG_QUALITY` | No (default `75`) | Stream image quality vs. bandwidth trade-off |

---

## Deployment Instructions

This app is a long-running Flask server with background camera/inference threads and an MJPEG video stream, so it needs a host that keeps a persistent process alive — **Render is the better fit**; Vercel's serverless functions are not designed for continuous background threads or live video streaming, though it can still host the app for **uploaded-video processing** if you're set on using it. Either way, live *webcam* capture only works when a physical camera is attached to the machine running the app — on any cloud host, only the **video-upload** mode will function (there is no webcam on the server).

### Option A — Render (recommended)

1. Push your code to GitHub.
2. On [Render](https://render.com), click **New → Web Service** and connect your repository.
3. Configure:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   *(add `gunicorn` to `requirements.txt` if it isn't already there; `app.run(debug=True)` in `app.py` is for local dev only)*
4. Add all variables from `.env` under **Environment → Environment Variables**.
5. Under **Disks**, optionally attach a persistent disk if you want `captures/`, `outputs/`, and `uploads/` to survive restarts (otherwise they reset on every redeploy — Supabase history is unaffected either way).
6. Deploy. Render gives you a public URL (`https://your-app.onrender.com`).

### Option B — Vercel (video-upload workflows only)

1. Push your code to GitHub and import the repo into [Vercel](https://vercel.com).
2. Add a `vercel.json` to route all traffic to your Flask app via Vercel's Python runtime:
   ```json
   {
     "builds": [{ "src": "app.py", "use": "@vercel/python" }],
     "routes": [{ "src": "/(.*)", "dest": "app.py" }]
   }
   ```
3. Add all `.env` variables under **Project Settings → Environment Variables**.
4. Be aware of Vercel's serverless limits: no persistent background threads, a hard execution timeout per request, and no writable local disk beyond `/tmp` — so live streaming (`/video_feed/*`), long processing jobs, and local CSV/log persistence will not behave the same as on Render.

---

## Team Members

| Name | Role |
|---|---|
| _Add your name_ | _e.g. Team Lead / ML Engineer_ |
| _Add teammate name_ | _e.g. Frontend / Backend_ |
| _Add teammate name_ | _e.g. Computer Vision_ |

*(Update this table with your actual team roster before submission.)*

---

## System Screenshots

> Add screenshots of the running dashboard here before submitting — e.g.:

```
docs/screenshots/
├── home_hub.png
├── gate_security.png
├── ai_segregation.png
└── dashboard_analytics.png
```

```markdown
![Home Hub](docs/screenshots/home_hub.png)
![Gate & Security](docs/screenshots/gate_security.png)
![AI Segregation](docs/screenshots/ai_segregation.png)
![Analytics Dashboard](docs/screenshots/dashboard_analytics.png)
```

---

## Live Demo Link

🔗 `https://your-app.onrender.com` — *(replace with your deployed Render/Vercel URL)*

---

## License

Built for academic/hackathon submission by Team EchoByte, NIT Raipur.
