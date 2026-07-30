"""
PWMU Unified AI Control-Room Dashboard
Team Ecobyte x Robosapiens

Modules:
  - waste   : Plastic waste segregation (2x YOLO models)
  - vehicle : Vehicle IN/OUT counting (YOLOv8 + ByteTrack)
  - plate   : Number plate detection + OCR (YOLOv8 + EasyOCR) -> Supabase
  - thief   : Loitering / unattended-object security detection -> Supabase (+ Telegram)
"""
import os
import threading
import time
import queue
import atexit
import signal
from datetime import datetime
from functools import wraps

import cv2
from flask import (Flask, Response, render_template, request, jsonify, redirect,
                    url_for, send_file, session)

import config
from modules.waste import WasteProcessor
from modules.waste_primary import WastePrimaryProcessor
from modules.vehicle import VehicleProcessor
from modules.plate import PlateProcessor
from modules.thief import ThiefProcessor
from modules.gate import GateProcessor
from modules import supabase_client
from modules import reports
from modules import pdf_report
from modules import auth

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max video upload
app.secret_key = config.FLASK_SECRET_KEY


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
PUBLIC_PATHS = {"/login", "/signup", "/logout"}


@app.before_request
def require_login():
    if request.path.startswith("/static/") or request.path in PUBLIC_PATHS:
        return
    if not session.get("user_id"):
        return redirect(url_for("login_page", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        return render_template("login.html", error=None)

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    user, error = auth.sign_in(email, password)
    if error:
        return render_template("login.html", error=error)

    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    return redirect(request.args.get("next") or url_for("home_page"))


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "GET":
        return render_template("signup.html", error=None)

    name = request.form.get("name", "").strip()
    pwm_unit = request.form.get("pwm_unit", "").strip()
    district = request.form.get("district", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not all([name, pwm_unit, district, email, password]):
        return render_template("signup.html", error="All fields are required.")
    if len(password) < 6:
        return render_template("signup.html", error="Password must be at least 6 characters.")

    user_id, error = auth.sign_up(email, password, name, pwm_unit, district)
    if error:
        return render_template("signup.html", error=error)

    session["user_id"] = user_id
    session["user_email"] = email
    return redirect(url_for("home_page"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


def _current_profile():
    return auth.get_profile(session.get("user_id"))

# ---------------------------------------------------------------------------
# Lazy-loaded processors (models load only when a module is first switched ON,
# so the app boots instantly and you don't pay GPU/RAM cost for unused modules)
# ---------------------------------------------------------------------------
_processors = {}
_processor_classes = {
    # New Command Center modules (2x2 grid)
    "gate": GateProcessor,                  # Gate 1: ANPR + Vehicle Counter combined
    "waste_primary": WastePrimaryProcessor,  # Conveyor 1: Metal vs Other
    "waste_secondary": WasteProcessor,       # Conveyor 2: Plastic type (PET/HDPE/...)
    "thief": ThiefProcessor,                 # PWMU Shed: Theft & Anomaly

    # Legacy single-module pages (still reachable directly, e.g. /vehicle, /plate)
    "waste": WasteProcessor,
    "vehicle": VehicleProcessor,
    "plate": PlateProcessor,
}


def get_processor(name):
    if name not in _processors:
        _processors[name] = _processor_classes[name]()
    return _processors[name]


# ---------------------------------------------------------------------------
# Camera / stream state — ek dict jo har module ke live camera ko manage karta hai
# ---------------------------------------------------------------------------
class StreamState:
    """Har module ka apna background capture+inference thread hota hai, jo
    Flask ke HTTP response thread se poori tarah decoupled hai. Yeh asli
    fix hai us lag ke liye jo pehle ho raha tha — pehle har HTTP frame-pull
    khud hi cv2.read() + YOLO inference + encode sequentially kar raha tha,
    jisse browser network speed pe hi poori pipeline atki rehti thi.
    """
    def __init__(self):
        self.active = False
        self.cap = None
        self.source = 0          # 0 = webcam, ya video file path
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)  # sirf LATEST annotated frame rakhta hai
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.writer = None              # cv2.VideoWriter — records annotated output for download
        self.output_path = None         # path to the most recently finished recording
        self.session_start = None


streams = {name: StreamState() for name in _processor_classes}


def _open_capture(source):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # webcam driver ka apna buffer bhi chhota rakho
    return cap


def _start_recording(module_name, st, cap):
    """Start a VideoWriter so the annotated output can be downloaded later."""
    module_dir = os.path.join(config.OUTPUTS_DIR, module_name)
    os.makedirs(module_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(module_dir, f"{module_name}_{ts}.mp4")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    if fps <= 1 or fps > 60:
        fps = 20
    st.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    st.session_start = time.time()
    st._pending_output_path = out_path
    print(f"[RECORD] {module_name}: recording annotated output to {out_path}")


def _stop_recording(module_name, st):
    if st.writer is not None:
        st.writer.release()
        st.writer = None
        st.output_path = getattr(st, "_pending_output_path", None)
        print(f"[RECORD] {module_name}: saved -> {st.output_path}")


def _queue_put_latest(q, frame):
    """Purana frame drop karke sirf latest rakho — isse browser ko hamesha
    freshest frame milta hai, koi buffered/laggy frame backlog nahi banta."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(frame)
    except queue.Full:
        pass


def _capture_worker(module_name, st):
    """Background thread: camera/video padhta hai, YOLO inference chalata hai
    (sirf har FRAME_SKIP-th frame par — bakayaki frames pe last annotated
    overlay reuse hota hai taaki stream smooth dikhe), aur latest annotated
    frame ko queue me daal deta hai jise HTTP stream turant serve kar deta hai
    — inference kitni bhi slow ho, browser ka stream isse block nahi hota."""
    processor = get_processor(module_name)
    cap = _open_capture(st.source)
    with st.lock:
        st.cap = cap

    if not cap.isOpened():
        print(f"[{module_name}] Could not open camera/video source: {st.source}")
        st.active = False
        return

    frame_idx = 0
    last_annotated = None
    recording_started = False

    # Plate/OCR sabse heavy hai — usko zyada skip do; baaki normal FRAME_SKIP.
    # Yeh sirf STARTING point hai — agar ADAPTIVE_SKIP on hai to yeh khud
    # runtime me measured inference-time ke hisaab se badhta/ghatta rahega.
    skip = config.PLATE_FRAME_SKIP if module_name == "plate" else config.FRAME_SKIP

    # ---- FPS / latency monitoring (console log every FPS_LOG_INTERVAL_SEC) ----
    fps_window_start = time.time()
    fps_frame_count = 0
    fps_infer_count = 0
    total_infer_time = 0.0

    while not st.stop_event.is_set():
        try:
            ok, frame = cap.read()
        except Exception as e:
            print(f"[{module_name}] camera read error: {e}")
            break
        if not ok:
            break  # video file khatam, ya webcam disconnect

        if not recording_started:
            _start_recording(module_name, st, cap)
            recording_started = True

        frame_idx += 1
        fps_frame_count += 1
        run_inference = (frame_idx % max(skip, 1) == 0) or (last_annotated is None)

        if run_inference:
            t0 = time.time()
            try:
                annotated = processor.process(frame)
            except Exception as e:
                annotated = frame
                cv2.putText(annotated, f"ERROR: {e}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"[{module_name}] inference error: {e}")
            infer_ms = (time.time() - t0) * 1000
            total_infer_time += infer_ms
            fps_infer_count += 1
            last_annotated = annotated

            # Adaptive skip: agar is module ka inference target se slow ho raha
            # hai, thoda zyada skip karo (max cap tak); agar tez hai to skip
            # ghata do (min 1) — stream kabhi bhi freeze/backlog nahi hoti.
            if config.ADAPTIVE_SKIP:
                if infer_ms > config.ADAPTIVE_SKIP_TARGET_MS and skip < config.ADAPTIVE_SKIP_MAX:
                    skip += 1
                elif infer_ms < config.ADAPTIVE_SKIP_TARGET_MS / 2 and skip > 1:
                    skip -= 1
        else:
            # Inference skip — purana annotated overlay hi dobara use karo taaki
            # boxes flicker na karein, bina dobara heavy model chalaye
            annotated = last_annotated

        if st.writer is not None:
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or annotated.shape[1]
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or annotated.shape[0]
                out_frame = cv2.resize(annotated, (w, h)) if (annotated.shape[1], annotated.shape[0]) != (w, h) else annotated
                st.writer.write(out_frame)
            except Exception as e:
                print(f"[RECORD] write failed: {e}")

        _queue_put_latest(st.frame_queue, annotated)

        # Periodic performance log — helps confirm GPU vs CPU speed live,
        # instead of guessing why a module "feels slow".
        elapsed = time.time() - fps_window_start
        if elapsed >= config.FPS_LOG_INTERVAL_SEC:
            stream_fps = fps_frame_count / elapsed
            avg_infer_ms = (total_infer_time / fps_infer_count) if fps_infer_count else 0.0
            print(f"[{module_name}] {stream_fps:.1f} FPS | avg inference {avg_infer_ms:.0f} ms "
                  f"| skip={skip} | device={config.DEVICE}")
            fps_window_start = time.time()
            fps_frame_count = 0
            fps_infer_count = 0
            total_infer_time = 0.0

    # Cleanup
    _stop_recording(module_name, st)
    cap.release()
    with st.lock:
        st.cap = None
        st.active = False
    print(f"[{module_name}] capture worker stopped")


def gen_frames(module_name):
    """HTTP-facing generator — sirf queue se latest annotated frame utha ke
    JPEG encode karke bhejta hai. Koi camera read ya YOLO inference yahan
    NAHI hota, isliye browser stream turant respond karta hai."""
    st = streams[module_name]
    while st.active:
        try:
            frame = st.frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def home_page():
    """HOME — overview cards for all modules."""
    status = {name: streams[name].active for name in streams}
    return render_template("hub.html", status=status, active_nav="home", user_profile=_current_profile())


@app.route("/gate-security")
def gate_security_page():
    """Gate & Security = Vehicle Counter + ANPR + PWMU Shed Security."""
    return render_template("gate_security.html", active_nav="gate", user_profile=_current_profile())


@app.route("/ai-segregation")
def ai_segregation_page():
    """AI Segregation = Primary Segregation + Secondary Plastic Classification."""
    return render_template("ai_segregation.html", active_nav="segregation", user_profile=_current_profile())


@app.route("/dashboard")
def dashboard_page():
    """Dashboard = Analytics & Audit Reports."""
    return render_template("dashboard.html", active_nav="dashboard", user_profile=_current_profile())


@app.route("/tabs")
def index_tabs():
    """5-tab combined view (Gate/Stage1/Stage2/Shed/Summary) — kept as an
    alternate layout alongside the card-based hub."""
    status = {name: streams[name].active for name in streams}
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return render_template("command_center.html", status=status, now=now)


@app.route("/classic")
def classic_index():
    """Legacy dark-theme overview (old individual module pages)."""
    status = {name: streams[name].active for name in ["waste", "vehicle", "plate", "thief"]}
    return render_template("index.html", status=status)


@app.route("/waste")
def waste_page():
    p = get_processor("waste")
    return render_template("waste.html", active=streams["waste"].active,
                            model_status=p.status(), counts=p.get_counts())


@app.route("/vehicle")
def vehicle_page():
    p = get_processor("vehicle")
    return render_template(
        "module_vehicle.html",
        active=streams["vehicle"].active,
        counts=p.get_counts(),
        active_nav="gate",
        user_profile=_current_profile(),
    )


@app.route("/plate")
def plate_page():
    p = get_processor("plate")
    return render_template("plate.html", active=streams["plate"].active, recent=p.get_recent())


@app.route("/thief")
def thief_page():
    p = get_processor("thief")
    return render_template("thief.html", active=streams["thief"].active, recent=p.get_recent())


# ---------------------------------------------------------------------------
# Dedicated module pages (linked from the Hub) — Vehicle Counter and ANPR are
# fully SEPARATE modules/cameras here, as requested.
# ---------------------------------------------------------------------------
@app.route("/module/vehicle")
def module_vehicle():
    p = get_processor("vehicle")
    return render_template("module_vehicle.html", active=streams["vehicle"].active, counts=p.get_counts())


@app.route("/module/plate")
def module_plate():
    p = get_processor("plate")
    return render_template("module_plate.html", active=streams["plate"].active, recent=p.get_recent())


@app.route("/module/waste_primary")
def module_waste_primary():
    return render_template("module_waste_primary.html", active=streams["waste_primary"].active)


@app.route("/module/waste_secondary")
def module_waste_secondary():
    return render_template("module_waste_secondary.html", active=streams["waste_secondary"].active)


@app.route("/module/thief")
def module_thief():
    p = get_processor("thief")
    return render_template("module_thief.html", active=streams["thief"].active, recent=p.get_recent())


@app.route("/module/analytics")
def module_analytics():
    return render_template("module_analytics.html")


# ---------------------------------------------------------------------------
# Streaming + control API
# ---------------------------------------------------------------------------
@app.route("/video_feed/<module>")
def video_feed(module):
    if module not in streams:
        return "Unknown module", 404
    return Response(gen_frames(module), mimetype="multipart/x-mixed-replace; boundary=frame")


def _stop_stream(module, st):
    st.active = False
    st.stop_event.set()
    if st.worker_thread is not None:
        st.worker_thread.join(timeout=3.0)
    st.worker_thread = None
    # Drain any leftover frame so the next start doesn't serve a stale one
    while not st.frame_queue.empty():
        try:
            st.frame_queue.get_nowait()
        except queue.Empty:
            break


def _start_stream(module, st, source):
    st.source = source
    st.stop_event.clear()
    st.active = True
    st.worker_thread = threading.Thread(target=_capture_worker, args=(module, st), daemon=True)
    st.worker_thread.start()


@app.route("/api/toggle/<module>", methods=["POST"])
def toggle(module):
    if module not in streams:
        return jsonify({"error": "unknown module"}), 404
    st = streams[module]
    if st.active:
        _stop_stream(module, st)
    else:
        source = request.json.get("source") if request.is_json else None
        _start_stream(module, st, source if source else config.CAMERA_INDEX)
    return jsonify({"active": st.active})


@app.route("/api/upload/<module>", methods=["POST"])
def upload_video(module):
    if module not in streams:
        return jsonify({"error": "unknown module"}), 404
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "no file"}), 400

    save_path = os.path.join(config.UPLOADS_DIR, file.filename)
    file.save(save_path)

    st = streams[module]
    if st.active:
        _stop_stream(module, st)
    _start_stream(module, st, save_path)
    return jsonify({"active": True, "source": save_path})


@app.route("/api/reset/<module>", methods=["POST"])
def reset_counts(module):
    p = get_processor(module)
    if hasattr(p, "reset"):
        p.reset()
    if hasattr(p, "reset_counts"):
        p.reset_counts()
    return jsonify({"ok": True})


@app.route("/api/kpis")
def kpis():
    """Summary metrics for the Command Center KPI cards."""
    gate = get_processor("gate")
    wp = get_processor("waste_primary")
    ws = get_processor("waste_secondary")
    thief = get_processor("thief")

    gate_counts = gate.get_counts()
    vehicle_in = gate_counts.get("in", 0)
    vehicle_out = gate_counts.get("out", 0)

    # "Intake" here = total items detected across both waste conveyors this session.
    # NOTE: real tonnage needs a weigh-bridge/load-cell sensor feed — this counts
    # detected objects, which is a stand-in until that hardware integration exists.
    primary_total = sum(wp.get_counts().values())
    secondary_total = sum(ws.get_counts().values())
    total_items = primary_total + secondary_total

    revenue_total = 0.0
    for label, count in ws.get_counts().items():
        price = config.PLASTIC_PRICE_PER_KG.get(label.upper(), 5)
        revenue_total += count * config.ASSUMED_AVG_ITEM_WEIGHT_KG * price

    return jsonify({
        "total_items_detected": total_items,
        "vehicle_in": vehicle_in,
        "vehicle_out": vehicle_out,
        "processing_balance": vehicle_in - vehicle_out,
        "active_alerts": len(thief.get_recent()),
        "waste_composition": ws.get_counts(),
        "waste_primary_composition": wp.get_counts(),
        "estimated_revenue_inr": round(revenue_total, 2),
    })


@app.route("/api/audit_log")
def audit_log():
    """Merged recent events for the audit table — ANPR reads + theft/anomaly alerts.

    Prefers reading straight from Supabase (so the log survives app restarts and
    shows real history). Falls back to the current session's in-memory data if
    Supabase isn't configured — so the dashboard still works locally.
    """
    events = []

    plate_rows = supabase_client.fetch_recent("plate_detections", limit=15, order_col="detected_at")
    thief_rows = supabase_client.fetch_recent("thief_alerts", limit=15, order_col="detected_at")

    if plate_rows or thief_rows:
        for p in plate_rows:
            events.append({
                "type": "ANPR", "detail": p.get("plate_number", "—"),
                "time": _short_time(p.get("detected_at")), "status": _plate_status(p.get("confidence")),
                "image": p.get("image_url"),
            })
        for a in thief_rows:
            events.append({
                "type": "Security", "detail": a.get("reason", ""),
                "time": _short_time(a.get("detected_at")), "status": "Flagged",
                "image": a.get("image_url"),
            })
    else:
        # Local-only fallback (Supabase not configured, or empty so far)
        gate = get_processor("gate")
        thief = get_processor("thief")
        for p in gate.get_recent():
            events.append({
                "type": "ANPR", "detail": p.get("plate", "—"),
                "time": p.get("time", ""), "status": _plate_status(p.get("conf")),
                "image": p.get("url"),
            })
        for a in thief.get_recent():
            events.append({
                "type": "Security", "detail": a.get("reason", ""),
                "time": a.get("time", ""), "status": "Flagged",
                "image": a.get("url"),
            })

    events.sort(key=lambda e: e["time"], reverse=True)
    return jsonify({"events": events[:20]})


def _short_time(iso_ts):
    """Turn an ISO timestamp (from Supabase) into a HH:MM:SS display string."""
    if not iso_ts:
        return ""
    try:
        return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except Exception:
        return iso_ts


def _plate_status(conf):
    try:
        return "Verified" if float(conf) > 0.6 else "Flagged"
    except (TypeError, ValueError):
        return "Flagged"


@app.route("/api/manual_entry", methods=["POST"])
def manual_entry():
    """Operator override — manually log a plate reading the OCR missed.
    Vehicle Counter aur ANPR ab do alag modules hain, isliye yeh seedha
    standalone 'plate' processor me likhta hai (gate processor me nahi)."""
    data = request.get_json(force=True, silent=True) or {}
    plate = (data.get("plate") or "").strip()
    if not plate:
        return jsonify({"error": "plate number required"}), 400
    vehicle_type = data.get("vehicle_type", "—")
    direction = data.get("direction", "—")
    plate_processor = get_processor("plate")
    plate_processor.manual_entry(plate, vehicle_type, direction)
    return jsonify({"ok": True})


@app.route("/api/report/<module>")
def report(module):
    """Download a CSV report for one module, filtered by period (daily/weekly/monthly/all)."""
    period = request.args.get("period", "all")
    filename, csv_text = reports.build_csv_response(module, period)
    return Response(csv_text, mimetype="text/csv",
                     headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route("/api/report/full")
def report_full():
    """Download a combined CSV report across all modules for a given period."""
    period = request.args.get("period", "all")
    filename, csv_text = reports.build_full_report_csv(period)
    return Response(csv_text, mimetype="text/csv",
                     headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route("/api/report_pdf/gate")
def report_pdf_gate():
    """Gate PDF covers both vehicle counting AND ANPR in one document."""
    period = request.args.get("period", "all")
    filename, pdf_bytes = pdf_report.build_gate_pdf(period)
    return Response(pdf_bytes, mimetype="application/pdf",
                     headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route("/api/report_pdf/full")
def report_pdf_full():
    period = request.args.get("period", "all")
    filename, pdf_bytes = pdf_report.build_full_pdf(period)
    return Response(pdf_bytes, mimetype="application/pdf",
                     headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route("/api/report_pdf/<module>")
def report_pdf_module(module):
    period = request.args.get("period", "all")
    filename, pdf_bytes = pdf_report.build_module_pdf(module, period)
    return Response(pdf_bytes, mimetype="application/pdf",
                     headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route("/api/hourly/<module>")
def hourly(module):
    period = request.args.get("period", "daily")
    return jsonify(reports.hourly_breakdown(module, period))


@app.route("/api/download/video/<module>")
def download_video(module):
    """Download the most recently recorded annotated video for a module
    (available once a live session or uploaded-video run has finished)."""
    st = streams.get(module)
    if not st or not st.output_path or not os.path.exists(st.output_path):
        return jsonify({"error": "no recording available yet — start & stop a session first"}), 404
    return send_file(st.output_path, as_attachment=True)


@app.route("/api/plastic_revenue")
def plastic_revenue():
    """Rough estimated resale revenue based on detected plastic counts.
    NOTE: this is an ESTIMATE — assumes config.ASSUMED_AVG_ITEM_WEIGHT_KG per
    item since there's no weighing-scale sensor feed yet."""
    ws = get_processor("waste_secondary")
    counts = ws.get_counts()
    breakdown = {}
    total = 0.0
    for label, count in counts.items():
        price = config.PLASTIC_PRICE_PER_KG.get(label.upper(), 5)
        est_kg = count * config.ASSUMED_AVG_ITEM_WEIGHT_KG
        est_revenue = round(est_kg * price, 2)
        breakdown[label] = {"count": count, "est_kg": round(est_kg, 3), "est_revenue_inr": est_revenue}
        total += est_revenue
    return jsonify({"breakdown": breakdown, "total_est_revenue_inr": round(total, 2),
                     "disclaimer": "Estimated from detection counts, not a calibrated weighing scale."})


@app.route("/api/counts/<module>")
def counts(module):
    p = get_processor(module)
    if module in ("waste", "waste_primary", "waste_secondary"):
        return jsonify(p.get_counts())
    if module in ("vehicle", "gate"):
        return jsonify(p.get_counts())
    if module == "plate":
        return jsonify({"recent": p.get_recent()})
    if module == "thief":
        return jsonify({"recent": p.get_recent()})
    return jsonify({})


def _shutdown_all_streams():
    """Graceful shutdown: stop every active capture thread, release every
    camera/VideoWriter cleanly, so re-running the app never fights over a
    camera device that was left open by a previous crashed/killed process."""
    for module_name, st in streams.items():
        if st.active or st.worker_thread is not None:
            print(f"[SHUTDOWN] stopping {module_name}...")
            try:
                _stop_stream(module_name, st)
            except Exception as e:
                print(f"[SHUTDOWN] error stopping {module_name}: {e}")
    print("[SHUTDOWN] all camera resources released.")


atexit.register(_shutdown_all_streams)


def _handle_signal(signum, frame):
    _shutdown_all_streams()
    raise SystemExit(0)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    # use_reloader=False is important here: the default Werkzeug auto-reloader
    # starts a 2nd watcher process alongside the real one. With OpenCV camera
    # threads + YOLO models in play, that 2nd process can end up fighting for
    # the same camera device / GPU memory, which is a classic cause of a
    # dashboard feeling slow/laggy in the browser even though a plain script
    # in VS Code (no reloader) runs the same model smoothly.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
