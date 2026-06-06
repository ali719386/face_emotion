import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import base64
import json
import sqlite3
import tempfile
import secrets
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from fusion import FusionEngine, normalize_scores

APP_NAME = "EmotionSense"
AI_AGENT_URL = "https://nic-fsd-testing-agent.vercel.app/"
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "webm", "m4a"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DEFAULT_WEIGHTS = {"face": 0.4, "voice": 0.3, "text": 0.3}
INPUT_MODE_WEIGHTS = {
    "all": DEFAULT_WEIGHTS,
    "image": {"face": 1.0, "voice": 0.0, "text": 0.0},
    "voice": {"face": 0.0, "voice": 1.0, "text": 0.0},
    "text": {"face": 0.0, "voice": 0.0, "text": 1.0},
}

MODEL_CACHE = {}

INPUT_MODES = [
    {
        "value": "all",
        "label": "All Inputs",
        "title": "Use image, voice, and text together",
        "description": "Best when you want the strongest result from multiple signals.",
    },
    {
        "value": "image",
        "label": "Image",
        "title": "Analyze from face image",
        "description": "Use the camera or upload a photo.",
    },
    {
        "value": "voice",
        "label": "Voice",
        "title": "Analyze from voice",
        "description": "Upload audio or record directly in the browser.",
    },
    {
        "value": "text",
        "label": "Text",
        "title": "Analyze from text",
        "description": "Type a message describing how you feel.",
    },
]

MODE_PAGE_CONTENT = {
    "image": {
        "eyebrow": "Image analysis",
        "title": "Check emotion from an image or live camera frame.",
        "description": "Upload a face photo or capture a camera frame for facial emotion detection.",
    },
    "voice": {
        "eyebrow": "Voice analysis",
        "title": "Check emotion from a voice recording.",
        "description": "Upload a short audio file or record directly in the browser.",
    },
    "text": {
        "eyebrow": "Text analysis",
        "title": "Check emotion from written text.",
        "description": "Type a sentence or paragraph about how you feel.",
    },
    "all": {
        "eyebrow": "Combined analysis",
        "title": "Use image, voice, and text together.",
        "description": "Combine all signals for a fuller emotion analysis result.",
    },
}

HOW_IT_WORKS = [
    {
        "step": "01",
        "title": "Choose one input style",
        "description": "Select image, voice, text, or all inputs from one simple menu.",
    },
    {
        "step": "02",
        "title": "Add your content",
        "description": "Upload a photo, record your voice, or type a short message in plain language.",
    },
    {
        "step": "03",
        "title": "See the result fast",
        "description": "Run the analysis and get emotion detection with simple support suggestions.",
    },
]

EMOTION_SUPPORT = {
    "sad": {
        "headline": "A softer routine can help steady the mood.",
        "summary": "Try a spiritual reset, comforting audio, and a guided AI check-in.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Recite Quran",
                "description": "Open a calm recitation or read a short Surah to slow the pace of the moment.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Comfort audio",
                "title": "Play gentle songs",
                "description": "Use a low-stimulation playlist with soft vocals or instrumentals.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Talk to an AI agent",
                "description": "Use a structured prompt to unpack the feeling and get small next steps.",
                "prompt": "I feel sad today. Help me calm down with three simple steps and one reflective question.",
            },
        ],
    },
    "happy": {
        "headline": "This is a good state to reinforce.",
        "summary": "Turn the energy into gratitude, celebration, and productive momentum.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Read a gratitude passage",
                "description": "Pair the positive emotion with gratitude and reflection.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Celebration audio",
                "title": "Play upbeat songs",
                "description": "Use music to sustain motivation without losing focus.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Plan the next win",
                "description": "Ask the AI agent to turn this high-energy state into a focused plan.",
                "prompt": "I am feeling happy and energized. Help me plan three productive tasks for today.",
            },
        ],
    },
    "joy": {
        "headline": "Use the positive momentum while it is present.",
        "summary": "Anchor it with gratitude, music, and a forward-looking plan.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Reflect with gratitude",
                "description": "Take a short gratitude pause and note what created the joyful moment.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Celebration audio",
                "title": "Play a bright playlist",
                "description": "Use a positive playlist to extend the mood in a healthy way.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Journal the moment",
                "description": "Let the AI help capture what went right and how to repeat it.",
                "prompt": "I feel joyful right now. Help me write a short reflection about what went well today.",
            },
        ],
    },
    "angry": {
        "headline": "Reduce intensity before making decisions.",
        "summary": "Shift into grounding, slower audio, and guided de-escalation.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Listen to recitation",
                "description": "Choose a slow recitation and step away from the trigger for a few minutes.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Reset audio",
                "title": "Switch to calm tracks",
                "description": "Avoid high-energy songs and use slower breathing-friendly audio.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "De-escalation coach",
                "description": "Use the AI agent to organize the situation before responding.",
                "prompt": "I am angry. Help me calm down first, then help me respond without making things worse.",
            },
        ],
    },
    "fear": {
        "headline": "Reduce uncertainty with grounding and structure.",
        "summary": "Pair reassurance with calm audio and step-by-step thinking.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Read reassuring verses",
                "description": "Use recitation and reflection to bring attention back to what is stable.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Calming audio",
                "title": "Play slower songs",
                "description": "Choose low-tempo tracks that do not add extra tension.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Fear breakdown",
                "description": "Ask the AI to separate real risk from imagined escalation.",
                "prompt": "I am feeling afraid. Help me identify what I can control, what I cannot, and what to do next.",
            },
        ],
    },
    "anxious": {
        "headline": "Bring the system down one layer at a time.",
        "summary": "Use calm recitation, low-pressure audio, and an AI grounding prompt.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Read a calming Surah",
                "description": "A short recitation can interrupt spiraling thoughts.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Calming audio",
                "title": "Play low-pressure songs",
                "description": "Choose tracks that are repetitive, slow, and not emotionally intense.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Grounding conversation",
                "description": "Use the AI to slow the thought loop and identify one immediate action.",
                "prompt": "I feel anxious. Walk me through a 2-minute grounding routine and one practical next step.",
            },
        ],
    },
    "surprise": {
        "headline": "Unexpected energy needs context.",
        "summary": "Pause, interpret the event, and decide whether the surprise is positive or disruptive.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Reflect before reacting",
                "description": "Take a short pause and use reflection to slow the first reaction.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Mood audio",
                "title": "Choose balanced songs",
                "description": "Use neutral or uplifting music while you process the event.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Interpret the moment",
                "description": "Use the AI to think through what happened and how to respond well.",
                "prompt": "Something surprised me today. Help me interpret it and choose a smart response.",
            },
        ],
    },
    "neutral": {
        "headline": "A neutral state is useful for focus and clarity.",
        "summary": "Keep it steady with structure, reflection, and light support options.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Read a short passage",
                "description": "Use a short reflective reading to maintain steadiness.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Focus audio",
                "title": "Play background songs",
                "description": "Use light music that supports work without pulling attention away.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Build the day plan",
                "description": "Use the AI agent to turn a steady mood into a focused routine.",
                "prompt": "I feel neutral and stable. Help me build a clean plan for the next few hours.",
            },
        ],
    },
    "disgust": {
        "headline": "Create distance from the trigger first.",
        "summary": "Step back, reset sensory overload, and process the situation calmly.",
        "cards": [
            {
                "label": "Spiritual support",
                "title": "Pause with recitation",
                "description": "Use a quiet break and reflective reading to regain composure.",
                "cta": "Open Quran",
                "url": "https://quran.com/",
            },
            {
                "label": "Reset audio",
                "title": "Switch to calm sounds",
                "description": "Move to audio that reduces sensory intensity rather than increasing it.",
                "cta": "Open music",
                "url": "https://open.spotify.com/",
            },
            {
                "label": "AI companion",
                "title": "Process the reaction",
                "description": "Use the AI to understand the trigger and decide how to move on.",
                "prompt": "I feel disgusted by something that happened. Help me process the reaction calmly and move forward.",
            },
        ],
    },
}


def create_app():
    app = Flask(__name__)
    os.makedirs(app.instance_path, exist_ok=True)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or get_or_create_secret_key(app.instance_path)
    app.config["DATABASE"] = os.path.join(app.instance_path, "emotionsense.db")
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    @app.before_request
    def load_logged_in_user():
        user_id = session.get("user_id")
        g.user = get_user_by_id(user_id) if user_id else None

    @app.before_request
    def protect_post_requests():
        if request.method != "POST":
            return None

        sent_token = request.form.get("csrf_token", "")
        session_token = session.get("_csrf_token", "")
        if not session_token or not sent_token or not secrets.compare_digest(sent_token, session_token):
            flash("Your session expired. Please try again.", "error")
            return redirect(request.referrer or url_for("landing"))
        return None

    @app.context_processor
    def inject_globals():
        return {
            "app_name": APP_NAME,
            "current_year": datetime.utcnow().year,
            "csrf_token": generate_csrf_token,
        }

    @app.route("/favicon.ico")
    def favicon():
        return redirect(url_for("static", filename="favicon.svg"))

    @app.route("/")
    def landing():
        if g.user:
            return redirect(url_for("dashboard"))
        return render_template("landing.html", how_it_works=HOW_IT_WORKS)

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if g.user:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not name or not email or not password:
                flash("Name, email, and password are required.", "error")
            elif password != confirm_password:
                flash("Passwords do not match.", "error")
            elif get_user_by_email(email):
                flash("That email is already registered.", "error")
            else:
                user_id = create_user(name, email, password)
                session["user_id"] = user_id
                flash("Account created successfully.", "success")
                return redirect(url_for("dashboard"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = get_user_by_email(email)

            if user is None or not check_password_hash(user["password_hash"], password):
                flash("Invalid email or password.", "error")
            else:
                session.clear()
                session["user_id"] = user["id"]
                flash("Welcome back.", "success")
                return redirect(url_for("dashboard"))

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been signed out.", "success")
        return redirect(url_for("landing"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template(
            "dashboard.html",
            **build_workspace_context(
                current_page="overview",
                page_title="Dashboard",
                page_eyebrow="Overview",
            ),
        )

    @app.route("/dashboard/results")
    @login_required
    def dashboard_results():
        return render_template(
            "results.html",
            **build_workspace_context(
                current_page="results",
                page_title="Results",
                page_eyebrow="Latest analysis",
            ),
        )

    @app.route("/dashboard/analyze/<mode>")
    @login_required
    def dashboard_mode(mode):
        if mode not in MODE_PAGE_CONTENT:
            return redirect(url_for("dashboard"))

        return render_template(
            "analyzer_page.html",
            **build_workspace_context(
                current_page=mode,
                page_title=f"{get_input_mode_label(mode)} Analysis",
                page_eyebrow=MODE_PAGE_CONTENT[mode]["eyebrow"],
                analyzer_mode=mode,
                mode_content=MODE_PAGE_CONTENT[mode],
                mode_label=get_input_mode_label(mode),
                submit_label=f"Analyze {get_input_mode_label(mode).lower()}",
            ),
        )

    @app.route("/analyze", methods=["POST"])
    @login_required
    def analyze():
        input_mode = request.form.get("input_mode", "all").strip().lower()
        if input_mode not in INPUT_MODE_WEIGHTS:
            input_mode = "all"

        text_input = request.form.get("text_input", "").strip()
        camera_snapshot = request.form.get("camera_snapshot", "").strip()
        face_image = request.files.get("face_image")

        audio_file = request.files.get("voice_file")
        recorded_audio = request.files.get("voice_recording")
        chosen_audio = choose_audio_source(audio_file, recorded_audio)
        chosen_face = choose_face_source(camera_snapshot, face_image)
        session["selected_input_mode"] = input_mode

        if input_mode == "image":
            chosen_audio = None
            text_input = ""
        elif input_mode == "voice":
            chosen_face = None
            text_input = ""
        elif input_mode == "text":
            chosen_face = None
            chosen_audio = None

        if input_mode == "all":
            weights = normalize_weights(
                {
                    "face": request.form.get("face_weight", DEFAULT_WEIGHTS["face"]),
                    "voice": request.form.get("voice_weight", DEFAULT_WEIGHTS["voice"]),
                    "text": request.form.get("text_weight", DEFAULT_WEIGHTS["text"]),
                }
            )
        else:
            weights = INPUT_MODE_WEIGHTS[input_mode].copy()

        validation_error = validate_analysis_input(input_mode, text_input, chosen_audio, chosen_face)
        if validation_error:
            flash(validation_error, "error")
            return redirect(get_analysis_redirect_url(input_mode))

        text_result, text_issue = predict_text(text_input)
        voice_result, voice_issue = predict_voice(chosen_audio)
        face_result, face_issue = predict_face(chosen_face)

        active_modalities = set()
        # Only count a modality as active if it produced a real (non-fallback) result
        if chosen_face is not None and face_result.get("label") not in (None, "neutral") or (
            chosen_face is not None and not face_issue
        ):
            active_modalities.add("face")
        if chosen_audio is not None and not voice_issue:
            active_modalities.add("voice")
        if text_input:
            active_modalities.add("text")
        # Fall back to treating all as active if nothing provided (validation catches this earlier)
        if not active_modalities:
            active_modalities = None
        fused = FusionEngine(weights).fuse(face_result, voice_result, text_result, active_modalities)
        sorted_scores = sort_scores(fused["scores"])
        dominant_percent = sorted_scores[0]["probability"] if sorted_scores else 0.0
        support = build_support_plan(fused["label"])
        issues = [issue for issue in (text_issue, voice_issue, face_issue) if issue]
        modality_details = build_modality_details(face_result, voice_result, text_result)
        fused_confidence = build_confidence_summary(fused["scores"])

        analysis = {
            "fused": {
                "label": fused["label"],
                "dominant_percent": dominant_percent,
                "scores": sorted_scores,
                "confidence": fused_confidence,
            },
            "modalities": {
                "face": face_result["label"],
                "voice": voice_result["label"],
                "text": text_result["label"],
            },
            "modality_details": modality_details,
            "weights": weights,
            "support": support,
            "issues": issues,
            "input_mode": input_mode,
            "input_mode_label": get_input_mode_label(input_mode),
            "used_inputs": get_used_inputs(chosen_face, chosen_audio, text_input),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

        session["last_analysis"] = analysis
        log_analysis(g.user["id"], analysis)
        flash("Emotion analysis completed.", "success")
        return redirect(url_for("dashboard_results"))

    def get_db_connection():
        connection = sqlite3.connect(app.config["DATABASE"])
        connection.row_factory = sqlite3.Row
        return connection

    def init_db():
        with get_db_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    dominant_emotion TEXT NOT NULL,
                    dominant_percent REAL NOT NULL,
                    text_label TEXT NOT NULL,
                    voice_label TEXT NOT NULL,
                    face_label TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
                """
            )

    def get_user_by_email(email):
        if not email:
            return None
        with get_db_connection() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()

    def get_user_by_id(user_id):
        if not user_id:
            return None
        with get_db_connection() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

    def create_user(name, email, password):
        created_at = datetime.utcnow().isoformat()
        password_hash = generate_password_hash(password)
        with get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, password_hash, created_at),
            )
            connection.commit()
            return cursor.lastrowid

    def log_analysis(user_id, analysis):
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    user_id,
                    dominant_emotion,
                    dominant_percent,
                    text_label,
                    voice_label,
                    face_label,
                    weights_json,
                    scores_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    analysis["fused"]["label"],
                    analysis["fused"]["dominant_percent"],
                    analysis["modalities"]["text"],
                    analysis["modalities"]["voice"],
                    analysis["modalities"]["face"],
                    json.dumps(analysis["weights"]),
                    json.dumps(analysis["fused"]["scores"]),
                    datetime.utcnow().isoformat(),
                ),
            )
            connection.commit()

    def get_recent_analyses(user_id, limit=5):
        with get_db_connection() as connection:
            rows = connection.execute(
                """
                SELECT dominant_emotion, dominant_percent, text_label, voice_label, face_label, created_at
                FROM analyses
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        items = []
        for row in rows:
            items.append(
                {
                    "dominant_emotion": row["dominant_emotion"],
                    "dominant_percent": round(row["dominant_percent"], 2),
                    "text_label": row["text_label"],
                    "voice_label": row["voice_label"],
                    "face_label": row["face_label"],
                    "created_at": format_timestamp(row["created_at"]),
                }
            )
        return items

    def build_workspace_context(current_page, page_title, page_eyebrow, **extra):
        context = {
            "current_page": current_page,
            "page_title_text": page_title,
            "page_eyebrow": page_eyebrow,
            "how_it_works": HOW_IT_WORKS,
            "last_analysis": hydrate_analysis(session.get("last_analysis")),
            "recent_analyses": get_recent_analyses(g.user["id"]),
            "default_weights": DEFAULT_WEIGHTS,
            "input_modes": INPUT_MODES,
            "selected_input_mode": session.get("selected_input_mode", "all"),
            "nav_items": [
                {"key": "overview", "label": "Overview", "href": url_for("dashboard")},
                {"key": "image", "label": "Image Page", "href": url_for("dashboard_mode", mode="image")},
                {"key": "voice", "label": "Voice Page", "href": url_for("dashboard_mode", mode="voice")},
                {"key": "text", "label": "Text Page", "href": url_for("dashboard_mode", mode="text")},
                {"key": "all", "label": "All Inputs", "href": url_for("dashboard_mode", mode="all")},
                {"key": "results", "label": "Results", "href": url_for("dashboard_results")},
            ],
        }
        context.update(extra)
        return context

    with app.app_context():
        init_db()
        # Preload models at startup so first request is fast
    print("[INFO] Preloading text model...")
    get_text_model()
    print("[INFO] Models ready.")

    return app


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please sign in first.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def get_or_create_secret_key(instance_path):
    secret_path = os.path.join(instance_path, "secret_key.txt")
    if os.path.exists(secret_path):
        with open(secret_path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
            if value:
                return value

    value = secrets.token_urlsafe(48)
    with open(secret_path, "w", encoding="utf-8") as handle:
        handle.write(value)
    return value


def generate_csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def get_text_model():
    if "text" not in MODEL_CACHE:
        from text_emotion import TextEmotion

        MODEL_CACHE["text"] = TextEmotion()
    return MODEL_CACHE["text"]


def get_voice_model():
    if "voice" not in MODEL_CACHE:
        from voice_emotion import VoiceEmotion

        MODEL_CACHE["voice"] = VoiceEmotion()
    return MODEL_CACHE["voice"]


def get_face_model():
    if "face" not in MODEL_CACHE:
        from face_emotion import FaceEmotion

        MODEL_CACHE["face"] = FaceEmotion()
    return MODEL_CACHE["face"]


def predict_text(text_input):
    neutral = {"label": "neutral", "scores": {"neutral": 1.0}}
    if not text_input:
        return neutral, None

    try:
        return get_text_model().predict(text_input), None
    except Exception as exc:
        return neutral, f"Text model fallback used: {exc}"


def predict_voice(audio_storage):
    neutral = {"label": "neutral", "scores": {"neutral": 1.0}}
    if audio_storage is None:
        return neutral, None

    filename = secure_filename(audio_storage.filename or "voice.wav")
    suffix = os.path.splitext(filename)[1] or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = handle.name
            audio_storage.save(handle)

        if get_voice_model().model is None:
            return neutral, "Voice model not available."

        return get_voice_model().predict_from_file(temp_path), None
    except Exception as exc:
        return neutral, f"Voice model fallback used: {exc}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def predict_face(face_source):
    neutral = {"label": "neutral", "scores": {"neutral": 1.0}}
    if face_source is None:
        return neutral, None

    if face_source["type"] == "snapshot":
        frame = decode_camera_snapshot(face_source["value"])
    else:
        frame = decode_uploaded_image(face_source["value"])

    if frame is None:
        return neutral, "Image could not be read. Try another photo or capture again."

    try:
        result = get_face_model().predict_from_frame(frame)
        if result.get("label") is None:
            # Low-confidence detection — face not clear enough; treat as no face signal
            return neutral, "Face not clearly detected. Try better lighting or a closer shot."
        return result, None
    except Exception as exc:
        return neutral, f"Face model fallback used: {exc}"


def decode_camera_snapshot(camera_snapshot):
    try:
        if "," not in camera_snapshot:
            return None
        _, encoded = camera_snapshot.split(",", 1)
        raw = base64.b64decode(encoded)
        frame_bytes = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def decode_uploaded_image(image_storage):
    try:
        raw = image_storage.read()
        image_storage.stream.seek(0)
        frame_bytes = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def choose_face_source(camera_snapshot, face_image):
    if camera_snapshot:
        return {"type": "snapshot", "value": camera_snapshot}

    if face_image and face_image.filename:
        extension = os.path.splitext(face_image.filename)[1].lower().lstrip(".")
        if extension in ALLOWED_IMAGE_EXTENSIONS:
            return {"type": "upload", "value": face_image}

    return None


def choose_audio_source(audio_file, recorded_audio):
    for candidate in (recorded_audio, audio_file):
        if candidate and candidate.filename:
            extension = os.path.splitext(candidate.filename)[1].lower().lstrip(".")
            if extension in ALLOWED_AUDIO_EXTENSIONS:
                return candidate
    return None


def validate_analysis_input(input_mode, text_input, chosen_audio, chosen_face):
    if input_mode == "text" and not text_input:
        return "Please type a message before running text analysis."
    if input_mode == "voice" and chosen_audio is None:
        return "Please upload or record audio before running voice analysis."
    if input_mode == "image" and chosen_face is None:
        return "Please capture a photo or upload an image before running image analysis."
    if input_mode == "all" and not any([text_input, chosen_audio, chosen_face]):
        return "Please provide at least one input: image, voice, or text."
    return None


def normalize_weights(raw_weights):
    parsed = {}
    total = 0.0
    for key, raw_value in raw_weights.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = DEFAULT_WEIGHTS[key]
        value = max(0.0, min(1.0, value))
        parsed[key] = value
        total += value

    if total <= 0:
        return DEFAULT_WEIGHTS.copy()

    return {key: round(value / total, 4) for key, value in parsed.items()}


def sort_scores(scores):
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "emotion": emotion.replace("_", " ").title(),
            "probability": round(score * 100, 2),
        }
        for emotion, score in ranked
    ]


def build_confidence_summary(scores):
    normalized = normalize_scores(scores)
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    top_emotion, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - runner_up_score

    if top_score >= 0.7 and margin >= 0.2:
        level = "High"
    elif top_score >= 0.45 and margin >= 0.1:
        level = "Medium"
    else:
        level = "Low"

    return {
        "level": level,
        "top_emotion": top_emotion.replace("_", " ").title(),
        "score_percent": round(top_score * 100, 2),
        "margin_percent": round(margin * 100, 2),
    }


def build_modality_details(face_result, voice_result, text_result):
    details = {}
    for key, result in {
        "face": face_result,
        "voice": voice_result,
        "text": text_result,
    }.items():
        confidence = build_confidence_summary(result.get("scores", {}))
        details[key] = {
            "label": result.get("label", "neutral").replace("_", " ").title(),
            "confidence_level": confidence["level"],
            "confidence_percent": confidence["score_percent"],
        }
    return details


def build_support_plan(emotion_label):
    emotion_key = (emotion_label or "neutral").lower()
    plan = EMOTION_SUPPORT.get(emotion_key, EMOTION_SUPPORT["neutral"])
    cards = []
    for card in plan["cards"]:
        item = dict(card)
        if item.get("prompt"):
            item["agent_url"] = AI_AGENT_URL
            item["agent_cta"] = "Open AI Agent"
        cards.append(item)
    return {
        "emotion": emotion_key.title(),
        "headline": plan["headline"],
        "summary": plan["summary"],
        "cards": cards,
    }


def get_input_mode_label(input_mode):
    for mode in INPUT_MODES:
        if mode["value"] == input_mode:
            return mode["label"]
    return "All Inputs"


def get_used_inputs(chosen_face, chosen_audio, text_input):
    used_inputs = []
    if chosen_face is not None:
        used_inputs.append("Image")
    if chosen_audio is not None:
        used_inputs.append("Voice")
    if text_input:
        used_inputs.append("Text")
    return used_inputs


def get_analysis_redirect_url(input_mode):
    if input_mode in MODE_PAGE_CONTENT:
        return url_for("dashboard_mode", mode=input_mode)
    return url_for("dashboard_mode", mode="all")


def hydrate_analysis(analysis):
    if not analysis:
        return analysis

    fused = analysis.setdefault("fused", {})
    fused_scores = {
        item["emotion"].replace(" ", "_").lower(): item["probability"] / 100.0
        for item in fused.get("scores", [])
        if isinstance(item, dict) and "emotion" in item and "probability" in item
    }
    if "confidence" not in fused:
        fused["confidence"] = build_confidence_summary(fused_scores)

    if "modality_details" not in analysis:
        modalities = analysis.get("modalities", {})
        analysis["modality_details"] = {
            "face": {
                "label": str(modalities.get("face", "neutral")).replace("_", " ").title(),
                "confidence_level": "Unknown",
                "confidence_percent": 0.0,
            },
            "voice": {
                "label": str(modalities.get("voice", "neutral")).replace("_", " ").title(),
                "confidence_level": "Unknown",
                "confidence_percent": 0.0,
            },
            "text": {
                "label": str(modalities.get("text", "neutral")).replace("_", " ").title(),
                "confidence_level": "Unknown",
                "confidence_percent": 0.0,
            },
        }

    return analysis


def format_timestamp(timestamp_text):
    try:
        parsed = datetime.fromisoformat(timestamp_text)
        return parsed.strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return timestamp_text


app = create_app()
app.jinja_env.globals["csrf_token"] = generate_csrf_token


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", "8003"))
    debug_enabled = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug_enabled, use_reloader=False, port=port)
