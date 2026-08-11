from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
import sqlite3
from functools import lru_cache
from datetime import datetime

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
    redirect,
    url_for,
    flash,
)
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import numpy as np
import xgboost as xgb

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage

from huggingface_hub import hf_hub_download

# Gemini
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

# Hugging Face model repository
HF_REPO_ID = "Ansh2005A/agronova_models"

# IMPORTANT:
# These filenames must exactly match the files in your
# Hugging Face repository.
HF_MODELS = {
    "price": "xgb_price_model.json",
    "soil": "soil_type_resnet50_finetuned.h5",
    "nut_quality": "arecanut_quality_effnet_initial.h5",
    "leaf": "leaf_disease_resnet50_finetuned.h5",
    "trunk": "trunk_disease_resnet50_finetuned.h5",
    "crop": "nut_disease_resnet50_finetuned.h5",
}

# Secrets MUST be stored as environment variables.
FLASK_SECRET_KEY = os.environ.get(
    "FLASK_SECRET_KEY",
    "development-secret-change-this"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

DB_PATH = os.environ.get("DB_PATH", "users.db")


# ============================================================
# FLASK INITIALIZATION
# ============================================================

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
DB_PATH = "users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ============================================================
# GEMINI INITIALIZATION
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client initialized.")
    except Exception as e:
        print("Gemini initialization failed:", e)
else:
    print("GEMINI_API_KEY not configured.")


# ============================================================
# HUGGING FACE MODEL DOWNLOADER
# ============================================================

@lru_cache(maxsize=None)
def download_model(filename):
    """
    Download a model from Hugging Face and return
    the local cached file path.

    Hugging Face handles caching automatically.
    """

    try:
        print(f"Downloading/checking model: {filename}")

        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
        )

        print(f"Model available at: {path}")

        return path

    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        raise


# ============================================================
# MODEL LOADERS
# ============================================================

@lru_cache(maxsize=1)
def get_price_model():

    try:
        model_path = download_model(HF_MODELS["price"])

        model = xgb.XGBRegressor()
        model.load_model(model_path)

        print("Price model loaded from Hugging Face.")

        return model

    except Exception as e:
        print("Price model loading failed:", e)
        return None


@lru_cache(maxsize=1)
def get_soil_model():

    try:
        model_path = download_model(HF_MODELS["soil"])

        model = load_model(model_path)

        print("Soil model loaded from Hugging Face.")

        return model

    except Exception as e:
        print("Soil model loading failed:", e)
        return None


@lru_cache(maxsize=1)
def get_leaf_model():

    try:
        model_path = download_model(HF_MODELS["leaf"])

        model = load_model(model_path)

        print("Leaf disease model loaded.")

        return model

    except Exception as e:
        print("Leaf model loading failed:", e)
        return None


@lru_cache(maxsize=1)
def get_trunk_model():

    try:
        model_path = download_model(HF_MODELS["trunk"])

        model = load_model(model_path)

        print("Trunk disease model loaded.")

        return model

    except Exception as e:
        print("Trunk model loading failed:", e)
        return None


@lru_cache(maxsize=1)
def get_crop_model():

    try:
        model_path = download_model(HF_MODELS["crop"])

        model = load_model(model_path)

        print("Crop disease model loaded.")

        return model

    except Exception as e:
        print("Crop model loading failed:", e)
        return None


@lru_cache(maxsize=1)
def get_nut_quality_model():

    try:
        model_path = download_model(HF_MODELS["nut_quality"])

        model = load_model(model_path)

        print("Nut quality model loaded.")

        return model

    except Exception as e:
        print("Nut quality model loading failed:", e)
        return None


# ============================================================
# PRELOAD ALL MODELS AT STARTUP
# ============================================================
# Since all loaders above are wrapped with @lru_cache, calling
# them once here forces the download + load to happen right
# now (at app startup / import time) instead of on the first
# incoming request. Later calls anywhere else in the code just
# return the cached model instantly.

print("Preloading all models at startup, please wait...")

get_price_model()
get_soil_model()
get_leaf_model()
get_trunk_model()
get_crop_model()
get_nut_quality_model()

print("All models preloaded.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_soil_image(img_path):

    img = kimage.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = kimage.img_to_array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


def preprocess_disease_image(img_path):

    img = kimage.load_img(
        img_path,
        target_size=(224, 224)
    )

    arr = kimage.img_to_array(img) / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


# ============================================================
# DATABASE
# ============================================================

def get_db():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# AUTH ROUTES
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if not username or not email or not password:

            flash(
                "All fields are required.",
                "error"
            )

            return redirect(url_for("signup"))

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

        if cur.fetchone():

            flash(
                "Username already exists.",
                "error"
            )

            conn.close()

            return redirect(url_for("login"))

        cur.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        )

        if cur.fetchone():

            flash(
                "Email already registered.",
                "error"
            )

            conn.close()

            return redirect(url_for("login"))

        hashed = generate_password_hash(password)

        cur.execute(
            """
            INSERT INTO users
            (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (
                username,
                email,
                hashed
            )
        )

        conn.commit()
        conn.close()

        flash(
            "Signup successful! Please login.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

        user = cur.fetchone()

        conn.close()

        if (
            user is None
            or not check_password_hash(
                user["password_hash"],
                password
            )
        ):

            flash(
                "Incorrect username or password.",
                "error"
            )

            return redirect(url_for("login"))

        session["logged_in"] = True
        session["username"] = user["username"]
        session["email"] = user["email"]

        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))


def login_required(f):

    def wrap(*args, **kwargs):

        if "logged_in" in session:

            return f(*args, **kwargs)

        flash(
            "You need to login first.",
            "error"
        )

        return redirect(url_for("login"))

    wrap.__name__ = f.__name__

    return wrap


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template("home.html")


@app.route("/price_prediction")
@login_required
def price_prediction_page():

    return render_template(
        "price_prediction.html"
    )


@app.route("/soil_analysis")
@login_required
def soil_analysis_page():

    return render_template(
        "soil_analysis.html"
    )


@app.route("/disease_detection")
@login_required
def disease_detection_page():

    return render_template(
        "disease_detection.html"
    )


@app.route("/expert_chatbot")
@login_required
def expert_chatbot_page():

    return render_template(
        "expert_chatbot.html"
    )


# ============================================================
# SOIL CLASSES
# ============================================================

SOIL_CLASSES = [
    "Laterite_Soil",
    "Alluvial_soil",
    "Red_soil",
    "Black_soil",
]


SOIL_SUITABILITY = {

    "Black_soil": (
        70,
        85
    ),

    "Laterite_Soil": (
        40,
        60
    ),

    "Red_soil": (
        55,
        75
    ),

    "Alluvial_soil": (
        85,
        95
    ),
}


# ============================================================
# DISEASE CLASSES
# ============================================================

LEAF_CLASSES = [
    "Healthy_leaf",
    "Yellow_leaf_disease",
]

TRUNK_CLASSES = [
    "Healthy_trunk",
    "stem_bleeding",
    "stem_crack",
]

CROP_CLASSES = [
    "Healthy_crop",
    "Mahali_koleroga",
]

NUT_CLASSES = [
    "GradeA",
    "GradeB",
    "GradeC",
]


# ============================================================
# PRICE TEMPLATE
# ============================================================

price_template = pd.DataFrame({

    "Year": [0],
    "Month": [0],
    "Day": [0],

    "Bangalore": [0],
    "Belgaum": [0],
    "Chamrajnagar": [0],
    "Chikmagalur": [0],
    "Chitradurga": [0],
    "Davangere": [0],
    "Hassan": [0],
    "Haveri": [0],
    "Karwar": [0],
    "Kolar": [0],
    "Madikeri": [0],
    "Mandya": [0],
    "Mangalore": [0],
    "Mysore": [0],
    "Shimoga": [0],
    "Tumkur": [0],
    "Udupi": [0],

    "Bette": [0],
    "Bilegotu": [0],
    "Chali": [0],
    "Chippu": [0],
    "Churu": [0],
    "Cqca": [0],
    "EDI": [0],
    "Factory": [0],
    "Gorabalu": [0],
    "Kempugotu": [0],
    "Kole": [0],
    "New Variety": [1],
    "Other": [0],
    "Pudi": [0],
    "Pylone": [0],
    "Rashi": [0],
    "Raw": [0],
    "Red": [0],
    "Ripe": [0],
    "Saraku": [0],
    "Sippegotu": [0],
    "Supari": [0],
    "Tattibettee": [0],

    "api": [0],
})


# ============================================================
# PRICE PREDICTION API
# ============================================================

@app.route(
    "/api/predict_price",
    methods=["POST"]
)
def api_predict_price():

    price_model = get_price_model()

    if price_model is None:

        return jsonify({
            "error": "Price model could not be loaded from Hugging Face."
        }), 500

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "Invalid JSON"
        }), 400

    district = data.get("district")
    variety = data.get("variety")
    date_str = data.get("date")

    if not all([
        district,
        variety,
        date_str
    ]):

        return jsonify({
            "error": "Missing fields"
        }), 400

    try:

        year, month, day = map(
            int,
            date_str.split("-")
        )

    except Exception:

        return jsonify({
            "error": "Invalid date format"
        }), 400

    row = price_template.copy()

    row.loc[:, :] = 0

    row.at[0, "Year"] = year
    row.at[0, "Month"] = month
    row.at[0, "Day"] = day

    if district in row.columns:

        row.at[0, district] = 1

    if variety in row.columns:

        row.at[0, variety] = 1

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    try:

        pred = price_model.predict(row)[0]

        pred_price = int(
            round(pred)
        )

    except Exception as e:

        return jsonify({
            "error": f"Prediction failed: {e}"
        }), 500

    # --------------------------------------------------------
    # TREND GRAPH
    # --------------------------------------------------------

    trend = []

    labels = [
        "3 months ago",
        "2 months ago",
        "1 month ago",
        "Today",
        "1 month ahead",
        "2 months ahead",
        "3 months ahead",
    ]

    for offset, label in zip(
        range(-3, 4),
        labels
    ):

        new_month = month + offset
        new_year = year

        while new_month < 1:

            new_month += 12
            new_year -= 1

        while new_month > 12:

            new_month -= 12
            new_year += 1

        row.at[0, "Year"] = new_year
        row.at[0, "Month"] = new_month
        row.at[0, "Day"] = 15

        try:

            price = int(
                round(
                    price_model.predict(row)[0]
                )
            )

        except Exception:

            price = None

        trend.append({

            "label": label,

            "date": (
                f"{new_year}-"
                f"{new_month:02d}-15"
            ),

            "price": price,
        })

    # --------------------------------------------------------
    # GEMINI REASONING
    # --------------------------------------------------------

    reasoning = (
        "AI reasoning unavailable."
    )

    if gemini_client:

        try:

            prompt = (

                "Analyze this arecanut price prediction:\n"

                f"District: {district}\n"
                f"Variety: {variety}\n"
                f"Predicted Price: ₹{pred_price}\n"

                "Explain in 2-3 lines the possible "
                "market factors that may influence "
                "this price. "

                "Do NOT generate fake data. "
                "Return plain text only. "
                "Do NOT use markdown or asterisks."
            )

            response = (
                gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
            )

            reasoning = response.text

        except Exception as e:

            print(
                "Gemini error:",
                e
            )

            reasoning = (
                "AI reasoning temporarily unavailable."
            )

    return jsonify({

        "price": pred_price,

        "graph": trend,

        "reasoning": reasoning,
    })


# ============================================================
# SOIL ANALYSIS API
# ============================================================

@app.route(
    "/api/analyze_soil",
    methods=["POST"]
)
def analyze_soil():

    soil_model = get_soil_model()

    if soil_model is None:

        return jsonify({
            "error": (
                "Soil model could not be loaded "
                "from Hugging Face."
            )
        }), 500

    if "soil_image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    img_file = request.files[
        "soil_image"
    ]

    if img_file.filename == "":

        return jsonify({
            "error": "Empty file name"
        }), 400

    temp_path = None

    try:

        # ----------------------------------------------------
        # Temporary file
        # ----------------------------------------------------

        suffix = os.path.splitext(
            img_file.filename
        )[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            img_file.save(temp_path)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        img_array = preprocess_soil_image(
            temp_path
        )

        preds = soil_model.predict(
            img_array
        )[0]

        class_index = int(
            np.argmax(preds)
        )

        soil_result = SOIL_CLASSES[
            class_index
        ]

        # ----------------------------------------------------
        # Brightness override
        # ----------------------------------------------------

        if soil_result == "Red_soil":

            try:

                img = Image.open(
                    temp_path
                ).convert("L")

                avg_brightness = np.mean(
                    img
                )

                if avg_brightness < 150:

                    soil_result = "Red_soil"

                else:

                    soil_result = "Alluvial_soil"

            except Exception:

                pass

        # ----------------------------------------------------
        # Suitability
        # ----------------------------------------------------

        base_prob = float(
            preds[class_index]
        )

        low, high = SOIL_SUITABILITY[
            soil_result
        ]

        suitability = int(
            low
            + (high - low) * base_prob
        )

    except Exception as e:

        return jsonify({
            "error": f"Prediction failed: {e}"
        }), 500

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass

    # --------------------------------------------------------
    # Gemini reasoning
    # --------------------------------------------------------

    reasoning = (
        "AI reasoning unavailable."
    )

    if gemini_client:

        try:

            prompt = (

                "You are an agricultural soil expert.\n"

                f"Predicted soil type: {soil_result}\n"

                f"Suitability score for Arecanut: "
                f"{suitability}%\n\n"

                "Explain in 2-3 lines:\n"

                "1. Why this soil is suitable or not.\n"
                "2. What fertilizers improve this soil.\n"
                "3. Steps to make the soil better "
                "for Arecanut cultivation.\n"

                "Plain text only. "
                "No markdown, bold or special formatting."
            )

            response = (
                gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
            )

            reasoning = response.text.replace(
                "*",
                ""
            )

        except Exception:

            reasoning = (
                "AI reasoning temporarily unavailable."
            )

    return jsonify({

        "result": soil_result,

        "suitability": suitability,

        "reasoning": reasoning,
    })


# ============================================================
# DISEASE DETECTION API
# ============================================================

@app.route(
    "/api/detect_disease",
    methods=["POST"]
)
def detect_disease():

    model_type = request.form.get(
        "type"
    )

    if "disease_image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    img_file = request.files[
        "disease_image"
    ]

    if img_file.filename == "":

        return jsonify({
            "error": "Empty file name"
        }), 400

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if model_type == "leaf":

        model = get_leaf_model()
        labels = LEAF_CLASSES

    elif model_type == "trunk":

        model = get_trunk_model()
        labels = TRUNK_CLASSES

    elif model_type == "crop":

        model = get_crop_model()
        labels = CROP_CLASSES

    elif model_type == "nut_quality":

        model = get_nut_quality_model()
        labels = NUT_CLASSES

    else:

        return jsonify({
            "error": "Invalid model type"
        }), 400

    if model is None:

        return jsonify({
            "error": (
                f"{model_type} model could not "
                "be loaded from Hugging Face."
            )
        }), 500

    temp_path = None

    try:

        # ----------------------------------------------------
        # Temporary image
        # ----------------------------------------------------

        suffix = os.path.splitext(
            img_file.filename
        )[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            img_file.save(temp_path)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        img_array = preprocess_disease_image(
            temp_path
        )

        preds = model.predict(
            img_array
        )[0]

        class_index = int(
            np.argmax(preds)
        )

        result_label = labels[
            class_index
        ]

        confidence = float(
            np.max(preds)
        )

    except Exception as e:

        return jsonify({
            "error": f"Prediction failed: {e}"
        }), 500

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass

    # --------------------------------------------------------
    # Gemini reasoning
    # --------------------------------------------------------

    reasoning = (
        "AI reasoning unavailable."
    )

    if gemini_client:

        try:

            prompt = (

                "You are an expert in Arecanut diseases.\n"

                f"Detected: {result_label}\n"

                f"Confidence: "
                f"{confidence * 100:.2f}%\n\n"

                "In 3 short lines, explain:\n"

                "1. What this disease/quality means.\n"
                "2. How serious it is.\n"
                "3. What treatment or care should be done.\n"

                "Use plain text only. "
                "No bold or markdown."
            )

            response = (
                gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
            )

            reasoning = response.text.replace(
                "*",
                ""
            )

        except Exception:

            reasoning = (
                "AI reasoning temporarily unavailable."
            )

    return jsonify({

        "result": result_label,

        "confidence": round(
            confidence * 100,
            2
        ),

        "reasoning": reasoning,
    })


# ============================================================
# EXPERT CHATBOT
# ============================================================

@app.route(
    "/api/expert_chat",
    methods=["POST"]
)
def expert_chat():

    data = request.get_json()

    if not data:

        return jsonify({
            "reply": "Please type a message."
        })

    user_msg = data.get(
        "message",
        ""
    ).strip().lower()

    if not user_msg:

        return jsonify({
            "reply": "Please type a message."
        })

    # --------------------------------------------------------
    # Greetings
    # --------------------------------------------------------

    greetings = [

        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good evening",
        "good afternoon",
        "yo",
        "namaste",
        "hola",
        "hi there",
        "hello there",
        "how are you",
        "what can you do",
        "who are you",
    ]

    if any(
        user_msg.startswith(g)
        for g in greetings
    ):

        return jsonify({

            "reply": (
                "Hello! I’m AgroNova, "
                "your assistant for arecanut "
                "farming and crop health. "
                "How can I help you today?"
            )
        })

    # --------------------------------------------------------
    # Agriculture topic check
    # --------------------------------------------------------

    allowed_keywords = [

        "arecanut",
        "areca",
        "supari",
        "crop",
        "soil",
        "disease",
        "fertilizer",
        "pest",
        "yield",
        "irrigation",
        "climate",
        "plant",
        "leaf",
        "nut",
        "root",
        "stem",
        "harvest",
        "farm",
        "agriculture",
        "weather",
        "manure",
    ]

    is_related = any(
        word in user_msg
        for word in allowed_keywords
    )

    if not is_related:

        return jsonify({

            "reply": (
                "I can help only with "
                "arecanut cultivation, soil, "
                "diseases, climate, pests, "
                "and agricultural practices. "
                "Please ask something related "
                "to farming."
            )
        })

    # --------------------------------------------------------
    # Gemini answer
    # --------------------------------------------------------

    if not gemini_client:

        return jsonify({

            "reply": (
                "AI service is not configured "
                "right now."
            )
        })

    try:

        prompt = (

            "You are AgroNova — an expert "
            "assistant specialized ONLY in "
            "Arecanut farming.\n"

            "Answer the user's question in "
            "2–3 short, practical lines.\n"

            "Avoid long paragraphs, avoid "
            "bold formatting, avoid chemical names.\n"

            f"User question: {user_msg}\n"

            "Now give a simple and helpful answer:"
        )

        response = (
            gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )

        final_reply = (
            response.text
            .replace("*", "")
            .strip()
        )

        return jsonify({
            "reply": final_reply
        })

    except Exception as e:

        print(
            "Chatbot error:",
            e
        )

        return jsonify({

            "reply": (
                "Sorry, I'm unable "
                "to respond right now."
            )
        })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "ok",

        "application": "AgroNova",

        "huggingface_repository":
            HF_REPO_ID,

        "models": list(
            HF_MODELS.keys()
        ),
    })


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    print(
        "AgroNova Server Running..."
    )

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True,
    )