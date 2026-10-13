import os
import time
import hmac
import secrets
import sqlite3
from functools import wraps

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    abort,
)

from werkzeug.security import check_password_hash

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from backend.auth_db import (
    DB_PATH,
    init_db,
    get_user_by_username,
    get_user_by_id,
)

# ==========================================
# PROJECT PATHS AND APPLICATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "random_forest_model.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "ai4i2020.csv"
)

STATIC_DIR = os.path.join(BACKEND_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BACKEND_DIR, "templates"),
    static_folder=STATIC_DIR,
)

# Persistent secret key for sessions
SECRET_FILE = os.path.join(BACKEND_DIR, ".secret_key")

if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE, "rb") as secret_file:
        SECRET_KEY = secret_file.read()
else:
    SECRET_KEY = secrets.token_bytes(32)
    with open(SECRET_FILE, "wb") as secret_file:
        secret_file.write(SECRET_KEY)

app.secret_key = SECRET_KEY

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=1800,
)

# ==========================================
# DATABASE INITIALIZATION
# ==========================================

init_db()


def init_security_tables():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                username TEXT PRIMARY KEY,
                failed_count INTEGER NOT NULL DEFAULT 0,
                first_failed REAL NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0
            )
        """)


def init_prediction_history_table():
    """Create the prediction history table."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL
                    DEFAULT (datetime('now', 'localtime')),
                machine_type TEXT NOT NULL,
                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                result TEXT NOT NULL,
                failure_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                threshold REAL NOT NULL,
                recommendation TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_history_user
            ON prediction_history(user_id, id)
        """)


init_security_tables()
init_prediction_history_table()

# ==========================================
# LOGIN SECURITY
# ==========================================


def login_is_locked(username):
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT locked_until
            FROM login_attempts
            WHERE username = ?
        """, (username.lower(),)).fetchone()

    return bool(row and row[0] > time.time())


def record_failed_login(username):
    now = time.time()
    username = username.lower()

    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute("""
            SELECT failed_count, first_failed
            FROM login_attempts
            WHERE username = ?
        """, (username,)).fetchone()

        if not row or now - row[1] > 900:
            count = 1
            first_failed = now
        else:
            count = row[0] + 1
            first_failed = row[1]

        locked_until = now + 900 if count >= 5 else 0

        connection.execute("""
            INSERT INTO login_attempts
                (username, failed_count, first_failed, locked_until)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                failed_count = excluded.failed_count,
                first_failed = excluded.first_failed,
                locked_until = excluded.locked_until
        """, (username, count, first_failed, locked_until))


def clear_failed_logins(username):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "DELETE FROM login_attempts WHERE username = ?",
            (username.lower(),),
        )

# ==========================================
# SESSION AND CSRF SECURITY
# ==========================================


def current_user():
    user_id = session.get("user_id")

    if user_id is None:
        return None

    user = get_user_by_id(user_id)

    if user is None or not user["approved"]:
        session.clear()
        return None

    return user


def login_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash(
                "Please log in to access the dashboard.",
                "error",
            )
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped


def admin_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        user = current_user()

        if user is None:
            return redirect(url_for("login"))

        if user["role"] != "admin":
            abort(403)

        return view_function(*args, **kwargs)

    return wrapped


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        submitted_token = request.form.get("csrf_token", "")
        saved_token = session.get("csrf_token", "")

        if (
            not submitted_token
            or not saved_token
            or not hmac.compare_digest(
                submitted_token, saved_token
            )
        ):
            abort(
                400,
                description="Invalid or missing security token.",
            )

    # Require a fresh login after 30 minutes of inactivity.
    if session.get("user_id"):
        now = time.time()
        last_activity = session.get("last_activity", now)

        if now - last_activity > 1800:
            session.clear()
            flash(
                "Your session expired. Please log in again.",
                "error",
            )
        else:
            session["last_activity"] = now


@app.context_processor
def add_template_security():
    if not session.get("csrf_token"):
        session["csrf_token"] = secrets.token_urlsafe(32)

    return {
        "csrf_token": session["csrf_token"],
        "logged_in_user": current_user(),
    }

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

PREDICTION_THRESHOLD = 0.40

# ==========================================
# PREPARE DATA FOR MODEL METRICS
# ==========================================

df = pd.read_csv(DATASET_PATH)

df = df.drop(
    columns=["UDI", "Product ID"],
    errors="ignore",
)

df["Type"] = df["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2,
})

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

X = df[features]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ==========================================
# MODEL PERFORMANCE
# ==========================================

# Dashboard metrics use the model's default classification threshold.
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

precision = precision_score(
    y_test, y_pred, zero_division=0
) * 100

recall = recall_score(
    y_test, y_pred, zero_division=0
) * 100

f1 = f1_score(
    y_test, y_pred, zero_division=0
) * 100


def performance_values():
    return {
        "accuracy": round(accuracy, 1),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
    }

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Failure"],
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()

plt.savefig(
    os.path.join(STATIC_DIR, "confusion_matrix.png")
)

plt.close()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

if hasattr(model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"],
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance - Random Forest")
    plt.tight_layout()

    plt.savefig(
        os.path.join(STATIC_DIR, "feature_importance.png")
    )

    plt.close()

# ==========================================
# WELCOME PAGE
# ==========================================


@app.route("/")
def welcome():
    if current_user() is not None:
        return redirect(url_for("home"))

    return render_template("welcome.html")

# ==========================================
# PROJECT DESCRIPTION
# ==========================================


@app.route("/description")
def description():
    return render_template("description.html")

# ==========================================
# LOGIN
# ==========================================


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user() is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get("password", "")

        if (
            not username
            or len(username) > 80
            or len(password) > 1024
        ):
            return render_template(
                "login.html",
                error="Invalid username or password.",
            ), 400

        if login_is_locked(username):
            return render_template(
                "login.html",
                error=(
                    "Too many failed attempts. "
                    "Try again in 15 minutes."
                ),
            ), 429

        user = get_user_by_username(username)

        # Check a password hash even if the username is unknown.
        if user:
            try:
                valid_password = check_password_hash(
                    user["password_hash"], password
                )
            except (ValueError, TypeError):
                valid_password = False
        else:
            valid_password = False

        if (
            user is None
            or not valid_password
            or not user["approved"]
        ):
            record_failed_login(username)

            return render_template(
                "login.html",
                error=(
                    "Invalid credentials or account "
                    "not approved."
                ),
            ), 401

        clear_failed_logins(username)

        session.clear()
        session["user_id"] = user["id"]
        session["csrf_token"] = secrets.token_urlsafe(32)
        session["last_activity"] = time.time()
        session.permanent = True

        return redirect(url_for("home"))

    return render_template("login.html")

# ==========================================
# DASHBOARD
# ==========================================


@app.route("/dashboard")
@login_required
def home():
    return render_template(
        "index.html",
        **performance_values(),
    )

# ==========================================
# PREDICTION HISTORY
# ==========================================


@app.route("/history")
@login_required
def history():
    user = current_user()

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        records = connection.execute("""
            SELECT *
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 100
        """, (user["id"],)).fetchall()

    return render_template(
        "history.html",
        records=records,
    )

# ==========================================
# PREDICTION
# ==========================================


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        machine_type = request.form.get(
            "machine_type", ""
        )

        type_values = {
            "L": 0,
            "M": 1,
            "H": 2,
        }

        if machine_type not in type_values:
            raise ValueError("Select a valid machine type.")

        air_temperature = float(
            request.form["air_temperature"]
        )

        process_temperature = float(
            request.form["process_temperature"]
        )

        rotational_speed = float(
            request.form["rotational_speed"]
        )

        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])

        values = [
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        ]

        if not all(pd.notna(value) for value in values):
            raise ValueError(
                "All measurements must be valid numbers."
            )

        if not 250 <= air_temperature <= 350:
            raise ValueError(
                "Air temperature must be between 250 and 350 K."
            )

        if not 250 <= process_temperature <= 400:
            raise ValueError(
                "Process temperature must be between 250 and 400 K."
            )

        if not 0 < rotational_speed <= 10000:
            raise ValueError(
                "Rotational speed must be between 1 and 10000 RPM."
            )

        if not 0 <= torque <= 1000:
            raise ValueError(
                "Torque must be between 0 and 1000 Nm."
            )

        if not 0 <= tool_wear <= 1000:
            raise ValueError(
                "Tool wear must be between 0 and 1000 minutes."
            )

        input_data = pd.DataFrame(
            [[
                type_values[machine_type],
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
            ]],
            columns=features,
        )

        # Estimate probability of machine failure.
        failure_class_index = list(model.classes_).index(1)

        failure_probability_raw = float(
            model.predict_proba(input_data)[0][
                failure_class_index
            ]
        )

        failure_percentage = round(
            failure_probability_raw * 100, 2
        )

        prediction = int(
            failure_probability_raw >= PREDICTION_THRESHOLD
        )

        # Preliminary project risk bands, not industrial safety limits.
        if failure_probability_raw < 0.20:
            risk_level = "Low"
            risk_status = "low"
        elif failure_probability_raw < PREDICTION_THRESHOLD:
            risk_level = "Medium"
            risk_status = "medium"
        else:
            risk_level = "High"
            risk_status = "high"

        if prediction == 0:
            result = "MACHINE NORMAL"

            message = (
                "The model predicts normal operation. "
                "Continue routine monitoring and follow "
                "maintenance guidance."
            )

            status = "normal"

            recommendation = (
                "Continue routine monitoring and follow "
                "the scheduled maintenance plan."
            )
        else:
            result = "MACHINE FAILURE"

            message = (
                "The model predicts a potential failure. "
                "Arrange an inspection and follow your "
                "maintenance procedures."
            )

            status = "failure"

            recommendation = (
                "Inspect the machine and review its "
                "maintenance requirements."
            )

        # Save this prediction for the currently logged-in user.
        user = current_user()

        with sqlite3.connect(DB_PATH) as connection:
            connection.execute("""
                INSERT INTO prediction_history (
                    user_id,
                    machine_type,
                    air_temperature,
                    process_temperature,
                    rotational_speed,
                    torque,
                    tool_wear,
                    result,
                    failure_probability,
                    risk_level,
                    threshold,
                    recommendation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user["id"],
                machine_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                result,
                failure_percentage,
                risk_level,
                PREDICTION_THRESHOLD,
                recommendation,
            ))

        return render_template(
            "index.html",
            result=result,
            message=message,
            status=status,
            failure_probability=failure_percentage,
            risk_level=risk_level,
            risk_status=risk_status,
            threshold=PREDICTION_THRESHOLD,
            recommendation=recommendation,
            **performance_values(),
        )

    except (ValueError, KeyError, TypeError):
        return render_template(
            "index.html",
            result="INVALID INPUT",
            message=(
                "Please enter valid values for all "
                "machine measurements."
            ),
            status="failure",
            **performance_values(),
        ), 400

# ==========================================
# LOGOUT
# ==========================================


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("You have logged out successfully.", "success")
    return redirect(url_for("login"))

# ==========================================
# ERROR PAGES
# ==========================================


@app.errorhandler(403)
def forbidden(error):
    return (
        "Access denied. You are not authorized to view this page.",
        403,
    )


@app.errorhandler(400)
def bad_request(error):
    return (
        "Request rejected. Refresh the page and try again.",
        400,
    )

# ==========================================
# RUN APPLICATION
# ==========================================


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )
