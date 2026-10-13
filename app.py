import os
import io
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PredictAI — Machine Predictive Maintenance Dashboard
# Uses the existing trained model and feature order.
# Put these files in your GitHub repository:
#   model/machine_failure_model.pkl
#   model/type_encoder.pkl
#   model/features.pkl
# ============================================================

st.set_page_config(
    page_title="PredictAI | Machine Mission Control",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOGIN GATE — login and dashboard are in this same app.py
# Configure PREDICTAI_USERNAME and PREDICTAI_PASSWORD in
# Streamlit Community Cloud -> App -> Settings -> Secrets.
# Demo fallback credentials are for testing only.
# ============================================================
def get_login_credentials():
    try:
        username = st.secrets.get("PREDICTAI_USERNAME", "admin")
        password = st.secrets.get("PREDICTAI_PASSWORD", "PredictAI@123")
    except Exception:
        username = "admin"
        password = "PredictAI@123"
    return str(username), str(password)


if "predictai_authenticated" not in st.session_state:
    st.session_state.predictai_authenticated = False

if not st.session_state.predictai_authenticated:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 15% 15%, rgba(49,199,213,.18), transparent 28%),
          radial-gradient(circle at 90% 85%, rgba(71,118,230,.18), transparent 30%),
          linear-gradient(135deg,#08142d 0%,#10244b 55%,#0a1936 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .login-wrap {
        max-width: 980px; margin: 6vh auto 2vh auto; padding: 26px;
        border: 1px solid rgba(255,255,255,.16); border-radius: 28px;
        background: rgba(255,255,255,.06);
        box-shadow: 0 24px 70px rgba(0,0,0,.22);
    }
    .login-title { color:#fff; font-size:2.55rem; font-weight:800; letter-spacing:-1px; }
    .login-subtitle { color:#c8d8f7; font-size:1.02rem; line-height:1.6; }
    .login-kicker { color:#57e2e7; font-size:.78rem; font-weight:800; letter-spacing:3px; }
    .login-feature {
        color:#eaf1ff; border:1px solid rgba(255,255,255,.14);
        background:rgba(255,255,255,.055); border-radius:14px;
        padding:13px 15px; margin:10px 0; font-size:.95rem;
    }
    [data-testid="stForm"] {
        border:1px solid rgba(255,255,255,.15); border-radius:20px;
        padding:24px; background:rgba(7,18,42,.55);
    }
    [data-testid="stForm"] label, [data-testid="stForm"] p { color:#eaf1ff !important; }
    [data-testid="stForm"] input { background:#f8fbff; color:#102044; }
    .login-note { color:#aebfdf; font-size:.82rem; }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("""
        <div class="login-wrap">
          <div class="login-kicker">PREDICTIVE INTELLIGENCE PLATFORM</div>
          <div class="login-title">PredictAI<br>Mission Control</div>
          <p class="login-subtitle">
            Machine health intelligence in one workspace. Sign in to access
            failure prediction, factory monitoring demos, model performance, sensor trends and reports.
          </p>
          <div class="login-feature">🧠 &nbsp; ML-powered failure probability</div>
          <div class="login-feature">🏭 &nbsp; Factory Mission Control demo</div>
                    <div class="login-feature">📊 &nbsp; Model metrics and confusion matrix</div>
          <div class="login-feature">🔔 &nbsp; Alert history and downloadable reports</div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
        st.markdown("## 🔐 Secure sign in")
        st.caption("Use the credentials configured for your PredictAI app.")
        with st.form("predictai_login_form", clear_on_submit=False):
            login_username = st.text_input("Username", placeholder="Enter username")
            login_password = st.text_input("Password", type="password", placeholder="Enter password")
            login_submit = st.form_submit_button("Sign in to PredictAI", type="primary", use_container_width=True)

        if login_submit:
            expected_username, expected_password = get_login_credentials()
            if login_username.strip() == expected_username and login_password == expected_password:
                st.session_state.predictai_authenticated = True
                st.rerun()
            else:
                st.error("Username or password is incorrect. Please try again.")

        st.markdown(
            "<p class='login-note'>Demo fallback: username <b>admin</b> · password <b>PredictAI@123</b>"
            "<br>Change these using Streamlit Secrets before sharing your app publicly.</p>",
            unsafe_allow_html=True,
        )
    st.stop()


# ---------- Existing project settings ----------
MODEL_PATH = "model/machine_failure_model.pkl"
ENCODER_PATH = "model/type_encoder.pkl"
FEATURES_PATH = "model/features.pkl"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]

# These metrics match the original Google Colab confusion matrix below:
# [[1927, 5], [30, 38]]. Update only if you want to show metrics
# from a different evaluation run.
MODEL_METRICS = {
    "Accuracy": 98.25,
    "Precision": 88.37,
    "Recall": 55.88,
    "F1 Score": 68.47,
}
CONFUSION_MATRIX = np.array([[1927, 5], [30, 38]])
CLASS_NAMES = ["Normal", "Failure"]

# Risk bands are dashboard labels, not certified safety thresholds.
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.70
FAILURE_DECISION_THRESHOLD = 0.50

TYPE_TO_CODE_FALLBACK = {"H": 0, "L": 1, "M": 2}
TYPE_EXAMPLES = {
    "H": {
        "air": 303.0, "process": 313.0, "rpm": 900,
        "torque": 60.0, "wear": 240,
        "note": "Example guidance only. It does not guarantee a failure prediction.",
    },
    "L": {
        "air": 303.0, "process": 313.0, "rpm": 900,
        "torque": 60.0, "wear": 240,
        "note": "Example guidance only. It does not guarantee a failure prediction.",
    },
    "M": {
        "air": 300.0, "process": 310.0, "rpm": 1400,
        "torque": 45.0, "wear": 150,
        "note": "Example guidance only. It does not guarantee a failure prediction.",
    },
}

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --navy:#101c3a; --blue:#4776e6; --cyan:#31c7d5; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
      background:
        radial-gradient(circle at 85% 0%, rgba(49,199,213,.12), transparent 27%),
        linear-gradient(135deg, #f5f8ff 0%, #f7fbff 55%, #eef5ff 100%);
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color:#101c3a; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#101c3a,#182b53); }
    [data-testid="stSidebar"] * { color:#f2f6ff !important; }
    [data-testid="stSidebar"] .stRadio label { background:rgba(255,255,255,.06); padding:9px 12px; border-radius:10px; }
    .hero {
      padding: 24px 28px; border-radius: 22px; color: white;
      background: linear-gradient(115deg,#101c3a 0%,#244a91 62%,#31a9c5 120%);
      box-shadow: 0 12px 35px rgba(25,55,110,.15); margin-bottom: 18px;
    }
    .hero h1 { color:white !important; margin:0; font-size:2.15rem; }
    .hero p { color:#dce8ff; margin:.45rem 0 0 0; }
    .metric-card {
      background:rgba(255,255,255,.88); border:1px solid #e1e9f7;
      padding:17px 18px; border-radius:16px; min-height:110px;
      box-shadow:0 5px 18px rgba(30,60,110,.05);
    }
    .metric-label { color:#61708b; font-size:.82rem; font-weight:600; }
    .metric-value { color:#101c3a; font-family:'Space Grotesk'; font-size:1.75rem; font-weight:700; margin-top:6px; }
    .metric-sub { color:#7a879d; font-size:.75rem; margin-top:2px; }
    .section-card { background:rgba(255,255,255,.85); border:1px solid #e1e9f7; padding:18px; border-radius:18px; }
    .status-pill { display:inline-block; padding:5px 10px; border-radius:999px; font-weight:700; font-size:.8rem; }
    .green { background:#dcfce7; color:#166534; }
    .orange { background:#ffedd5; color:#9a3412; }
    .red { background:#fee2e2; color:#991b1b; }
    .blue { background:#dbeafe; color:#1d4ed8; }
    .small-muted { color:#697891; font-size:.86rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid #e3eaf6; padding:12px 14px; border-radius:14px; }
    div.stButton > button { border-radius:11px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load existing artifacts ----------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, FEATURES, f"Model file not found: {MODEL_PATH}"
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
        features = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else FEATURES
        return model, encoder, list(features), None
    except Exception as exc:
        return None, None, FEATURES, str(exc)

model, type_encoder, model_features, load_error = load_artifacts()

# Use the saved encoder when available; fallback matches the original Colab mapping.
def encode_type(machine_type):
    if type_encoder is not None:
        try:
            return int(type_encoder.transform([machine_type])[0])
        except Exception:
            pass
    return TYPE_TO_CODE_FALLBACK[machine_type]

def get_failure_probability(row_df):
    """Return model probability for Machine failure class 1."""
    if model is None:
        return None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row_df)[0]
        classes = list(getattr(model, "classes_", range(len(probs))))
        if 1 in classes:
            return float(probs[classes.index(1)])
        return float(probs[-1])
    return float(model.predict(row_df)[0])

def risk_band(probability):
    if probability < LOW_RISK_MAX:
        return "LOW", "green"
    if probability < MEDIUM_RISK_MAX:
        return "MEDIUM", "orange"
    return "HIGH", "red"

def explain_prediction(probability, machine_type, air, process, rpm, torque, wear):
    """A transparent, local explanation; no external AI service is required."""
    band, _ = risk_band(probability)
    notes = []
    if torque >= 50:
        notes.append("torque is relatively high")
    if rpm < 1000:
        notes.append("rotational speed is relatively low")
    elif rpm > 2000:
        notes.append("rotational speed is relatively high")
    if wear >= 200:
        notes.append("tool wear is high")
    if process - air < 8:
        notes.append("the air/process temperature gap is small")
    if not notes:
        notes.append("the entered values do not cross the simple sensor heuristics used for this explanation")
    outcome = "failure is predicted" if probability >= FAILURE_DECISION_THRESHOLD else "normal operation is predicted"
    return (
        f"The model estimates a **{probability:.1%}** failure probability, so the dashboard labels this "
        f"as **{band.lower()} risk**. Using the separate classification cutoff of "
        f"{FAILURE_DECISION_THRESHOLD:.0%}, **{outcome}**. "
        f"Sensor observations: {', '.join(notes)}. Machine type: **{machine_type}**. "
        "These observations are guidance, not proof of causation; the prediction comes from the trained model."
    )

# ---------- Session state ----------
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ---------- Sidebar navigation ----------
st.sidebar.markdown("## ⚙️ PredictAI")
st.sidebar.caption("Predictive Maintenance • ML Control Center")
if st.sidebar.button("Log out", use_container_width=True):
    st.session_state.predictai_authenticated = False
    st.rerun()
page_options = [
    "Overview", "Machine Prediction", "Factory Mission Control",
    "Model Performance", "Sensor Trends", "Alert History", "About Project",
]
requested_page = st.query_params.get("page", "Overview")
if requested_page not in page_options:
    requested_page = "Overview"
page = st.sidebar.radio(
    "NAVIGATION",
    page_options,
    index=page_options.index(requested_page),
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
if model is not None:
    st.sidebar.success("Trained model loaded")
else:
    st.sidebar.error("Model not loaded")
st.sidebar.caption("Risk bands are demo thresholds, not certified safety limits.")

def render_hero(title, subtitle, emoji="⚙️"):
    st.markdown(
        f'<div class="hero"><h1>{emoji} {title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def make_row(air, process, rpm, torque, wear, machine_type):
    values = {
        "Air temperature [K]": float(air),
        "Process temperature [K]": float(process),
        "Rotational speed [rpm]": float(rpm),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(wear),
        "Type": encode_type(machine_type),
    }
    # Preserve the exact feature order used during training.
    return pd.DataFrame([[values[f] for f in model_features]], columns=model_features)

def validate_inputs(air, process, rpm, torque, wear):
    problems = []
    if not (250 <= air <= 330): problems.append("Air temperature must be between 250 and 330 K.")
    if not (250 <= process <= 350): problems.append("Process temperature must be between 250 and 350 K.")
    if process <= air: problems.append("Process temperature should be higher than air temperature.")
    if not (0 <= rpm <= 5000): problems.append("Rotational speed must be between 0 and 5000 rpm.")
    if not (0 <= torque <= 100): problems.append("Torque must be between 0 and 100 Nm.")
    if not (0 <= wear <= 300): problems.append("Tool wear must be between 0 and 300 min.")
    return problems

def run_prediction(air, process, rpm, torque, wear, machine_type, machine_id="Machine"):
    problems = validate_inputs(air, process, rpm, torque, wear)
    if problems:
        for problem in problems:
            st.error(problem)
        return None
    if model is None:
        st.error("The trained model could not be loaded. Check the model files in your GitHub repository.")
        return None
    row = make_row(air, process, rpm, torque, wear, machine_type)
    probability = get_failure_probability(row)
    predicted_failure = probability >= FAILURE_DECISION_THRESHOLD
    band, color = risk_band(probability)
    result = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id": machine_id,
        "machine_type": machine_type,
        "air": float(air),
        "process": float(process),
        "rpm": int(rpm),
        "torque": float(torque),
        "wear": int(wear),
        "probability": probability,
        "prediction": "MACHINE FAILURE" if predicted_failure else "NORMAL",
        "risk": band,
        "risk_color": color,
        "explanation": explain_prediction(probability, machine_type, air, process, rpm, torque, wear),
    }
    st.session_state.last_prediction = result
    st.session_state.alerts.insert(0, result)
    st.session_state.alerts = st.session_state.alerts[:100]
    return result

def show_result(result):
    if not result:
        return
    st.markdown("### Prediction result")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("FAILURE PROBABILITY", f"{result['probability']:.2%}", "Model probability for failure class")
    with c2:
        metric_card("PREDICTED STATE", result["prediction"], f"Decision cutoff: {FAILURE_DECISION_THRESHOLD:.0%}")
    with c3:
        metric_card("RISK LEVEL", result["risk"], "Demo risk band")
    st.progress(min(max(result["probability"], 0.0), 1.0), text=f"Estimated failure probability: {result['probability']:.1%}")
    st.markdown(result["explanation"])
    st.caption("Important: risk band and predicted state are different. For example, a 62.67% probability predicts failure at a 50% cutoff, while remaining MEDIUM risk under these dashboard bands.")

# ---------- Pages ----------
if page == "Overview":
    render_hero("Machine Health Overview", "A clear view of machine condition, failure probability, and model performance.", "🛰️")
    if load_error:
        st.error(f"Model loading issue: {load_error}")
        st.info("Check that the three model files are committed to the repository in the model/ folder.")
    latest = st.session_state.last_prediction
    if latest:
        status = latest["prediction"]
        prob = latest["probability"]
        risk = latest["risk"]
    else:
        status, prob, risk = "Awaiting input", 0.0, "—"
    cols = st.columns(4)
    with cols[0]: metric_card("MODEL", "Random Forest", "Existing trained model")
    with cols[1]: metric_card("LAST PREDICTION", status, "Run a prediction to update")
    with cols[2]: metric_card("FAILURE PROBABILITY", f"{prob:.1%}" if latest else "—", "Most recent prediction")
    with cols[3]: metric_card("RISK BAND", risk, "Low <30% • Medium 30–70% • High ≥70%")
    st.markdown("### Quick actions")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown('<div class="section-card"><h3>🧪 Test a machine</h3><p class="small-muted">Enter sensor readings and estimate failure probability.</p></div>', unsafe_allow_html=True)
        if st.button("Open Machine Prediction", use_container_width=True):
            st.query_params["page"] = "Machine Prediction"
            st.rerun()
    with q2:
        st.markdown('<div class="section-card"><h3>🏭 Mission Control</h3><p class="small-muted">Explore a simulated factory floor and machine states.</p></div>', unsafe_allow_html=True)
        if st.button("Open Factory Map", use_container_width=True):
            st.query_params["page"] = "Factory Mission Control"
            st.rerun()
    st.markdown("### Evaluation snapshot")
    m = st.columns(4)
    for col, (name, val) in zip(m, MODEL_METRICS.items()):
        with col: metric_card(name.upper(), f"{val:.2f}%", "Original Colab evaluation")
    st.markdown("### How to read risk")
    r1, r2, r3 = st.columns(3)
    with r1: st.markdown('<div class="section-card"><span class="status-pill green">LOW • &lt;30%</span><p>Lower estimated failure probability. Continue normal monitoring.</p></div>', unsafe_allow_html=True)
    with r2: st.markdown('<div class="section-card"><span class="status-pill orange">MEDIUM • 30–70%</span><p>Review sensor values and consider closer inspection.</p></div>', unsafe_allow_html=True)
    with r3: st.markdown('<div class="section-card"><span class="status-pill red">HIGH • ≥70%</span><p>Prioritize inspection according to your maintenance procedure.</p></div>', unsafe_allow_html=True)
    st.caption("The factory map and sample machines are a demonstration, not a live industrial monitoring connection.")

elif page == "Machine Prediction":
    render_hero("Machine Prediction", "Enter real sensor readings. Example values are guidance only and do not auto-fill the form.", "🧪")
    if model is None:
        st.error(f"Model is not available. {load_error or ''}")
    with st.form("prediction_form"):
        machine_id = st.text_input("Machine ID", value="Machine-01")
        machine_type = st.selectbox("Machine type", ["H", "L", "M"], index=2, help="The H/L/M encoding is preserved from your original model.")
        ex = TYPE_EXAMPLES[machine_type]
        st.info(f"**Example values for type {machine_type}:** Air {ex['air']} K • Process {ex['process']} K • RPM {ex['rpm']} • Torque {ex['torque']} Nm • Tool wear {ex['wear']} min. {ex['note']}")
        st.markdown("#### Sensor readings")
        a, b = st.columns(2)
        with a:
            air = st.number_input("Air temperature (K)", min_value=250.0, max_value=330.0, value=300.0, step=0.1)
            rpm = st.number_input("Rotational speed (rpm)", min_value=0, max_value=5000, value=1400, step=10)
            wear = st.number_input("Tool wear (min)", min_value=0, max_value=300, value=150, step=1)
        with b:
            process = st.number_input("Process temperature (K)", min_value=250.0, max_value=350.0, value=310.0, step=0.1)
            torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=45.0, step=0.5)
        submitted = st.form_submit_button("🔍 Predict Machine Health", type="primary", use_container_width=True)
    if submitted:
        result = run_prediction(air, process, rpm, torque, wear, machine_type, machine_id.strip() or "Machine-01")
        if result:
            show_result(result)
    elif st.session_state.last_prediction:
        show_result(st.session_state.last_prediction)

elif page == "Factory Mission Control":
    render_hero("Factory Mission Control", "Interactive-style overview of a sample factory floor. Select a machine to inspect its details.", "🏭")
    st.info("This is a simulated factory map using sample readings. It is not connected to live factory sensors.")
    sample_machines = [
        {"id":"M-01","type":"M","air":300,"process":310,"rpm":1400,"torque":45,"wear":150,"x":0},
        {"id":"M-02","type":"L","air":303,"process":313,"rpm":900,"torque":60,"wear":240,"x":1},
        {"id":"M-03","type":"H","air":299,"process":309,"rpm":1500,"torque":35,"wear":80,"x":2},
        {"id":"M-04","type":"M","air":301,"process":311,"rpm":1100,"torque":52,"wear":210,"x":3},
        {"id":"M-05","type":"L","air":300,"process":310,"rpm":1700,"torque":30,"wear":40,"x":4},
        {"id":"M-06","type":"H","air":302,"process":312,"rpm":950,"torque":58,"wear":230,"x":5},
    ]
    factory_results = []
    if model is not None:
        for item in sample_machines:
            try:
                row = make_row(item["air"], item["process"], item["rpm"], item["torque"], item["wear"], item["type"])
                p = get_failure_probability(row)
                band, color = risk_band(p)
            except Exception:
                p, band, color = 0.0, "LOW", "green"
            item.update(probability=p, risk=band, color=color, prediction="MACHINE FAILURE" if p >= FAILURE_DECISION_THRESHOLD else "NORMAL")
            factory_results.append(item)
    else:
        for item in sample_machines:
            item.update(probability=0.0, risk="MODEL OFFLINE", color="blue", prediction="Not available")
            factory_results.append(item)
    legend = st.columns(3)
    with legend[0]: st.markdown('<span class="status-pill green">● LOW RISK</span>', unsafe_allow_html=True)
    with legend[1]: st.markdown('<span class="status-pill orange">● MEDIUM RISK</span>', unsafe_allow_html=True)
    with legend[2]: st.markdown('<span class="status-pill red">● HIGH RISK</span>', unsafe_allow_html=True)
    # A responsive grid of selectable machine tiles.
    for start in range(0, len(factory_results), 3):
        cols = st.columns(3)
        for col, item in zip(cols, factory_results[start:start+3]):
            with col:
                icon = {"green":"🟢","orange":"🟠","red":"🔴","blue":"🔵"}.get(item["color"], "⚪")
                st.markdown(
                    f'<div class="section-card"><div style="font-size:1.8rem">{icon} ⚙️</div>'
                    f'<h3>{item["id"]}</h3><p>Type {item["type"]} • {item["rpm"]} rpm</p>'
                    f'<p><b>{item["risk"]}</b> risk • {item["probability"]:.1%}</p>'
                    f'<p class="small-muted">{item["prediction"]}</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Inspect {item['id']}", key=f"inspect_{item['id']}", use_container_width=True):
                    st.session_state["selected_factory_machine"] = item["id"]
    selected_id = st.session_state.get("selected_factory_machine", "M-01")
    selected = next((m for m in factory_results if m["id"] == selected_id), factory_results[0])
    st.markdown("---")
    st.markdown(f"### Machine details: {selected['id']}")
    d1, d2, d3, d4 = st.columns(4)
    with d1: metric_card("AIR TEMP", f"{selected['air']} K")
    with d2: metric_card("PROCESS TEMP", f"{selected['process']} K")
    with d3: metric_card("TORQUE", f"{selected['torque']} Nm")
    with d4: metric_card("TOOL WEAR", f"{selected['wear']} min")
    st.write(f"**Model output:** {selected['prediction']} • **Failure probability:** {selected['probability']:.2%} • **Risk band:** {selected['risk']}")
    if st.button("Add selected machine to alert history"):
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "machine_id": selected["id"],
            "machine_type": selected["type"], "air": selected["air"], "process": selected["process"],
            "rpm": selected["rpm"], "torque": selected["torque"], "wear": selected["wear"],
            "probability": selected["probability"], "prediction": selected["prediction"],
            "risk": selected["risk"], "risk_color": selected["color"],
            "explanation": "Sample machine selected from the simulated factory map."
        }
        st.session_state.alerts.insert(0, record)
        st.success("Added to alert history.")

elif page == "Model Performance":
    render_hero("Model Performance", "The same original Colab evaluation snapshot: Random Forest, six input features, and the original confusion matrix.", "📈")
    st.info("Metrics below correspond to the original Colab confusion matrix. Keep these values aligned with the evaluation run you report in your project.")
    cols = st.columns(4)
    for col, (name, val) in zip(cols, MODEL_METRICS.items()):
        with col: metric_card(name.upper(), f"{val:.2f}%", "Original Colab result")
    st.markdown("### Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(CONFUSION_MATRIX, interpolation="nearest")
    ax.set_title("Confusion Matrix — Google Colab")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    for i in range(CONFUSION_MATRIX.shape[0]):
        for j in range(CONFUSION_MATRIX.shape[1]):
            ax.text(j, i, f"{CONFUSION_MATRIX[i, j]}", ha="center", va="center", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Rows = actual labels; columns = predicted labels. Normal is class 0 and Failure is class 1.")
    st.markdown("### Feature importance")
    if model is not None and hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        names = list(model_features)
        if len(importances) == len(names):
            imp_df = pd.DataFrame({"Feature": names, "Importance": importances}).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(imp_df["Feature"], imp_df["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title("Random Forest Feature Importance")
            ax.grid(axis="x", alpha=.25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.dataframe(imp_df.sort_values("Importance", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.warning("Feature importance count does not match the saved feature list.")
    else:
        st.warning("Feature importance is unavailable because the model is not loaded or does not expose feature_importances_.")
    st.markdown("### What the metrics mean")
    st.write("- **Accuracy:** percentage of all test cases classified correctly.")
    st.write("- **Precision:** among predicted failures, the percentage that were actual failures.")
    st.write("- **Recall:** among actual failures, the percentage detected by the model.")
    st.write("- **F1 score:** balance between precision and recall.")
    st.caption("Failure cases are rare in this dataset, so accuracy alone is not enough to judge performance.")

elif page == "Sensor Trends":
    render_hero("Sensor Trends", "Explore sample sensor readings and compare distributions. Upload your own CSV to explore your dataset.", "📉")
    uploaded = st.file_uploader("Optional: upload a CSV containing sensor columns", type=["csv"])
    if uploaded is not None:
        try:
            trend_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            trend_df = pd.DataFrame()
    else:
        rng = np.random.default_rng(42)
        n = 80
        air_vals = rng.normal(300, 2, n)
        process_vals = air_vals + rng.normal(10, 1.5, n)
        trend_df = pd.DataFrame({
            "Sample": np.arange(1, n+1),
            "Air temperature [K]": air_vals,
            "Process temperature [K]": process_vals,
            "Rotational speed [rpm]": rng.normal(1500, 180, n).clip(500, 3000),
            "Torque [Nm]": rng.normal(40, 10, n).clip(5, 80),
            "Tool wear [min]": np.sort(rng.integers(0, 300, n)),
        })
        st.caption("Showing synthetic sample trends. They are for UI demonstration only, not real machine readings.")
    available = [f for f in FEATURES[:-1] if f in trend_df.columns]
    if available:
        selected_sensor = st.selectbox("Choose a sensor", available)
        if "Sample" not in trend_df.columns:
            trend_df = trend_df.copy()
            trend_df["Sample"] = np.arange(1, len(trend_df)+1)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(trend_df["Sample"], trend_df[selected_sensor], marker=".", linewidth=1.5)
        ax.set_title(selected_sensor)
        ax.set_xlabel("Sample / row")
        ax.set_ylabel(selected_sensor)
        ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        stat_cols = st.columns(3)
        with stat_cols[0]: metric_card("AVERAGE", f"{pd.to_numeric(trend_df[selected_sensor], errors='coerce').mean():.2f}")
        with stat_cols[1]: metric_card("MINIMUM", f"{pd.to_numeric(trend_df[selected_sensor], errors='coerce').min():.2f}")
        with stat_cols[2]: metric_card("MAXIMUM", f"{pd.to_numeric(trend_df[selected_sensor], errors='coerce').max():.2f}")
        st.dataframe(trend_df.head(20), use_container_width=True)
    else:
        st.warning("The uploaded CSV does not contain any expected sensor column names.")

elif page == "Alert History":
    render_hero("Alert History & Reports", "Review predictions generated in this app session and download a CSV report.", "🔔")
    if not st.session_state.alerts:
        st.info("No predictions recorded yet. Run a prediction or add a sample machine from Mission Control.")
    else:
        alerts_df = pd.DataFrame(st.session_state.alerts)
        preferred = ["time","machine_id","machine_type","prediction","probability","risk","air","process","rpm","torque","wear"]
        alerts_df = alerts_df[[c for c in preferred if c in alerts_df.columns]]
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        csv_bytes = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download alert history (CSV)", data=csv_bytes, file_name="predictai_alert_history.csv", mime="text/csv", use_container_width=True)
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": "Random Forest",
            "record_count": len(alerts_df),
            "alerts": st.session_state.alerts,
        }
        st.download_button("⬇️ Download report (JSON)", data=json.dumps(report, indent=2).encode("utf-8"), file_name="predictai_report.json", mime="application/json", use_container_width=True)
        if st.button("Clear session alert history"):
            st.session_state.alerts = []
            st.rerun()
    st.caption("Alert history is stored only for the current Streamlit session; it is not a permanent database.")

elif page == "About Project":
    render_hero("About PredictAI", "A student machine-learning project for predictive maintenance.", "💡")
    st.markdown("""
    ### Project summary
    PredictAI uses your existing Random Forest classifier to estimate machine-failure probability from six features:
    - Air temperature (K)
    - Process temperature (K)
    - Rotational speed (rpm)
    - Torque (Nm)
    - Tool wear (min)
    - Machine type (H, L, M)

    ### How the dashboard works
    1. Enter sensor values.
    2. The saved model estimates failure probability.
    3. The model prediction uses a 50% decision cutoff.
    4. The dashboard separately labels risk as Low (<30%), Medium (30% to <70%), or High (≥70%).
    5. The explanation describes the result in plain language, and the report page lets you export session alerts.

    ### Important limitations
    - The Mission Control map and default trends use sample/synthetic data.
    - The model only works with the same feature order and type encoding used during training.
    - Risk cutoffs are demonstration choices, not official safety standards.
    - Predictions support learning and should not replace inspection or a qualified maintenance process.
    """)
    st.markdown("### Model files")
    st.code("model/machine_failure_model.pkl\nmodel/type_encoder.pkl\nmodel/features.pkl")

# Small footer
st.markdown("---")
st.caption("PredictAI • Student ML project • Predictions are estimates, not a guarantee of machine condition.")
