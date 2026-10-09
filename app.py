
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PredictAI | Smart Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #ecfeff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a3a, #243b70);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    color: white;
    background: linear-gradient(110deg, #172554, #2563eb, #0891b2);
    margin-bottom: 22px;
    box-shadow: 0 10px 30px #1e3a8a25;
}
.hero h1 { color: white; margin-bottom: 6px; }
.hero p { color: #e0f2fe; font-size: 17px; }
.metric-card {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 5px 18px #0f172a0c;
}
.section-title { color: #1d4ed8; font-weight: 700; }
.login-box {
    background: white;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #dbeafe;
    box-shadow: 0 12px 35px #1e3a8a18;
}
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1rem;
}
.stButton > button:hover {
    color: white;
    border: 1px solid #60a5fa;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL FILES ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "machine_failure_model.pkl"
ENCODER_PATH = BASE_DIR / "type_encoder.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, encoder, features

# ---------------- LOGIN ----------------
def login_page():
    st.markdown("""
    <div class="hero">
        <h1>⚙️ PredictAI</h1>
        <p>Intelligent predictive maintenance for smarter, safer operations.</p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.1, 1])
    with center:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.subheader("🔐 Welcome back")
        st.write("Sign in to open your predictive maintenance dashboard.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter password"
            )
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            # Demo-only login. Configure real credentials in Streamlit Secrets
            # before using this app beyond a classroom demonstration.
            try:
                valid_user = st.secrets["APP_USERNAME"]
                valid_password = st.secrets["APP_PASSWORD"]
            except Exception:
                valid_user = "admin"
                valid_password = "admin123"

            if username == valid_user and password == valid_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.caption("Demo login: username `admin` · password `admin123`")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DASHBOARD HEADER ----------------
def show_header():
    st.markdown("""
    <div class="hero">
        <h1>⚙️ PredictAI</h1>
        <p>AI-powered machine failure prediction and maintenance insights</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
if not st.session_state.get("authenticated", False):
    login_page()
    st.stop()

# Load model safely and show a useful error if files are missing
try:
    model, encoder, features = load_model_files()
except Exception as error:
    st.error("Could not load the model files.")
    st.write(
        "Check that these three files are in the same GitHub folder as app.py:"
    )
    st.code(
        "machine_failure_model.pkl\n"
        "type_encoder.pkl\n"
        "features.pkl"
    )
    st.caption(f"Technical details: {error}")
    st.stop()

show_header()

with st.sidebar:
    st.title("🧭 Navigation")
    page = st.radio(
        "Choose a page",
        ["Dashboard", "Machine Prediction", "Model Information"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("### 🟢 System Status")
    st.success("Model files loaded")
    st.caption("PredictAI • Student project")
    if st.button("Log out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ---------------- DASHBOARD PAGE ----------------
if page == "Dashboard":
    st.subheader("📊 Dashboard Overview")
    st.write("Monitor machine conditions and estimate the risk of failure.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-card"><h4>🤖 Model</h4>'
            '<h2>Random Forest</h2><p>Classification model</p></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h4>🧩 Input Features</h4>'
            f'<h2>{len(features)}</h2><p>Machine condition measurements</p></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><h4>⚡ Prediction</h4>'
            '<h2>Real-time</h2><p>Based on entered readings</p></div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.info(
        "Use **Machine Prediction** in the sidebar to enter sensor readings "
        "and estimate the machine's failure risk."
    )

    st.subheader("🛠️ What PredictAI does")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### 🌡️")
        st.markdown("**Monitors conditions**")
        st.write("Uses temperature, speed, torque, tool wear, and machine type.")
    with b:
        st.markdown("### 🧠")
        st.markdown("**Predicts failure risk**")
        st.write("Uses the trained machine-learning model to estimate failure probability.")
    with c:
        st.markdown("### 🧰")
        st.markdown("**Supports maintenance**")
        st.write("Helps operators decide when further inspection may be needed.")

# ---------------- PREDICTION PAGE ----------------
elif page == "Machine Prediction":
    st.subheader("🔍 Machine Failure Prediction")
    st.write("Enter the current machine readings below.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            air_temp = st.number_input(
                "Air temperature (K)",
                min_value=250.0, max_value=350.0,
                value=300.0, step=0.1
            )
            process_temp = st.number_input(
                "Process temperature (K)",
                min_value=250.0, max_value=400.0,
                value=310.0, step=0.1
            )
            rpm = st.number_input(
                "Rotational speed (rpm)",
                min_value=0, max_value=10000,
                value=1500, step=10
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0, max_value=200.0,
                value=40.0, step=0.1
            )
            tool_wear = st.number_input(
                "Tool wear (min)",
                min_value=0, max_value=500,
                value=100, step=1
            )
            machine_type = st.selectbox(
                "Machine type",
                options=["L", "M", "H"],
                format_func=lambda x: {
                    "L": "L — Low",
                    "M": "M — Medium",
                    "H": "H — High"
                }[x]
            )

        submitted = st.form_submit_button(
            "🔮 Predict Failure Risk",
            use_container_width=True
        )

    if submitted:
        if process_temp < air_temp:
            st.warning(
                "Process temperature is below air temperature. "
                "Please verify these readings before interpreting the result."
            )

        try:
            encoded_type = int(encoder.transform([machine_type])[0])

            input_values = {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": encoded_type
            }

            # Preserve the exact feature order used during training
            input_df = pd.DataFrame(
                [[input_values[name] for name in features]],
                columns=features
            )

            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                failure_index = classes.index(1)
                failure_probability = float(probabilities[failure_index])
            else:
                st.error("This model does not support probability estimates.")
                st.stop()

            st.divider()
            st.subheader("📋 Prediction Result")

            result_col, risk_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.error("⚠️ Model prediction: FAILURE")
                else:
                    st.success("✅ Model prediction: NO FAILURE")

            with risk_col:
                st.metric(
                    "Estimated failure probability",
                    f"{failure_probability * 100:.2f}%"
                )

            st.progress(min(max(failure_probability, 0.0), 1.0))

            # These are illustrative UI thresholds, not validated safety limits.
            if failure_probability < 0.30:
                st.success("🟢 Low estimated risk")
            elif failure_probability < 0.70:
                st.warning("🟠 Medium estimated risk")
            else:
                st.error("🔴 High estimated risk")

            st.caption(
                "Risk bands are illustrative demo thresholds, not certified "
                "industrial safety standards. Model estimates do not replace "
                "inspection or maintenance procedures."
            )

            with st.expander("View readings used for this prediction"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as error:
            st.error("Prediction could not be completed.")
            st.write(
                "Check that the feature names, encoder, and saved model "
                "match the files used during training."
            )
            st.caption(f"Technical details: {error}")

# ---------------- MODEL INFORMATION PAGE ----------------
elif page == "Model Information":
    st.subheader("🧠 Model Information")

    st.markdown("""
    This application uses a trained machine-learning classifier to estimate
    machine failure from operating conditions.

    **Input measurements**
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear
    - Machine type (L, M, or H)
    """)

    st.markdown("### 📦 Model files")
    st.write("**Classifier:** `machine_failure_model.pkl`")
    st.write("**Machine-type encoder:** `type_encoder.pkl`")
    st.write("**Training feature order:** `features.pkl`")

    st.markdown("### ⚠️ Important limitations")
    st.write(
        "This is a demonstration project. Predictions may be wrong and "
        "should not be used as the sole basis for real-world safety, "
        "shutdown, or maintenance decisions."
    )

    st.markdown("### 🧰 Technology stack")
    st.write("Python · Streamlit · Pandas · Scikit-learn · Joblib")
