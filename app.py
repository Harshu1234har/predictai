
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# PREDICTAI - AI-BASED PREDICTIVE MAINTENANCE SYSTEM
# =====================================================

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
    background: linear-gradient(135deg, #0b1220 0%, #111d35 55%, #172554 100%);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #172554);
}
h1, h2, h3, p, label, [data-testid="stMetricLabel"] {
    color: #f8fafc !important;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #2563eb, #7c3aed);
    margin-bottom: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,.25);
}
.hero h1 { color: white !important; margin-bottom: 5px; }
.hero p { color: #e0e7ff !important; }
.panel {
    padding: 20px;
    border: 1px solid #334155;
    border-radius: 16px;
    background: rgba(30, 41, 59, .75);
    margin-bottom: 16px;
}
.login-wrap {
    max-width: 520px;
    margin: 50px auto;
    padding: 30px;
    background: rgba(30, 41, 59, .9);
    border: 1px solid #475569;
    border-radius: 20px;
}
.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 10px;
    color: white;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    font-weight: 700;
    padding: 10px 14px;
}
.stButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, .8);
    border: 1px solid #475569;
    padding: 18px;
    border-radius: 14px;
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

model = None
encoder = None
features = None
model_error = None

try:
    model, encoder, features = load_model_files()
except Exception as e:
    model_error = str(e)

# ---------------- LOGIN ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
    <div class="login-wrap">
        <h1 style="text-align:center;">⚙️ PredictAI</h1>
        <p style="text-align:center;">
        Intelligent Machine Failure Prediction
        </p>
        <hr>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.5, 1])

    with center:
        st.subheader("🔐 Welcome Back")
        st.write("Sign in to access your maintenance dashboard.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )
            submitted = st.form_submit_button("Login to PredictAI")

        if submitted:
            # DEMO ONLY: replace with proper authentication for real use
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.info("Demo login: username `admin` | password `admin123`")
        st.caption(
            "Demo authentication only. Do not use these credentials "
            "to protect private or sensitive information."
        )

    st.stop()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# ⚙️ PredictAI")
    st.caption("SMART INDUSTRIAL MAINTENANCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Machine Prediction",
            "📊 Model Information",
            "💻 Technology Stack"
        ]
    )

    st.divider()
    st.markdown("### System Status")
    if model is not None:
        st.success("Prediction model loaded")
    else:
        st.error("Model files could not be loaded")

    st.caption("PredictAI • Project Demo")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------- COMMON HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>⚙️ PredictAI</h1>
    <p>AI-powered predictive maintenance for smarter,
    safer and more efficient machine operations.</p>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error("The model could not be loaded.")
    st.write(
        "Check that machine_failure_model.pkl, type_encoder.pkl, "
        "and features.pkl are present in the repository root."
    )
    st.code(model_error)
    st.stop()

# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":
    st.subheader("📈 Operations Overview")
    st.write("Welcome to your intelligent maintenance control center.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("AI Model", "Random Forest")

    with c2:
        st.metric("Input Features", str(len(features)))

    with c3:
        st.metric("System", "Ready")

    st.markdown("### 🚀 What you can do")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="panel">
            <h3>🔍 Failure Prediction</h3>
            <p>Enter machine operating conditions and estimate
            the model's probability of failure.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <h3>📊 Model Information</h3>
            <p>Inspect the model type, required inputs, and
            available feature importance information.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧭 How it works")

    st.write("1. Enter the machine's operating measurements.")
    st.write("2. Run the prediction using the trained model.")
    st.write("3. Review the estimated failure probability and risk category.")
    st.warning(
        "This is a student project and decision-support demo. "
        "Predictions are not a substitute for industrial safety "
        "systems or scheduled maintenance procedures."
    )

# ---------------- MACHINE PREDICTION ----------------

elif page == "🔍 Machine Prediction":
    st.subheader("🔍 Machine Failure Prediction")
    st.write("Enter the machine's current operating conditions.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            air_temp = st.number_input(
                "Air temperature (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0,
                step=0.1
            )

            process_temp = st.number_input(
                "Process temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=310.0,
                step=0.1
            )

            rpm = st.number_input(
                "Rotational speed (rpm)",
                min_value=0,
                max_value=10000,
                value=1500,
                step=10
            )

        with col2:
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
                step=0.5
            )

            tool_wear = st.number_input(
                "Tool wear (minutes)",
                min_value=0,
                max_value=1000,
                value=100,
                step=1
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

        submitted_prediction = st.form_submit_button(
            "⚡ Predict Machine Failure"
        )

    if submitted_prediction:
        try:
            encoded_type = int(
                encoder.transform([machine_type])[0]
            )

            input_values = {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": encoded_type
            }

            # Use the exact feature order saved during training.
            input_df = pd.DataFrame(
                [[input_values[name] for name in features]],
                columns=features
            )

            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    failure_probability = float(
                        probabilities[classes.index(1)]
                    )
                else:
                    failure_probability = 0.0
            else:
                failure_probability = None

            st.divider()
            st.subheader("📋 Prediction Result")

            if prediction == 1:
                st.error("⚠️ The model predicts MACHINE FAILURE.")
            else:
                st.success("✅ The model predicts NO MACHINE FAILURE.")

            if failure_probability is not None:
                st.metric(
                    "Estimated Failure Probability",
                    f"{failure_probability * 100:.2f}%"
                )
                st.progress(
                    min(max(failure_probability, 0.0), 1.0)
                )

                # Illustrative demo thresholds, not safety standards.
                if failure_probability < 0.30:
                    st.success("🟢 Low estimated risk")
                elif failure_probability < 0.70:
                    st.warning("🟠 Medium estimated risk")
                else:
                    st.error("🔴 High estimated risk")

                st.caption(
                    "Risk labels use illustrative demo thresholds: "
                    "below 30% = low, 30–69.99% = medium, "
                    "70% and above = high. These are not validated "
                    "industrial safety thresholds."
                )

            st.markdown("### Input values used")
            st.dataframe(input_df, use_container_width=True)

            st.info(
                "Use this output as an estimate only. Verify machine "
                "condition with appropriate inspections and safety "
                "procedures before making operational decisions."
            )

        except Exception as e:
            st.error("Prediction failed. Check the model and encoder files.")
            st.code(str(e))

# ---------------- MODEL INFORMATION ----------------

elif page == "📊 Model Information":
    st.subheader("📊 Model Information")

    st.markdown("""
    <div class="panel">
        <h3>🌳 Random Forest Classifier</h3>
        <p>The application loads your trained classification model
        from the repository. It predicts whether machine failure
        is expected based on six input features.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Required input features")
    st.dataframe(
        pd.DataFrame({"Feature": list(features)}),
        use_container_width=True
    )

    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": list(features),
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"]
        )
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info(
            "Feature importance is not available for this model."
        )

    st.caption(
        "No accuracy or validation score is displayed here because "
        "this app does not load a separate evaluation report."
    )

# ---------------- TECHNOLOGY STACK ----------------

elif page == "💻 Technology Stack":
    st.subheader("💻 Technology Stack")

    tech = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "Scikit-learn",
            "Joblib",
            "GitHub",
            "Streamlit Community Cloud"
        ],
        "Purpose": [
            "Programming language",
            "Web application and user interface",
            "Input data handling",
            "Machine learning model",
            "Saving and loading trained model files",
            "Version control and source code hosting",
            "Application deployment and hosting"
        ]
    })

    st.dataframe(tech, use_container_width=True)

    st.markdown("### 📌 Project Summary")
    st.write(
        "PredictAI demonstrates how machine learning can be used "
        "to estimate machine failure from operating measurements."
    )

    st.markdown("### 🔒 Security note")
    st.write(
        "The included login is only a demonstration. For a real "
        "application, configure secure authentication and do not "
        "store passwords or API tokens directly in source code."
    )

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "PredictAI | AI-Based Predictive Maintenance System | "
    "Educational demonstration"
)
