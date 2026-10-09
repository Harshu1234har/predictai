
%%writefile /content/app.py
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="PredictAI | Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "/content/model/machine_failure_model.pkl"
FEATURES_PATH = "/content/model/features.pkl"

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #101d36 55%, #11112a 100%);
    color: #f3f6ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f, #101d36);
    border-right: 1px solid #24466b;
}
h1, h2, h3, p, label, [data-testid="stMetricLabel"] {
    color: #f3f6ff !important;
}
.hero {
    background: linear-gradient(120deg, #103957, #29356e, #522c70);
    padding: 30px;
    border: 1px solid #42659c;
    border-radius: 22px;
    margin-bottom: 22px;
    box-shadow: 0 10px 35px #0005;
}
.hero h1 { font-size: 40px; margin: 0; }
.hero p { color: #d9e8ff !important; }
.metric-card {
    background: linear-gradient(145deg, #172b46, #111c31);
    border: 1px solid #315780;
    padding: 20px;
    border-radius: 16px;
    min-height: 115px;
}
.metric-label { color: #b8cbe8; font-size: 14px; }
.metric-value { color: #79c8ff; font-size: 30px; font-weight: 800; }
div.stButton > button {
    background: linear-gradient(90deg, #1689ef, #7657f5);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 22px;
}
div.stButton > button:hover {
    border: 1px solid #8ed7ff;
    color: white;
}
.login-card {
    background: linear-gradient(145deg, #102945, #211b46);
    padding: 35px;
    border: 1px solid #42659c;
    border-radius: 22px;
}
.small-note { color: #afc4e4; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- LOGIN ----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("""
        <div style="padding:65px 15px 20px 10px">
          <div style="font-size:65px">⚡</div>
          <h1 style="font-size:55px">PredictAI</h1>
          <h2>Predictive Maintenance</h2>
          <p style="color:#b8cbe8">
          Smart machine monitoring powered by machine learning.
          Identify possible failures before unexpected downtime.
          </p>
          <br>
          <h3>🛠 Intelligent monitoring</h3>
          <h3>📊 Failure risk analytics</h3>
          <h3>🚨 Early warning predictions</h3>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.write("")
        st.write("")
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("## 🔐 Welcome Back")
        st.write("Sign in to access your industrial dashboard.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter password"
            )
            submitted = st.form_submit_button(
                "Sign In →", use_container_width=True
            )

        if submitted:
            if username == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.markdown("""
        <p class="small-note">
        Demo login: admin / admin123<br>
        Change these demo credentials before any real deployment.
        </p></div>
        """, unsafe_allow_html=True)

    st.stop()

if model is None:
    st.error(
        "Model file not found at /content/model/machine_failure_model.pkl. "
        "Train and save your model first."
    )
    st.stop()

# ---------------- NAVIGATION ----------------
with st.sidebar:
    st.markdown("# ⚡ PredictAI")
    st.caption("PREDICTIVE MAINTENANCE")
    st.success("● SYSTEM ONLINE")
    page = st.radio(
        "🧭 NAVIGATION",
        [
            "Dashboard",
            "Machine Prediction",
            "Model Performance",
            "Feature Importance",
            "Technology Stack"
        ]
    )
    st.divider()
    st.markdown("### 🤖 AI MODEL")
    st.info("Random Forest Classifier")
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <p>⚡ AI-POWERED INDUSTRIAL INTELLIGENCE</p>
  <h1>Predictive Maintenance Dashboard</h1>
  <p>Monitor machine conditions and estimate failure risk before downtime.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("📊 Model Overview")

    # Metrics from the original model evaluation
    metric_data = [
        ("MODEL ACCURACY", "98.25%"),
        ("PRECISION", "88.37%"),
        ("RECALL", "55.88%"),
        ("F1 SCORE", "68.47%")
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metric_data):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("### 🏭 System Status")
        st.success("Model loaded and ready for predictions.")
        st.markdown("**Algorithm:** Random Forest")
        st.markdown("**Input features:** 6")
        st.markdown("**Dataset:** AI4I 2020 Predictive Maintenance")
    with b:
        st.markdown("### 🚦 Risk Categories")
        st.success("🟢 LOW — below 30%")
        st.warning("🟡 MEDIUM — 30% to below 70%")
        st.error("🔴 HIGH — 70% or above")

    st.caption(
        "Metrics are from the original held-out test evaluation. "
        "Risk thresholds are demonstration settings, not industrial safety limits."
    )

# ---------------- PREDICTION ----------------
elif page == "Machine Prediction":
    st.subheader("⚙️ Machine Failure Prediction")
    st.write("Enter machine operating conditions.")

    c1, c2 = st.columns(2)

    with c1:
        air_temp = st.number_input(
            "Air Temperature [K]", min_value=250.0,
            max_value=350.0, value=300.0, step=0.1
        )
        process_temp = st.number_input(
            "Process Temperature [K]", min_value=250.0,
            max_value=400.0, value=310.0, step=0.1
        )
        rpm = st.number_input(
            "Rotational Speed [rpm]", min_value=0,
            max_value=10000, value=1500
        )

    with c2:
        torque = st.number_input(
            "Torque [Nm]", min_value=0.0,
            max_value=200.0, value=40.0, step=0.5
        )
        tool_wear = st.number_input(
            "Tool Wear [min]", min_value=0,
            max_value=500, value=100
        )
        machine_type = st.selectbox(
            "Machine Type", ["L", "M", "H"],
            help="AI4I product quality category, not necessarily a physical machine model."
        )

    if st.button("🔍 Predict Failure Risk", use_container_width=True):
        type_mapping = {"H": 0, "L": 1, "M": 2}

        input_df = pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": type_mapping[machine_type]
        }])

        try:
            probability = float(model.predict_proba(input_df)[0][1])
            predicted_class = int(model.predict(input_df)[0])
            risk_percent = probability * 100

            st.markdown("### 📈 Prediction Result")
            st.progress(float(np.clip(probability, 0, 1)))
            st.metric("Estimated Failure Probability", f"{risk_percent:.2f}%")

            if probability < 0.30:
                st.success("🟢 LOW RISK — Machine is in the low predicted-risk category.")
            elif probability < 0.70:
                st.warning("🟡 MEDIUM RISK — Consider inspecting the machine.")
            else:
                st.error("🔴 HIGH RISK — Prompt inspection is recommended.")

            st.write(
                "**Model classification:**",
                "Failure predicted" if predicted_class == 1 else "No failure predicted"
            )
            st.caption(
                "This is a model estimate, not a guarantee of machine safety. "
                "Use appropriate industrial monitoring and maintenance procedures."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- PERFORMANCE ----------------
elif page == "Model Performance":
    st.subheader("📈 Model Performance")
    st.write("Results from the original test-set evaluation.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "98.25%")
    c2.metric("Precision", "88.37%")
    c3.metric("Recall", "55.88%")
    c4.metric("F1 Score", "68.47%")

    st.markdown("### Confusion Matrix")
    st.write("Rows = actual class; columns = predicted class.")
    st.dataframe(
        pd.DataFrame(
            [[1927, 5], [30, 38]],
            index=["Actual: No Failure", "Actual: Failure"],
            columns=["Predicted: No Failure", "Predicted: Failure"]
        ),
        use_container_width=True
    )
    st.caption(
        "Matrix and metrics correspond to the original saved model's test results. "
        "They should be recalculated if you replace the model."
    )

# ---------------- FEATURE IMPORTANCE ----------------
elif page == "Feature Importance":
    st.subheader("⭐ Feature Importance")

    names = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type"
    ]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        if len(values) == len(names):
            importance_df = pd.DataFrame({
                "Feature": names,
                "Importance": values
            }).sort_values("Importance", ascending=True)

            st.bar_chart(
                importance_df.set_index("Feature")["Importance"],
                horizontal=True
            )
            st.dataframe(
                importance_df.sort_values("Importance", ascending=False),
                use_container_width=True
            )
        else:
            st.warning("The saved model has a different number of input features.")
    else:
        st.info("Feature importance is unavailable for this model.")

# ---------------- TECHNOLOGY ----------------
elif page == "Technology Stack":
    st.subheader("🛠️ Technology Stack")

    technologies = [
        ("🐍 Python", "Core programming language"),
        ("🐼 Pandas & NumPy", "Data preparation and numerical processing"),
        ("🤖 Scikit-learn", "Machine learning and Random Forest"),
        ("📊 Streamlit", "Interactive dashboard and web interface"),
        ("💾 Joblib", "Saving and loading the trained model"),
        ("☁️ Google Colab", "Cloud notebook environment"),
        ("🌐 ngrok", "Temporary public HTTPS tunnel"),
        ("🏭 AI4I 2020 Dataset", "Predictive maintenance training data")
    ]

    for i in range(0, len(technologies), 2):
        cols = st.columns(2)
        for col, item in zip(cols, technologies[i:i+2]):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h3>{item[0]}</h3><p>{item[1]}</p></div>',
                    unsafe_allow_html=True
                )
                st.write("")

st.divider()
st.caption("PredictAI • Predictive Maintenance • Educational demonstration")
