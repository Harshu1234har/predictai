
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PredictAI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# COLORFUL UI
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f7ff 0%, #e8efff 100%);
    color: #172554;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101d40 0%, #263f79 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #172e65;
}
.subtitle {
    font-size: 17px;
    color: #64748b;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dbe4f0;
    box-shadow: 0 4px 14px rgba(31, 41, 55, 0.06);
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGIN
# Demo fallback credentials: admin / admin123
# For a real deployment, configure credentials in Streamlit
# secrets instead of relying on demo credentials.
# =========================================================
try:
    LOGIN_USERNAME = st.secrets["login"]["username"]
    LOGIN_PASSWORD = st.secrets["login"]["password"]
except Exception:
    LOGIN_USERNAME = os.getenv("PREDICTAI_USERNAME", "admin")
    LOGIN_PASSWORD = os.getenv("PREDICTAI_PASSWORD", "admin123")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown(
            "<h1 style='text-align:center;color:#172e65;'>⚙️ PredictAI</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center;'>Smart Machine Monitoring "
            "and Failure Prediction</p>",
            unsafe_allow_html=True
        )

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login", use_container_width=True
            )

        if submitted:
            if (
                username == LOGIN_USERNAME
                and password == LOGIN_PASSWORD
            ):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.caption(
            "Demo fallback login: admin / admin123. "
            "Change this before using the app for anything private."
        )

    st.stop()

# =========================================================
# LOAD SAVED MODEL FILES
# =========================================================
@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "machine_failure_model.pkl"
    encoder_path = BASE_DIR / "type_encoder.pkl"
    features_path = BASE_DIR / "features.pkl"

    for path in [model_path, encoder_path, features_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file is missing: {path.name}"
            )

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    features = joblib.load(features_path)

    return model, encoder, features


try:
    model, encoder, features = load_artifacts()
except Exception as error:
    st.error(f"Could not load the ML model: {error}")
    st.info(
        "Check that all three .pkl files are uploaded to the "
        "same GitHub folder as app.py."
    )
    st.stop()

# =========================================================
# SAVED EVALUATION RESULTS
# From your earlier evaluation:
# [[1927, 5],
#  [30, 38]]
# These are recorded evaluation results, not recalculated
# against a test set every time the app opens.
# =========================================================
CONFUSION_MATRIX = np.array([
    [1927, 5],
    [30, 38]
])

accuracy = (1927 + 38) / CONFUSION_MATRIX.sum()
precision = 38 / (38 + 5)
recall = 38 / (38 + 30)
f1_score = 2 * precision * recall / (precision + recall)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("# ⚙️ PredictAI")
    st.caption("SMART MACHINE MONITORING")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔮 Machine Prediction",
            "📊 Model Performance",
            "🧠 Feature Importance",
            "ℹ️ About Project"
        ]
    )

    st.divider()
    st.success("ML model loaded")
    st.write("**Algorithm:** Random Forest")
    st.write(f"**Input features:** {len(features)}")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# COMMON FEATURE ORDER AND ENCODING
# =========================================================
def make_input_dataframe(
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    machine_type
):
    encoded_type = int(encoder.transform([machine_type])[0])

    values = {
        "Air temperature [K]": float(air_temp),
        "Process temperature [K]": float(process_temp),
        "Rotational speed [rpm]": float(rpm),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
        "Type": encoded_type
    }

    # Keep exactly the feature order used to train the model.
    return pd.DataFrame(
        [[values[name] for name in features]],
        columns=features
    )


def get_failure_probability(input_df):
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    return float(probabilities[classes.index(1)])


# =========================================================
# SEARCH FOR A MODEL-PREDICTED FAILURE EXAMPLE
# Values are displayed as an example only. They are NOT
# automatically copied into the user's input controls.
# =========================================================
def find_failure_example(machine_type, attempts=20000):
    rng = np.random.default_rng(42)

    # Approximate ranges from the AI4I 2020 dataset.
    # These are for finding a demonstration example, not
    # recommended operating limits for real machinery.
    air = rng.uniform(295.3, 304.5, attempts)
    process = rng.uniform(305.7, 313.8, attempts)
    rpm = rng.integers(1168, 2887, attempts)
    torque = rng.uniform(3.8, 76.6, attempts)
    wear = rng.uniform(0, 253, attempts)

    # Preserve the usual process-temperature relationship.
    process = np.maximum(process, air + 1.0)
    process = np.minimum(process, 313.8)

    encoded_type = int(encoder.transform([machine_type])[0])

    values = {
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Type": np.full(attempts, encoded_type)
    }

    candidates = pd.DataFrame(
        {name: values[name] for name in features},
        columns=features
    )

    predictions = model.predict(candidates)
    failure_indices = np.flatnonzero(predictions == 1)

    if len(failure_indices) == 0:
        return None

    index = int(failure_indices[0])
    probability = get_failure_probability(
        candidates.iloc[[index]]
    )

    return {
        "Air temperature (K)": round(float(air[index]), 2),
        "Process temperature (K)": round(float(process[index]), 2),
        "Rotational speed (rpm)": int(rpm[index]),
        "Torque (Nm)": round(float(torque[index]), 2),
        "Tool wear (min)": round(float(wear[index]), 2),
        "Machine type": machine_type,
        "Estimated failure probability": round(probability * 100, 2)
    }


# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="main-title">PredictAI Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-powered machine health monitoring '
        'and failure prediction</div>',
        unsafe_allow_html=True
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Precision", f"{precision * 100:.2f}%")
    c3.metric("Recall", f"{recall * 100:.2f}%")
    c4.metric("F1-score", f"{f1_score * 100:.2f}%")

    st.divider()
    st.subheader("📌 Project Overview")

    a, b = st.columns(2)

    with a:
        st.markdown("### 🔮 Predict Machine Failure")
        st.write(
            "Enter the sensor readings and use the trained "
            "Random Forest model to estimate machine failure."
        )
        st.markdown("### 📊 Model Performance")
        st.write(
            "Review the saved confusion matrix and the "
            "Accuracy, Precision, Recall and F1-score."
        )

    with b:
        st.markdown("### 🧠 Feature Importance")
        st.write(
            "Compare how much each input feature contributes "
            "to the trained Random Forest model."
        )
        st.markdown("### 🛠️ Technology")
        st.write(
            "Python, Streamlit, Pandas, NumPy, Matplotlib "
            "and Scikit-learn."
        )

    st.info(
        "These metrics come from your earlier model evaluation. "
        "They are not a guarantee of future real-world performance."
    )

# =========================================================
# MACHINE PREDICTION PAGE
# =========================================================
elif page == "🔮 Machine Prediction":
    st.title("🔮 Predict Machine Failure")
    st.write(
        "Select the machine type first. Enter the sensor readings "
        "manually, then click Predict Failure Risk."
    )

    machine_type = st.selectbox(
        "1. Select machine type",
        options=["L", "M", "H"],
        format_func=lambda x: {
            "L": "L — Low type",
            "M": "M — Medium type",
            "H": "H — High type"
        }[x],
        key="machine_type"
    )

    st.caption(
        "The machine type is selected first. The example search "
        "below uses this selected type."
    )

    st.markdown("### Sensor readings")

    left, right = st.columns(2)

    with left:
        air_temp = st.number_input(
            "Air temperature (K)",
            min_value=290.0,
            max_value=310.0,
            value=300.0,
            step=0.1
        )
        process_temp = st.number_input(
            "Process temperature (K)",
            min_value=295.0,
            max_value=320.0,
            value=310.0,
            step=0.1
        )
        rpm = st.number_input(
            "Rotational speed (rpm)",
            min_value=500,
            max_value=4000,
            value=1700,
            step=10
        )

    with right:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=0.5
        )
        tool_wear = st.number_input(
            "Tool wear (min)",
            min_value=0.0,
            max_value=300.0,
            value=150.0,
            step=1.0
        )

    st.caption(
        "The default numbers are starting values only. "
        "They are not guaranteed low-, medium-, or high-risk settings."
    )

    predict_clicked = st.button(
        "🚀 Predict Failure Risk",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:
        if process_temp <= air_temp:
            st.warning(
                "Process temperature is usually higher than air "
                "temperature in this dataset. Check the readings."
            )

        input_df = make_input_dataframe(
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            machine_type
        )

        predicted_class = int(model.predict(input_df)[0])
        probability = get_failure_probability(input_df)

        if predicted_class == 1:
            st.error("🚨 The model predicts MACHINE FAILURE.")
        else:
            st.success("✅ The model predicts NORMAL operation.")

        st.subheader("📋 Prediction Result")

        r1, r2 = st.columns(2)
        r1.metric(
            "Estimated failure probability",
            f"{probability * 100:.2f}%"
        )
        r2.metric(
            "Model prediction",
            "FAILURE" if predicted_class == 1 else "NORMAL"
        )

        # Risk bands are illustrative probability bands, not
        # industry-approved safety thresholds.
        if probability < 0.30:
            risk = "LOW"
            st.success("Risk band: LOW")
        elif probability < 0.70:
            risk = "MEDIUM"
            st.warning("Risk band: MEDIUM")
        else:
            risk = "HIGH"
            st.error("Risk band: HIGH")

        st.progress(min(max(probability, 0.0), 1.0))

        st.caption(
            "Risk bands are demo thresholds. The predicted class and "
            "the estimated probability are different outputs, so "
            "they may not always imply the same label. Do not use "
            "this student project as the sole basis for safety decisions."
        )

        with st.expander("View the exact values submitted to the model"):
            st.dataframe(input_df, use_container_width=True)

    st.divider()
    st.subheader("🧪 Need values that the model predicts as failure?")

    st.write(
        "Click below to search for one example for the selected "
        "machine type. The app will display the values; it will "
        "not change your input boxes. Copy them manually and click "
        "Predict Failure Risk to test them."
    )

    if st.button(
        "🔎 Find a Failure Example",
        use_container_width=True
    ):
        with st.spinner("Searching for an example..."):
            example = find_failure_example(machine_type)

        if example is None:
            st.warning(
                "No failure example was found in this search. "
                "Try another machine type or check your model."
            )
        else:
            st.session_state["failure_example"] = example

    example = st.session_state.get("failure_example")

    if example and example["Machine type"] == machine_type:
        st.success(
            "Found a set of values that the saved model predicts as failure."
        )
        st.dataframe(
            pd.DataFrame([example]),
            use_container_width=True,
            hide_index=True
        )
        st.info(
            "Copy the five sensor values above into the input fields "
            "manually. These are model-generated demonstration values, "
            "not recommended real-machine operating settings."
        )

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.write("Saved evaluation results from your Random Forest model.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Precision", f"{precision * 100:.2f}%")
    c3.metric("Recall", f"{recall * 100:.2f}%")
    c4.metric("F1-score", f"{f1_score * 100:.2f}%")

    st.divider()
    st.subheader("Confusion Matrix — Random Forest")

    fig, ax = plt.subplots(figsize=(7, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=CONFUSION_MATRIX,
        display_labels=["Normal", "Failure"]
    )
    display.plot(
        ax=ax,
        cmap="viridis",
        colorbar=True,
        values_format="d"
    )
    ax.set_title("Confusion Matrix - Random Forest")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **How to read this matrix:**
    - **1927:** Normal machines correctly predicted as normal.
    - **5:** Normal machines incorrectly predicted as failures.
    - **30:** Failures missed by the model.
    - **38:** Failures correctly predicted by the model.
    """)

    st.warning(
        "Recall is 55.88% in this evaluation, so the model missed "
        "30 of the 68 actual failures. High accuracy alone does not "
        "mean that failure detection is reliable."
    )

# =========================================================
# FEATURE IMPORTANCE PAGE
# =========================================================
elif page == "🧠 Feature Importance":
    st.title("🧠 Feature Importance")
    st.write(
        "This graph shows the relative feature importance values "
        "learned by the saved Random Forest model."
    )

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        importance_names = list(features)

        importance_df = pd.DataFrame({
            "Feature": importance_names,
            "Importance": importance_values
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            color="teal"
        )
        ax.set_title("Feature Importance - Random Forest")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            importance_df.sort_values("Importance", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error(
            "This saved model does not expose feature_importances_."
        )

    st.info(
        "Feature importance indicates how the model uses features. "
        "It does not prove that a feature causes machine failure."
    )

# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "ℹ️ About Project":
    st.title("ℹ️ About PredictAI")

    st.markdown("""
    ### Project
    **AI-Based Predictive Maintenance System**

    ### Objective
    Predict possible machine failures from sensor readings so that
    maintenance teams can investigate potential problems earlier.

    ### Machine input features
    - Air temperature (K)
    - Process temperature (K)
    - Rotational speed (rpm)
    - Torque (Nm)
    - Tool wear (min)
    - Machine type (L, M, H)

    ### Technology stack
    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn Random Forest
    - Matplotlib
    - Joblib

    ### Important limitation
    This is an educational predictive-maintenance application.
    Model probabilities, risk bands and predictions are not a
    substitute for industrial inspection, maintenance procedures,
    or safety systems.
    """)
