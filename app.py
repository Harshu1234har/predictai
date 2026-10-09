
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PredictAI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f3f7ff 0%, #eef2ff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101d40, #263b72);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
    margin-bottom: 3px;
}
.subtitle {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}
.metric-card {
    background: white;
    padding: 22px 18px;
    border-radius: 16px;
    border-left: 5px solid #6366f1;
    box-shadow: 0 5px 18px rgba(30, 41, 59, 0.08);
    margin-bottom: 12px;
}
.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}
.metric-value {
    color: #172554;
    font-size: 29px;
    font-weight: 800;
    margin-top: 6px;
}
.panel {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(30, 41, 59, 0.07);
}
.login-box {
    background: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 8px 35px rgba(30, 41, 59, 0.12);
}
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOGIN ----------------
# Demo login only. Do not use this as production authentication.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='login-box'>"
            "<h1 style='text-align:center;color:#4338ca;'>⚙️ PredictAI</h1>"
            "<p style='text-align:center;color:#64748b;'>"
            "AI-Based Predictive Maintenance System</p>"
            "</div>",
            unsafe_allow_html=True
        )

        with st.form("login_form"):
            st.subheader("Welcome back")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login", use_container_width=True
            )

            if submitted:
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        st.caption(
            "Demo credentials: username admin | password admin123"
        )
    st.stop()


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "machine_failure_model.pkl")
    encoder = joblib.load(BASE_DIR / "type_encoder.pkl")
    features = joblib.load(BASE_DIR / "features.pkl")
    return model, encoder, features


try:
    model, encoder, features = load_artifacts()
except Exception as e:
    st.error("Unable to load the model files.")
    st.write(
        "Check that machine_failure_model.pkl, type_encoder.pkl, "
        "and features.pkl are in the same GitHub folder as app.py."
    )
    st.exception(e)
    st.stop()


# ---------------- MODEL EVALUATION RESULTS ----------------
# These values are from your Google Colab evaluation.
accuracy = 98.25
precision = 88.37
recall = 55.88
f1_score = 68.47

# Rows = actual class; columns = predicted class.
# Order: Normal, Failure.
confusion = np.array([
    [1927, 5],
    [30, 38]
])


# ---------------- SIDEBAR ----------------
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
    st.caption("Algorithm: Random Forest")
    st.caption("Input features: 6")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-title'>PredictAI Dashboard</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='subtitle'>"
    "AI-powered machine health monitoring and failure prediction"
    "</div>",
    unsafe_allow_html=True
)


# ---------------- METRIC CARD HELPER ----------------
def metric_card(label, value, color):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# DASHBOARD
# =====================================================
if page == "🏠 Dashboard":

    st.subheader("📌 Model Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Accuracy", f"{accuracy:.2f}%", "#6366f1")
    with c2:
        metric_card("Precision", f"{precision:.2f}%", "#06b6d4")
    with c3:
        metric_card("Recall", f"{recall:.2f}%", "#f59e0b")
    with c4:
        metric_card("F1-score", f"{f1_score:.2f}%", "#10b981")

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 🎯 Confusion Matrix")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        image = ax.imshow(confusion, cmap="viridis")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Failure"])
        ax.set_yticklabels(["Normal", "Failure"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix - Random Forest")

        for i in range(2):
            for j in range(2):
                text_color = "white" if confusion[i, j] < 1000 else "black"
                ax.text(
                    j, i, str(confusion[i, j]),
                    ha="center", va="center",
                    color=text_color, fontsize=13
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("### 🧠 Important Machine Features")

        importance = getattr(model, "feature_importances_", None)

        if importance is not None and len(importance) == len(features):
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values("Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"],
                color="#0d9488"
            )
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance - Random Forest")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Feature importance is unavailable for this model.")

    st.markdown("---")
    st.subheader("🛠️ System Status")

    a, b, c = st.columns(3)
    with a:
        st.info("**Model:** Random Forest")
    with b:
        st.success("**Status:** Loaded successfully")
    with c:
        st.info("**Purpose:** Predict machine failures")


# =====================================================
# MACHINE PREDICTION
# =====================================================
elif page == "🔮 Machine Prediction":

    st.subheader("🔮 Predict Machine Failure")
    st.write(
        "Enter the machine sensor readings below, then click "
        "**Predict Failure Risk**."
    )

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
                value=40.0, step=0.5
            )
            tool_wear = st.number_input(
                "Tool wear (min)",
                min_value=0, max_value=500,
                value=100, step=1
            )
            machine_type = st.selectbox(
                "Machine type",
                options=["L", "M", "H"]
            )

        predict_button = st.form_submit_button(
            "🚀 Predict Failure Risk",
            use_container_width=True
        )

    if predict_button:
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

            input_df = pd.DataFrame(
                [[input_values[f] for f in features]],
                columns=features
            )

            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                failure_index = classes.index(1)
                failure_probability = float(probabilities[failure_index])
            else:
                failure_probability = None

            st.markdown("---")
            st.subheader("📋 Prediction Result")

            if prediction == 1:
                st.error("⚠️ The model predicts a MACHINE FAILURE.")
            else:
                st.success("✅ The model predicts NORMAL operation.")

            if failure_probability is not None:
                st.metric(
                    "Estimated failure probability",
                    f"{failure_probability * 100:.2f}%"
                )
                st.progress(
                    min(max(failure_probability, 0.0), 1.0)
                )

                if failure_probability < 0.30:
                    st.success("Demo risk category: LOW")
                elif failure_probability < 0.70:
                    st.warning("Demo risk category: MEDIUM")
                else:
                    st.error("Demo risk category: HIGH")

            st.caption(
                "This is a student-project prediction, not a guarantee "
                "of machine safety. Risk categories use illustrative "
                "probability thresholds."
            )

        except Exception as e:
            st.error("Prediction could not be completed.")
            st.exception(e)


# =====================================================
# MODEL PERFORMANCE
# =====================================================
elif page == "📊 Model Performance":

    st.subheader("📊 Random Forest Evaluation")
    st.write("Evaluation results recorded from your Google Colab run.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Accuracy", f"{accuracy:.2f}%", "#6366f1")
    with c2:
        metric_card("Precision", f"{precision:.2f}%", "#06b6d4")
    with c3:
        metric_card("Recall", f"{recall:.2f}%", "#f59e0b")
    with c4:
        metric_card("F1-score", f"{f1_score:.2f}%", "#10b981")

    st.markdown("### What do these metrics mean?")

    st.markdown("""
    - **Accuracy:** Percentage of all predictions that were correct.
    - **Precision:** Among predicted failures, the percentage that
      were actual failures.
    - **Recall:** Among actual failures, the percentage the model
      successfully identified.
    - **F1-score:** Balance between precision and recall.
    """)

    st.markdown("### Confusion Matrix")

    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual Normal", "Actual Failure"],
            columns=["Predicted Normal", "Predicted Failure"]
        ),
        use_container_width=True
    )

    st.info(
        "The dataset contains many more normal cases than failures. "
        "Accuracy alone can therefore hide missed failures. Pay close "
        "attention to recall when evaluating predictive maintenance."
    )


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
elif page == "🧠 Feature Importance":

    st.subheader("🧠 Feature Importance Analysis")
    st.write(
        "These scores show how much each feature contributes to the "
        "Random Forest's split-based importance. They do not prove "
        "causation."
    )

    importance = getattr(model, "feature_importances_", None)

    if importance is not None and len(importance) == len(features):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        st.dataframe(
            importance_df.style.format({"Importance": "{:.4f}"}),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        chart_df = importance_df.sort_values("Importance", ascending=True)
        ax.barh(
            chart_df["Feature"],
            chart_df["Importance"],
            color="#0d9488"
        )
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance - Random Forest")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("The loaded model does not expose feature importance.")


# =====================================================
# ABOUT PROJECT
# =====================================================
elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About PredictAI")

    st.markdown("""
    **PredictAI** is a machine-learning student project designed to
    demonstrate predictive maintenance for industrial equipment.

    ### Technologies
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest classifier
    - Streamlit dashboard
    - Matplotlib visualizations

    ### Input sensor readings
    - Air temperature
    - Process temperature
    - Rotational speed
    - Torque
    - Tool wear
    - Machine type

    ### Project objective
    Estimate whether machine readings indicate a possible failure so
    maintenance teams can investigate early.

    **Important:** This is a demonstration system. It should not be used
    as the sole basis for real industrial safety decisions.
    """)

st.markdown("---")
st.caption("PredictAI | AI-Based Predictive Maintenance System")
