
import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PredictAI | Smart Machine Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f3f6ff 0%, #e9efff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101d40 0%, #263f78 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    color: #172b58;
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    color: #627394;
    font-size: 17px;
}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #dbe4f5;
    box-shadow: 0 4px 14px rgba(30, 50, 90, 0.08);
}
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGIN PAGE
# Demo credentials: admin / admin123
# For demonstration only, not secure production authentication.
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown(
        "<h1 style='text-align:center;color:#172b58;'>⚙️ PredictAI</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#627394;'>"
        "AI-Powered Predictive Maintenance System</p>",
        unsafe_allow_html=True
    )

    _, login_col, _ = st.columns([1, 1.2, 1])

    with login_col:
        st.markdown("### 🔐 Welcome Back")
        st.write("Sign in to monitor machine health.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )
            login_clicked = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if login_clicked:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        st.caption("Demo login: admin  |  Password: admin123")

    st.stop()

# =========================================================
# LOAD MODEL FILES
# Files must be in the same GitHub folder as app.py.
# =========================================================
@st.cache_resource
def load_model_files():
    model = joblib.load("machine_failure_model.pkl")
    encoder = joblib.load("type_encoder.pkl")
    features = joblib.load("features.pkl")
    return model, encoder, features

try:
    model, encoder, features = load_model_files()
except Exception as error:
    st.error("Unable to load the model files.")
    st.write(
        "Check that machine_failure_model.pkl, "
        "type_encoder.pkl, and features.pkl are in the repository "
        "root alongside app.py."
    )
    st.code(str(error))
    st.stop()

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
    st.success("✅ ML model loaded")
    st.write("**Algorithm:** Random Forest")
    st.write(f"**Input features:** {len(features)}")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# MODEL METRICS
# These match the evaluation results shown in your earlier
# confusion matrix screenshot. Update if you retrain the model.
# =========================================================
accuracy = 98.25
precision = 88.37
recall = 55.88
f1_score = 68.47

# Confusion matrix: rows = true labels, columns = predictions
confusion_data = pd.DataFrame(
    [[1927, 5], [30, 38]],
    index=["Normal", "Failure"],
    columns=["Predicted Normal", "Predicted Failure"]
)

# =========================================================
# DASHBOARD PAGE
# =========================================================
if page == "🏠 Dashboard":
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
    st.write("")

    st.subheader("📈 Model Performance Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.2f}%")
    c2.metric("Precision", f"{precision:.2f}%")
    c3.metric("Recall", f"{recall:.2f}%")
    c4.metric("F1-score", f"{f1_score:.2f}%")

    st.divider()

    st.subheader("⚙️ System Overview")
    left, right = st.columns(2)

    with left:
        st.markdown("### 🤖 Machine Learning")
        st.write("**Model:** Random Forest Classifier")
        st.write("**Task:** Machine failure prediction")
        st.write("**Sensor inputs:** 5 numeric values + machine type")
        st.write("**Output:** Predicted failure and estimated probability")

    with right:
        st.markdown("### 🧭 How to Use")
        st.write("1. Open Machine Prediction.")
        st.write("2. Select machine type L, M, or H.")
        st.write("3. Use the grey example values as a guide.")
        st.write("4. Enter all five sensor readings.")
        st.write("5. Click Predict Failure Risk.")

    st.info(
        "These performance metrics describe the earlier evaluation "
        "results. Recalculate them if you replace or retrain the model."
    )

# =========================================================
# MACHINE PREDICTION PAGE
# =========================================================
elif page == "🔮 Machine Prediction":
    st.markdown(
        "<div class='main-title'>🔮 Predict Machine Failure</div>",
        unsafe_allow_html=True
    )
    st.write(
        "Select the machine type first. Example values appear as "
        "light grey hints; they are not entered automatically."
    )

    # Example values are illustrative, not guaranteed risk labels.
    examples = {
        "L": {
            "air": "298",
            "process": "308",
            "speed": "1500",
            "torque": "30",
            "wear": "10"
        },
        "M": {
            "air": "300",
            "process": "310",
            "speed": "1400",
            "torque": "45",
            "wear": "150"
        },
        "H": {
            "air": "303",
            "process": "313",
            "speed": "1200",
            "torque": "60",
            "wear": "240"
        }
    }

    machine_type = st.selectbox(
        "🏭 Select Machine Type",
        ["L", "M", "H"],
        format_func=lambda x: {
            "L": "L — Low product quality type",
            "M": "M — Medium product quality type",
            "H": "H — High product quality type"
        }[x],
        key="selected_machine_type"
    )

    example = examples[machine_type]
    st.caption(
        f"Example hints for type {machine_type}: "
        "use them as a guide or enter your own readings."
    )

    # Clear old text inputs when the user changes machine type.
    if "previous_machine_type" not in st.session_state:
        st.session_state.previous_machine_type = machine_type

    if st.session_state.previous_machine_type != machine_type:
        for key in [
            "air_input", "process_input", "speed_input",
            "torque_input", "wear_input"
        ]:
            st.session_state.pop(key, None)

        st.session_state.previous_machine_type = machine_type

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            air_text = st.text_input(
                "Air temperature (K)",
                placeholder=example["air"],
                key="air_input"
            )
            process_text = st.text_input(
                "Process temperature (K)",
                placeholder=example["process"],
                key="process_input"
            )
            speed_text = st.text_input(
                "Rotational speed (rpm)",
                placeholder=example["speed"],
                key="speed_input"
            )

        with col2:
            torque_text = st.text_input(
                "Torque (Nm)",
                placeholder=example["torque"],
                key="torque_input"
            )
            wear_text = st.text_input(
                "Tool wear (min)",
                placeholder=example["wear"],
                key="wear_input"
            )

        predict_clicked = st.form_submit_button(
            "🚀 Predict Failure Risk",
            use_container_width=True
        )

    if predict_clicked:
        raw_values = [
            air_text.strip(),
            process_text.strip(),
            speed_text.strip(),
            torque_text.strip(),
            wear_text.strip()
        ]

        if not all(raw_values):
            st.warning(
                "Please enter all five sensor readings. "
                "Grey example hints are not actual input values."
            )
        else:
            try:
                air_temp = float(air_text)
                process_temp = float(process_text)
                rot_speed = float(speed_text)
                torque = float(torque_text)
                tool_wear = float(wear_text)

                if not all(
                    pd.notna(v)
                    for v in [
                        air_temp, process_temp, rot_speed,
                        torque, tool_wear
                    ]
                ):
                    raise ValueError("Values must be finite numbers.")

                if not (250 <= air_temp <= 350):
                    st.error("Air temperature must be between 250 and 350 K.")
                    st.stop()

                if not (250 <= process_temp <= 400):
                    st.error(
                        "Process temperature must be between 250 and 400 K."
                    )
                    st.stop()

                if not (0 <= rot_speed <= 5000):
                    st.error("Rotational speed must be between 0 and 5000 rpm.")
                    st.stop()

                if not (0 <= torque <= 200):
                    st.error("Torque must be between 0 and 200 Nm.")
                    st.stop()

                if not (0 <= tool_wear <= 300):
                    st.error("Tool wear must be between 0 and 300 minutes.")
                    st.stop()

                encoded_type = encoder.transform([machine_type])[0]

                input_values = [
                    air_temp,
                    process_temp,
                    rot_speed,
                    torque,
                    tool_wear,
                    encoded_type
                ]

                input_df = pd.DataFrame(
                    [input_values],
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
                    failure_probability = float(prediction == 1)

                probability_percent = failure_probability * 100

                st.divider()
                st.subheader("📋 Prediction Result")

                result_col, probability_col = st.columns(2)

                with result_col:
                    if prediction == 1:
                        st.error("⚠️ The model predicts MACHINE FAILURE.")
                    else:
                        st.success("✅ The model predicts NORMAL operation.")

                with probability_col:
                    st.metric(
                        "Estimated failure probability",
                        f"{probability_percent:.2f}%"
                    )
                    st.progress(
                        max(0.0, min(1.0, failure_probability))
                    )

                if probability_percent < 30:
                    st.info("🟢 Probability band: Low")
                elif probability_percent < 70:
                    st.warning("🟠 Probability band: Medium")
                else:
                    st.error("🔴 Probability band: High")

                st.caption(
                    "The Low/Medium/High bands are illustrative thresholds "
                    "for this demo, not validated industrial safety limits. "
                    "The predicted class comes from the trained model."
                )

                with st.expander("View submitted sensor values"):
                    st.dataframe(
                        pd.DataFrame({
                            "Feature": [
                                "Air temperature (K)",
                                "Process temperature (K)",
                                "Rotational speed (rpm)",
                                "Torque (Nm)",
                                "Tool wear (min)",
                                "Machine type"
                            ],
                            "Submitted value": [
                                air_temp,
                                process_temp,
                                rot_speed,
                                torque,
                                tool_wear,
                                machine_type
                            ]
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

            except ValueError as error:
                st.error(
                    "Please enter valid numeric values in all sensor fields. "
                    f"Details: {error}"
                )
            except Exception as error:
                st.error("Prediction failed. Check the model and feature files.")
                st.code(str(error))

# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================
elif page == "📊 Model Performance":
    st.markdown(
        "<div class='main-title'>📊 Model Performance</div>",
        unsafe_allow_html=True
    )
    st.write("Evaluation results from the earlier model test.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.2f}%")
    c2.metric("Precision", f"{precision:.2f}%")
    c3.metric("Recall", f"{recall:.2f}%")
    c4.metric("F1-score", f"{f1_score:.2f}%")

    st.divider()
    st.subheader("Confusion Matrix — Random Forest")

    st.caption(
        "Rows show actual labels; columns show predicted labels."
    )
    st.dataframe(
        confusion_data,
        use_container_width=True
    )

    st.write("**Interpretation**")
    st.write("- True normal predictions: 1927")
    st.write("- Normal machines incorrectly predicted as failure: 5")
    st.write("- Failures incorrectly predicted as normal: 30")
    st.write("- Failures correctly predicted: 38")

    st.warning(
        "These values are the saved example evaluation results from "
        "your earlier screenshot. They will not update automatically "
        "if you retrain or replace the model."
    )

# =========================================================
# FEATURE IMPORTANCE PAGE
# =========================================================
elif page == "🧠 Feature Importance":
    st.markdown(
        "<div class='main-title'>🧠 Feature Importance</div>",
        unsafe_allow_html=True
    )
    st.write(
        "Feature importance shows how much each input contributed "
        "to the Random Forest model's learned decisions."
    )

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.set_index("Feature"),
            horizontal=True
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "This model does not expose feature_importances_. "
            "If you use Random Forest, check that the correct model "
            "file was uploaded."
        )

# =========================================================
# ABOUT PROJECT PAGE
# =========================================================
elif page == "ℹ️ About Project":
    st.markdown(
        "<div class='main-title'>ℹ️ About PredictAI</div>",
        unsafe_allow_html=True
    )

    st.write(
        "PredictAI is a machine-learning demonstration project "
        "that estimates machine failure from sensor readings."
    )

    st.subheader("Technology Stack")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Pandas")
    st.write("- Scikit-learn Random Forest")
    st.write("- Joblib for saved model files")
    st.write("- AI4I 2020 Predictive Maintenance dataset")

    st.subheader("Input Features")
    for feature in features:
        st.write(f"- {feature}")

    st.info(
        "This is an educational demonstration, not a certified "
        "industrial safety or maintenance system."
    )

st.divider()
st.caption("PredictAI | Smart Machine Monitoring | Student Project")
