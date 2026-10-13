
import streamlit as st
import base64

st.set_page_config(
    page_title="PredictAI | Secure Login",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- LOGIN SETTINGS ----------
# Demo credentials: change these before deploying publicly.
USERNAME = st.secrets.get("PREDICTAI_USERNAME", "admin")
PASSWORD = st.secrets.get("PREDICTAI_PASSWORD", "PredictAI@123")

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap');

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="collapsedControl"] {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 20%, rgba(0, 160, 255, 0.16), transparent 42%),
        radial-gradient(ellipse at 90% 80%, rgba(0, 255, 210, 0.09), transparent 38%),
        linear-gradient(135deg, #020916 0%, #07172b 48%, #020711 100%);
    color: #eef6ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #f4fbff;
}

.brand span {
    color: #00d9ff;
}

.top-label {
    color: #91a9c4;
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 700;
    margin-top: 8px;
}

.status {
    display: inline-block;
    border: 1px solid rgba(0, 232, 180, 0.35);
    background: rgba(0, 232, 180, 0.07);
    color: #00edbd;
    border-radius: 30px;
    padding: 8px 13px;
    font-size: 11px;
    letter-spacing: 1px;
}

.hero {
    padding: 50px 10px 20px 0;
}

.eyebrow {
    color: #00d9ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 4px;
    margin-bottom: 18px;
}

.hero h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(36px, 4vw, 60px);
    line-height: 1.17;
    font-weight: 800;
    color: #f5faff;
    margin: 0 0 22px 0;
}

.hero h1 span {
    color: #00d9ff;
    text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
}

.hero p {
    color: #9bb0c9;
    font-size: 15px;
    line-height: 1.9;
    max-width: 540px;
}

.factory {
    margin-top: 32px;
    min-height: 220px;
    border: 1px solid rgba(0, 207, 255, 0.22);
    border-radius: 18px;
    padding: 22px;
    background:
        linear-gradient(120deg, rgba(0, 22, 46, 0.92), rgba(3, 45, 73, 0.45)),
        repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent 48px,
            rgba(0, 217, 255, 0.06) 49px,
            transparent 50px
        );
    box-shadow: 0 0 40px rgba(0, 126, 255, 0.08);
}

.factory-title {
    font-size: 11px;
    color: #7f9cb9;
    letter-spacing: 2px;
    font-weight: 700;
}

.factory-emoji {
    font-size: 52px;
    margin: 16px 0 12px;
    filter: drop-shadow(0 0 12px rgba(0, 217, 255, 0.4));
}

.machine-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
}

.machine {
    border: 1px solid rgba(0, 217, 255, 0.25);
    border-radius: 10px;
    padding: 10px 13px;
    color: #b5d4ec;
    background: rgba(0, 217, 255, 0.045);
    font-size: 11px;
}

.machine b {
    color: #00edbd;
}

.login-card {
    background: linear-gradient(
        145deg,
        rgba(13, 31, 53, 0.96),
        rgba(5, 17, 33, 0.98)
    );
    border: 1px solid rgba(0, 217, 255, 0.24);
    border-radius: 22px;
    padding: 30px 30px 22px;
    box-shadow:
        0 20px 70px rgba(0, 0, 0, 0.32),
        0 0 35px rgba(0, 174, 255, 0.06);
}

.login-tag {
    color: #00edbd;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
}

.login-title {
    color: #f3f8ff;
    font-family: 'Orbitron', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin: 13px 0 8px;
}

.login-subtitle {
    color: #8ea6c1;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 24px;
}

div[data-testid="stTextInput"] label {
    color: #c4d5e8 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

div[data-testid="stTextInput"] input {
    background: #071629 !important;
    color: #f5faff !important;
    border: 1px solid #24415f !important;
    border-radius: 10px !important;
    min-height: 48px !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #00d9ff !important;
    box-shadow: 0 0 0 1px #00d9ff !important;
}

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 48px;
    border: 0;
    border-radius: 10px;
    background: linear-gradient(90deg, #007dff, #00c9ef);
    color: #001323;
    font-weight: 800;
    letter-spacing: 1px;
    transition: 0.2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 0 25px rgba(0, 217, 255, 0.25);
    color: #001323;
    border: 0;
}

.security {
    text-align: center;
    color: #7e95ae;
    font-size: 11px;
    line-height: 1.8;
    margin-top: 18px;
}

.footer {
    text-align: center;
    color: #607891;
    font-size: 11px;
    padding-top: 35px;
    letter-spacing: 0.5px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    .hero {
        padding-top: 20px;
    }
    .login-card {
        padding: 24px 20px;
    }
    .hero h1 {
        font-size: 36px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">PREDICT<span>AI</span></div>'
        '<div class="top-label">INTELLIGENT INDUSTRIAL MONITORING</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;padding-top:8px;">'
        '<span class="status">● SYSTEM ONLINE</span></div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div style="height:1px;background:linear-gradient(90deg,'
    'rgba(0,217,255,.45),rgba(0,217,255,.02));margin:22px 0 20px;"></div>',
    unsafe_allow_html=True
)

# ---------- PAGE LAYOUT ----------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">NEXT-GEN PREDICTIVE MAINTENANCE</div>
        <h1>THE FUTURE OF<br>FACTORY<br><span>INTELLIGENCE.</span></h1>
        <p>
            Predict machine failures before they happen.
            Monitor industrial health, discover hidden patterns,
            and make smarter maintenance decisions with AI.
        </p>
    </div>

    <div class="factory">
        <div class="factory-title">PREDICTAI // FACTORY CONTROL CENTER</div>
        <div class="factory-emoji">🏭 ⚙️ 🤖</div>
        <div style="color:#e9f7ff;font-size:17px;font-weight:700;">
            Your Factory. Under Control.
        </div>
        <div style="color:#8da9c4;font-size:12px;margin-top:8px;">
            AI-powered monitoring • Predictive insights • Smarter operations
        </div>
        <div class="machine-row">
            <div class="machine">M-01 &nbsp; <b>● HEALTHY</b></div>
            <div class="machine">M-02 &nbsp; <b style="color:#ffc857">● MONITOR</b></div>
            <div class="machine">M-03 &nbsp; <b>● HEALTHY</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div style="height:34px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="login-card">
        <div class="login-tag">SECURE ACCESS PORTAL</div>
        <div class="login-title">Welcome Back</div>
        <div class="login-subtitle">
            Sign in to access your PredictAI dashboard
            and monitor your machines.
        </div>
    """, unsafe_allow_html=True)

    # ---------- LOGIN FORM ----------
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("predictai_login", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )
            submitted = st.form_submit_button("SIGN IN  →", use_container_width=True)

        if submitted:
            if username == USERNAME and password == PASSWORD:
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")

        st.markdown("""
        <div class="security">
            🔒 Protected access to your industrial intelligence workspace.<br>
            Authorized users only.
        </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Demo login details"):
            st.write("Username: `admin`")
            st.write("Password: `PredictAI@123`")
            st.caption(
                "Demo credentials are for testing only. Configure Streamlit "
                "Secrets before deploying publicly."
            )

    else:
        st.success("Login successful!")
        st.markdown(
            f"Welcome, **{st.session_state.get('current_user', 'User')}**."
        )
        st.info(
            "Your login page is working. Connect your existing PredictAI "
            "dashboard code in this section to display the dashboard after login."
        )

        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.pop("current_user", None)
            st.rerun()

st.markdown("""
<div class="footer">
    PREDICTAI © 2026 &nbsp; • &nbsp; AI-POWERED MACHINE HEALTH INTELLIGENCE
    <br><br>
    Built for smarter, safer and more efficient manufacturing.
</div>
""", unsafe_allow_html=True)
