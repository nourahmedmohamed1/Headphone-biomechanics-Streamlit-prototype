import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import math
import time
from PIL import Image
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CervicalIQ — Headphone Biomechanics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #111318;
    --surface2: #1a1d24;
    --accent: #00e5ff;
    --accent2: #ff3d71;
    --accent3: #a259ff;
    --warn: #ffaa00;
    --ok: #00e676;
    --text: #e8eaf0;
    --muted: #6b7280;
    --border: rgba(0,229,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Slider */
.stSlider > div > div > div { background: var(--accent) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2);
    border-radius: 10px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #000 !important;
    border-radius: 8px;
}

/* Cards */
.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Risk badges */
.badge-low  { background:#00e67622; color:var(--ok);   border:1px solid var(--ok);   border-radius:8px; padding:4px 14px; font-family:'Space Mono',monospace; font-size:0.85rem; }
.badge-med  { background:#ffaa0022; color:var(--warn);  border:1px solid var(--warn);  border-radius:8px; padding:4px 14px; font-family:'Space Mono',monospace; font-size:0.85rem; }
.badge-high { background:#ff3d7122; color:var(--accent2); border:1px solid var(--accent2); border-radius:8px; padding:4px 14px; font-family:'Space Mono',monospace; font-size:0.85rem; }

/* Hero */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent3) 60%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.step-num {
    display: inline-block;
    width: 32px; height: 32px;
    background: linear-gradient(135deg,var(--accent),var(--accent3));
    color: #000;
    font-weight: 700;
    font-family: 'Space Mono',monospace;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    margin-right: 10px;
    flex-shrink: 0;
}
.step-row { display:flex; align-items:center; margin-bottom:14px; }
.gauge-container { text-align:center; }
hr { border-color: var(--border); }

/* Progress bar override */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent3)); }

/* Expander */
details { background: var(--surface2); border:1px solid var(--border); border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
#  BIOMECHANICS ENGINE
# ─────────────────────────────────────────────
def calculate_cva(landmarks, image_shape):
    """
    Craniovertebral Angle (CVA): angle between horizontal at C7
    and the line from C7 to Tragus (ear landmark).
    """
    h, w = image_shape[:2]
    # C7 approximated by midpoint of shoulders
    lsh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rsh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    # Ear (tragus proxy)
    lear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    rear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

    c7_x = (lsh.x + rsh.x) / 2 * w
    c7_y = (lsh.y + rsh.y) / 2 * h
    # Use the more visible ear
    if lear.visibility > rear.visibility:
        ear_x, ear_y = lear.x * w, lear.y * h
    else:
        ear_x, ear_y = rear.x * w, rear.y * h

    dx = ear_x - c7_x
    dy = c7_y - ear_y  # inverted Y in image space
    cva = math.degrees(math.atan2(dy, abs(dx)))
    return max(0, min(90, cva)), (c7_x, c7_y), (ear_x, ear_y)


def neck_load_lbs(cva_deg, headphone_g):
    """
    Spinal load model based on Hansraj (2014).
    Neutral (CVA~50°+) ≈ 10-12 lbs; at 60° forward tilt → ~60 lbs.
    Linear interpolation + headphone additive load.
    """
    # Forward tilt angle from vertical = 90 - CVA (approx)
    tilt = max(0, 90 - cva_deg)
    # Hansraj model: load_lbs ≈ 10 + (tilt/60)*50
    base_load = 10 + (tilt / 60) * 50
    # Convert headphone weight (grams) to lbs additive (lever arm ≈ 0.3)
    hp_load = (headphone_g / 453.592) * 0.3 * (1 + tilt / 30)
    return round(base_load + hp_load, 1)


def risk_level(cva_deg):
    if cva_deg >= 50:
        return "LOW", "#00e676", "badge-low"
    elif cva_deg >= 35:
        return "MEDIUM", "#ffaa00", "badge-med"
    else:
        return "HIGH", "#ff3d71", "badge-high"


def draw_overlay(image, landmarks, cva, c7_pt, ear_pt, load_lbs, risk):
    """Draw biomechanics overlay on frame."""
    h, w = image.shape[:2]
    c7 = (int(c7_pt[0]), int(c7_pt[1]))
    ear = (int(ear_pt[0]), int(ear_pt[1]))

    # Lines
    color_map = {"LOW": (0, 230, 118), "MEDIUM": (255, 170, 0), "HIGH": (255, 61, 113)}
    col = color_map[risk]
    cv2.line(image, c7, ear, col, 2)
    # Horizontal reference
    cv2.line(image, (c7[0] - 40, c7[1]), (c7[0] + 40, c7[1]), (100, 200, 255), 1)

    # Circles at keypoints
    cv2.circle(image, c7, 8, (0, 229, 255), -1)
    cv2.circle(image, ear, 8, (162, 89, 255), -1)

    # CVA label
    cv2.putText(image, f"CVA: {cva:.1f}deg", (c7[0] + 15, c7[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(image, f"Load: {load_lbs} lbs", (c7[0] + 15, c7[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)

    # Risk badge in corner
    cv2.rectangle(image, (10, 10), (160, 50), (17, 19, 36), -1)
    cv2.putText(image, f"RISK: {risk}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return image


def process_frame(frame_bgr, headphone_g):
    """Run MediaPipe pose on a BGR frame, return annotated frame + metrics."""
    result = {"cva": None, "load": None, "risk": None, "color": None, "badge": None}
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            cva, c7, ear = calculate_cva(lm, frame_bgr.shape)
            load = neck_load_lbs(cva, headphone_g)
            risk, color_hex, badge = risk_level(cva)
            col_bgr = (int(color_hex[5:7], 16), int(color_hex[3:5], 16), int(color_hex[1:3], 16))
            annotated = draw_overlay(frame_bgr.copy(), lm, cva, c7, ear, load, risk)
            result = {"cva": cva, "load": load, "risk": risk,
                      "color": color_hex, "badge": badge, "frame": annotated}
        else:
            result["frame"] = frame_bgr
    return result


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 20px'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                    color:#00e5ff;letter-spacing:0.15em;margin-bottom:4px'>
            CERVICAL IQ
        </div>
        <div style='font-size:1.4rem;font-weight:800'>🎧 BiomechanicAI</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "**SELECT MODE**",
        ["🏠 Home / Guide", "📷 Live Simulation", "📁 Upload Video / Image"],
        index=0
    )

    st.markdown("---")
    st.markdown("**⚙️ HEADPHONE WEIGHT**")
    headphone_weight = st.slider(
        "Weight (grams)", min_value=10, max_value=500,
        value=250, step=5,
        help="Adjust the headphone weight to see how it affects neck load"
    )
    st.markdown(f"""
    <div class='card' style='padding:12px;margin-top:8px'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                    color:#6b7280;margin-bottom:6px'>SELECTED WEIGHT</div>
        <div style='font-size:1.6rem;font-weight:800;color:#00e5ff'>{headphone_weight}g</div>
        <div style='font-size:0.75rem;color:#6b7280'>{headphone_weight/1000:.3f} kg</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📐 Biomechanics Reference"):
        st.markdown("""
        **CVA Thresholds (Hansraj 2014)**
        | CVA | Tilt | Spinal Load |
        |-----|------|------------|
        | >50° | Neutral | ~10–12 lbs |
        | ~35–50° | Moderate | ~27–40 lbs |
        | <35° | Severe | ~49–60 lbs |
        """)

# ─────────────────────────────────────────────
#  HOME PAGE
# ─────────────────────────────────────────────
if "Home" in mode:
    st.markdown("""
    <div style='padding: 40px 0 20px'>
        <div class='hero-title'>Cervical IQ</div>
        <div class='hero-sub'>Real-Time Head Posture & Neck Load Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### How to Start")
        steps = [
            ("Choose Mode", "Select **Live Simulation** for real-time camera analysis or **Upload** for offline video/image."),
            ("Set Headphone Weight", "Use the sidebar slider to enter the gram weight of your headphones (10–500g)."),
            ("Position Camera", "Ensure your full upper body (shoulders to head) is visible in side profile for best accuracy."),
            ("Read Results", "Monitor your CVA angle, neck load estimate, and Risk Level in real time."),
            ("Adjust & Improve", "Tilt your head back toward neutral to see load drop. A CVA > 50° = healthy posture."),
        ]
        for i, (title, desc) in enumerate(steps):
            st.markdown(f"""
            <div class='step-row'>
                <span class='step-num'>{i+1}</span>
                <div>
                    <strong>{title}</strong><br>
                    <span style='color:#9ca3af;font-size:0.9rem'>{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### The Science")
        st.markdown("""
        <div class='card'>
            <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                        color:#00e5ff;margin-bottom:10px'>LOAD MODEL · HANSRAJ 2014</div>
            <div style='font-size:2.4rem;font-weight:800;margin-bottom:4px'>60 lbs</div>
            <div style='color:#9ca3af;font-size:0.85rem'>
                of cervical spine stress at a mere <strong style='color:#ff3d71'>60° forward tilt</strong>
            </div>
            <hr style='margin:14px 0'>
            <div style='display:flex;gap:16px;justify-content:space-around;text-align:center'>
                <div>
                    <div style='font-size:1.4rem;font-weight:800;color:#00e676'>12 lbs</div>
                    <div style='font-size:0.75rem;color:#6b7280'>Neutral</div>
                </div>
                <div>
                    <div style='font-size:1.4rem;font-weight:800;color:#ffaa00'>32 lbs</div>
                    <div style='font-size:0.75rem;color:#6b7280'>30° tilt</div>
                </div>
                <div>
                    <div style='font-size:1.4rem;font-weight:800;color:#ff3d71'>60 lbs</div>
                    <div style='font-size:0.75rem;color:#6b7280'>60° tilt</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#a259ff;margin-bottom:10px'>WHAT WE TRACK</div>
            <div style='font-size:0.9rem;line-height:1.8'>
                🔵 <strong>Tragus</strong> — ear canal landmark<br>
                🟣 <strong>C7 Vertebra</strong> — shoulder midpoint proxy<br>
                📐 <strong>CVA</strong> — Craniovertebral Angle<br>
                ⚡ <strong>Load Formula</strong> — Hansraj spinal stress model
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Risk Level Guide")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class='card' style='border-color:rgba(0,230,118,0.3);text-align:center'>
            <div style='font-size:2rem'>🟢</div>
            <div style='font-weight:800;font-size:1.1rem;color:#00e676'>LOW RISK</div>
            <div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#6b7280'>CVA ≥ 50°</div>
            <div style='font-size:0.85rem;margin-top:8px;color:#9ca3af'>
                Good posture. Neck load near neutral ~10–22 lbs. Keep it up!
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class='card' style='border-color:rgba(255,170,0,0.3);text-align:center'>
            <div style='font-size:2rem'>🟡</div>
            <div style='font-weight:800;font-size:1.1rem;color:#ffaa00'>MEDIUM RISK</div>
            <div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#6b7280'>CVA 35°–50°</div>
            <div style='font-size:0.85rem;margin-top:8px;color:#9ca3af'>
                Moderate forward head. 27–40 lbs load. Take breaks, adjust screen height.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class='card' style='border-color:rgba(255,61,113,0.3);text-align:center'>
            <div style='font-size:2rem'>🔴</div>
            <div style='font-weight:800;font-size:1.1rem;color:#ff3d71'>HIGH RISK</div>
            <div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#6b7280'>CVA < 35°</div>
            <div style='font-size:0.85rem;margin-top:8px;color:#9ca3af'>
                Severe forward tilt. 49–60+ lbs load. Risk of chronic neck pain. Act now!
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LIVE SIMULATION (WebRTC via streamlit-webrtc)
# ─────────────────────────────────────────────
elif "Live" in mode:
    st.markdown("## 📷 Live Simulation")
    st.markdown("""
    <div class='card' style='border-color:rgba(0,229,255,0.3);margin-bottom:20px'>
        <strong style='color:#00e5ff'>💡 Setup Tip:</strong>
        Position yourself in <strong>side profile</strong> so both your ear and shoulder are clearly visible.
        Ensure good lighting for best MediaPipe tracking accuracy.
    </div>
    """, unsafe_allow_html=True)

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av

        RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class BiomechanicsProcessor(VideoProcessorBase):
            def __init__(self):
                self.headphone_g = 250
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.last_metrics = {}

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.pose.process(img_rgb)
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    cva, c7, ear = calculate_cva(lm, img.shape)
                    load = neck_load_lbs(cva, self.headphone_g)
                    risk, color_hex, _ = risk_level(cva)
                    img = draw_overlay(img, lm, cva, c7, ear, load, risk)
                    self.last_metrics = {"cva": cva, "load": load, "risk": risk}
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        col_cam, col_dash = st.columns([3, 2], gap="large")

        with col_cam:
            ctx = webrtc_streamer(
                key="biomechanics",
                video_processor_factory=BiomechanicsProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                ctx.video_processor.headphone_g = headphone_weight

        with col_dash:
            st.markdown("### 📊 Live Metrics")
            if ctx.video_processor and ctx.video_processor.last_metrics:
                m = ctx.video_processor.last_metrics
                cva = m["cva"]
                load = m["load"]
                risk, color, badge = risk_level(cva)

                st.metric("Craniovertebral Angle (CVA)", f"{cva:.1f}°")
                st.metric("Neck Load Estimate", f"{load} lbs")
                st.markdown(f"**Risk Level:** <span class='{badge}'>{risk}</span>", unsafe_allow_html=True)
                st.progress(min(int((90 - cva) / 90 * 100), 100))

                tilt = 90 - cva
                st.markdown(f"""
                <div class='card' style='margin-top:16px'>
                    <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                                color:#6b7280;margin-bottom:8px'>FORWARD HEAD TILT</div>
                    <div style='font-size:2rem;font-weight:800;color:{color}'>{tilt:.1f}°</div>
                    <div style='font-size:0.8rem;color:#6b7280'>from vertical</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Start the camera stream to see live metrics.")
                _demo_cva_val = 42.0
                _demo_load_val = neck_load_lbs(_demo_cva_val, headphone_weight)
                _risk_d, _col_d, _badge_d = risk_level(_demo_cva_val)
                st.metric("CVA (demo)", f"{_demo_cva_val}°")
                st.metric("Neck Load (demo)", f"{_demo_load_val} lbs")
                st.markdown(f"Risk (demo): <span class='{_badge_d}'>{_risk_d}</span>", unsafe_allow_html=True)

    except ImportError:
        st.warning("⚠️ `streamlit-webrtc` not installed. Showing simulated live demo instead.")
        _sim_cva = st.slider("Simulate CVA angle", 10, 80, 45, key="sim_cva")
        _sim_load = neck_load_lbs(_sim_cva, headphone_weight)
        _r, _c, _b = risk_level(_sim_cva)

        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.markdown(f"""
            <div class='card' style='text-align:center;padding:60px 20px;border-color:{_c}40'>
                <div style='font-size:5rem'>🎧</div>
                <div style='font-family:Space Mono,monospace;font-size:0.85rem;
                            color:{_c};margin-top:10px'>SIMULATION MODE</div>
                <div style='font-size:1rem;color:#6b7280;margin-top:8px'>
                    Move the CVA slider to simulate different postures
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("### 📊 Simulated Metrics")
            st.metric("Craniovertebral Angle", f"{_sim_cva}°")
            st.metric("Neck Load", f"{_sim_load} lbs")
            st.metric("Headphone Weight", f"{headphone_weight}g")
            st.markdown(f"**Risk:** <span class='{_b}'>{_r}</span>", unsafe_allow_html=True)
            st.progress(min(int((90 - _sim_cva) / 90 * 100), 100))

# ─────────────────────────────────────────────
#  UPLOAD VIDEO / IMAGE
# ─────────────────────────────────────────────
elif "Upload" in mode:
    st.markdown("## 📁 Upload Video / Image")

    tab_img, tab_vid = st.tabs(["🖼️  Image Analysis", "🎬  Video Analysis"])

    # ── IMAGE TAB ─────────────────────────────
    with tab_img:
        uploaded_img = st.file_uploader(
            "Upload a side-profile photo (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_upload"
        )
        if uploaded_img:
            img_pil = Image.open(uploaded_img).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing posture with MediaPipe..."):
                result = process_frame(img_bgr, headphone_weight)

            col_orig, col_anno = st.columns(2, gap="medium")
            with col_orig:
                st.markdown("**Original**")
                st.image(img_pil, use_container_width=True)
            with col_anno:
                st.markdown("**Annotated**")
                if "frame" in result:
                    st.image(cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB),
                             use_container_width=True)

            if result["cva"] is not None:
                st.markdown("---")
                st.markdown("### Analysis Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("CVA Angle", f"{result['cva']:.1f}°")
                m2.metric("Forward Tilt", f"{90 - result['cva']:.1f}°")
                m3.metric("Neck Load", f"{result['load']} lbs")
                m4.metric("HP Weight", f"{headphone_weight}g")

                risk, color, badge = risk_level(result["cva"])
                st.markdown(f"""
                <div class='card' style='border-color:{color}40;margin-top:16px'>
                    <div style='display:flex;align-items:center;gap:20px'>
                        <div style='font-size:3rem'>
                            {"🟢" if risk=="LOW" else "🟡" if risk=="MEDIUM" else "🔴"}
                        </div>
                        <div>
                            <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                                        color:#6b7280'>RISK ASSESSMENT</div>
                            <span class='{badge}'>{risk} RISK</span>
                            <div style='font-size:0.85rem;color:#9ca3af;margin-top:6px'>
                                {'Excellent posture — keep it up!' if risk=='LOW' 
                                 else 'Moderate forward head posture. Consider ergonomic adjustments.' if risk=='MEDIUM'
                                 else 'Severe forward head posture. High cervical spine stress detected!'}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No pose detected. Try a clearer side-profile photo with good lighting.")

    # ── VIDEO TAB ─────────────────────────────
    with tab_vid:
        uploaded_vid = st.file_uploader(
            "Upload a side-profile video (MP4, MOV, AVI)",
            type=["mp4", "mov", "avi"],
            key="vid_upload"
        )
        if uploaded_vid:
            import tempfile, os
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()
            tfile.close()

            st.video(tfile.name)

            if st.button("▶ Analyze Video", key="analyze_vid"):
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                sample_every = max(1, int(fps // 2))  # 2 samples/sec

                cva_list, load_list = [], []
                progress_bar = st.progress(0)
                status = st.empty()
                frame_idx = 0
                processed = 0

                with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                  min_detection_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % sample_every == 0:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = pose.process(rgb)
                            if res.pose_landmarks:
                                cva, _, _ = calculate_cva(res.pose_landmarks.landmark, frame.shape)
                                load = neck_load_lbs(cva, headphone_weight)
                                cva_list.append(cva)
                                load_list.append(load)
                            processed += 1
                        frame_idx += 1
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        status.markdown(f"Processing frame {frame_idx}/{total_frames}…")

                cap.release()
                os.unlink(tfile.name)
                progress_bar.empty()
                status.empty()

                if cva_list:
                    avg_cva = np.mean(cva_list)
                    min_cva = np.min(cva_list)
                    avg_load = np.mean(load_list)
                    max_load = np.max(load_list)
                    risk, color, badge = risk_level(avg_cva)

                    st.markdown("### Video Analysis Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg CVA", f"{avg_cva:.1f}°")
                    c2.metric("Min CVA (worst)", f"{min_cva:.1f}°")
                    c3.metric("Avg Load", f"{avg_load:.1f} lbs")
                    c4.metric("Peak Load", f"{max_load:.1f} lbs")

                    st.markdown(f"**Overall Risk:** <span class='{badge}'>{risk}</span>",
                                unsafe_allow_html=True)

                    # Simple ASCII-style chart via st.line_chart
                    import pandas as pd
                    df = pd.DataFrame({"CVA (°)": cva_list, "Neck Load (lbs)": load_list})
                    st.markdown("#### CVA & Load Over Time")
                    st.line_chart(df, height=200)
                else:
                    st.warning("No pose landmarks detected in video. Check lighting and camera angle.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:20px 0;font-family:Space Mono,monospace;
            font-size:0.7rem;color:#374151'>
    CervicalIQ · Biomechanics model based on Hansraj (2014) · For educational use only<br>
    Built with MediaPipe · Streamlit · OpenCV
</div>
""", unsafe_allow_html=True)
