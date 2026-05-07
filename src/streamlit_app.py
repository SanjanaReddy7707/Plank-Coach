"""
T6.5 — Plank Timer + Form Checker
SMAI Assignment 3 · IIIT Hyderabad 2025-26

Uses MediaPipe Pose Landmarker (Tasks API) to:
  - Detect 33 body keypoints
  - Check hip-line straightness (shoulder → hip → ankle angle)
  - Check head alignment (ear → shoulder angle)
  - Time how long valid form is held
  - Works on uploaded images AND videos
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode
)
from mediapipe.tasks.python import BaseOptions
from PIL import Image
import time
import os
import tempfile
import urllib.request

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Plank Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — Dark athletic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --card:      #1a1a26;
    --border:    #2a2a3a;
    --accent:    #00e5a0;
    --accent2:   #ff4060;
    --accent3:   #ffd166;
    --text:      #e8e8f0;
    --muted:     #6a6a8a;
    --good:      #00e5a0;
    --warn:      #ffd166;
    --bad:       #ff4060;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }

/* Headings */
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; }

/* Hero title */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 8vw, 6rem);
    line-height: 1;
    background: linear-gradient(135deg, var(--accent) 0%, #00b4d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.08em;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    line-height: 1;
    color: var(--accent);
}
.metric-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
}
.badge-good  { background: rgba(0,229,160,0.15); color: var(--good);  border: 1px solid var(--good); }
.badge-warn  { background: rgba(255,209,102,0.15); color: var(--warn); border: 1px solid var(--warn); }
.badge-bad   { background: rgba(255,64,96,0.15);  color: var(--bad);  border: 1px solid var(--bad); }

/* Issue list */
.issue-item {
    background: rgba(255,64,96,0.08);
    border-left: 3px solid var(--bad);
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: #ffb0bc;
}

/* Angle display */
.angle-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.angle-name { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; }
.angle-val  { font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; }

/* Instruction cards */
.tip-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: var(--text);
    line-height: 1.6;
}
.tip-card .tip-icon { font-size: 1.4rem; margin-bottom: 0.4rem; display: block; }
.tip-card strong { color: var(--accent); }

/* Section divider */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.1em !important;
    font-size: 1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Uploader */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* Progress bar */
.prog-wrap {
    background: var(--surface);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.1em !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Hide streamlit cruft */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "/home/user/pose_landmarker_lite.task"

BODY_ANGLE_MIN  = 160   # degrees — below this = hip sag
BODY_ANGLE_MAX  = 195   # degrees — above this = hip pike
HEAD_ANGLE_MIN  = 145   # ear-shoulder-hip — below this = head drooping
GOOD_HOLD_SECS  = 30    # seconds considered a solid plank

# Landmark indices (from PoseLandmark enum)
LANDMARKS = {
    "left_ear":      7,
    "right_ear":     8,
    "left_shoulder": 11,
    "right_shoulder":12,
    "left_hip":      23,
    "right_hip":     24,
    "left_knee":     25,
    "right_knee":    26,
    "left_ankle":    27,
    "right_ankle":   28,
    "left_elbow":    13,
    "right_elbow":   14,
    "left_wrist":    15,
    "right_wrist":   16,
}

# Skeleton connections for drawing
CONNECTIONS = [
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pose_model():
    """Download (once) and load the MediaPipe pose landmarker."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        with st.spinner("⬇️ Downloading pose model (first run only, ~3 MB)…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────
def angle_between(a, b, c):
    """Angle at point b, formed by a-b-c (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def get_xy(landmark, w, h):
    return [landmark.x * w, landmark.y * h]


def get_visibility(landmark):
    return landmark.visibility if hasattr(landmark, "visibility") else 1.0


# ─────────────────────────────────────────────
# Plank form analysis
# ─────────────────────────────────────────────
def analyse_plank(landmarks, w, h):
    """
    Returns:
        is_good     : bool
        body_angle  : float  (shoulder-hip-ankle)
        head_angle  : float  (ear-shoulder-hip)
        issues      : list[str]
        side        : 'left' | 'right' | 'avg'
    """
    lm = landmarks

    def lm_xy(name):
        idx = LANDMARKS[name]
        return get_xy(lm[idx], w, h)

    def visibility(name):
        idx = LANDMARKS[name]
        return get_visibility(lm[idx])

    # Choose best side (more visible)
    left_vis  = visibility("left_shoulder")  + visibility("left_hip")  + visibility("left_ankle")
    right_vis = visibility("right_shoulder") + visibility("right_hip") + visibility("right_ankle")
    side = "left" if left_vis >= right_vis else "right"

    shoulder = lm_xy(f"{side}_shoulder")
    hip      = lm_xy(f"{side}_hip")
    ankle    = lm_xy(f"{side}_ankle")
    ear      = lm_xy(f"{side}_ear")
    elbow    = lm_xy(f"{side}_elbow")
    wrist    = lm_xy(f"{side}_wrist")

    body_angle = angle_between(shoulder, hip, ankle)
    head_angle = angle_between(ear, shoulder, hip)
    arm_angle  = angle_between(shoulder, elbow, wrist)

    issues = []

    if body_angle < BODY_ANGLE_MIN:
        drop = BODY_ANGLE_MIN - body_angle
        issues.append(f"Hip sagging ↓  ({body_angle:.0f}° — lift by ~{drop:.0f}°)")
    elif body_angle > BODY_ANGLE_MAX:
        rise = body_angle - BODY_ANGLE_MAX
        issues.append(f"Hip piked ↑  ({body_angle:.0f}° — lower by ~{rise:.0f}°)")

    if head_angle < HEAD_ANGLE_MIN:
        issues.append(f"Head drooping — keep neck neutral ({head_angle:.0f}°)")

    is_good = len(issues) == 0
    return is_good, body_angle, head_angle, arm_angle, issues, side


# ─────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────
def draw_overlay(frame, landmarks, w, h, is_good, body_angle, head_angle):
    """Draw skeleton + angle arcs on the frame."""
    img = frame.copy()
    lm  = landmarks

    good_colour = (0, 229, 160)   # green
    bad_colour  = (64, 64, 255)   # red (BGR)
    node_colour = (255, 255, 255)
    line_colour = good_colour if is_good else bad_colour

    def pt(name):
        idx = LANDMARKS[name]
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Draw skeleton connections
    for a, b in CONNECTIONS:
        try:
            cv2.line(img, pt(a), pt(b), line_colour, 2, cv2.LINE_AA)
        except Exception:
            pass

    # Draw keypoint dots
    for name in LANDMARKS:
        try:
            x, y = pt(name)
            cv2.circle(img, (x, y), 5, node_colour, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 5, line_colour,  1, cv2.LINE_AA)
        except Exception:
            pass

    # Body angle label at hip
    try:
        hx, hy = pt("left_hip") if LANDMARKS["left_hip"] else pt("right_hip")
        colour = good_colour if BODY_ANGLE_MIN <= body_angle <= BODY_ANGLE_MAX else bad_colour
        cv2.putText(img, f"{body_angle:.0f}deg", (hx + 8, hy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, colour, 1, cv2.LINE_AA)
    except Exception:
        pass

    # Status banner
    banner_text = "GOOD FORM" if is_good else "FIX FORM"
    banner_col  = good_colour if is_good else bad_colour
    cv2.rectangle(img, (0, 0), (220, 42), (20, 20, 30), -1)
    cv2.putText(img, banner_text, (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, banner_col, 1, cv2.LINE_AA)

    return img


# ─────────────────────────────────────────────
# Process a single image
# ─────────────────────────────────────────────
def process_image(pil_img, detector):
    frame = np.array(pil_img.convert("RGB"))
    h, w  = frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result   = detector.detect(mp_image)

    if not result.pose_landmarks:
        return frame, None, None, None, None, ["No person detected — make sure your full body is visible"], None

    landmarks  = result.pose_landmarks[0]
    is_good, body_angle, head_angle, arm_angle, issues, side = analyse_plank(landmarks, w, h)
    annotated  = draw_overlay(frame, landmarks, w, h, is_good, body_angle, head_angle)
    return annotated, is_good, body_angle, head_angle, arm_angle, issues, side


# ─────────────────────────────────────────────
# Process video
# ─────────────────────────────────────────────
def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_results = []   # (is_good, body_angle, head_angle, issues)
    annotated_frames = []
    sample_rate = max(1, int(fps / 10))  # process 10 frames/sec max (speed)

    frame_idx = 0
    progress = st.progress(0, text="Analysing video…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                is_good, body_angle, head_angle, arm_angle, issues, side = analyse_plank(landmarks, w, h)
                annotated = draw_overlay(rgb, landmarks, w, h, is_good, body_angle, head_angle)
                frame_results.append((is_good, body_angle, head_angle, issues))
            else:
                annotated = rgb
                frame_results.append((False, 0, 0, ["No pose detected"]))

            annotated_frames.append(annotated)
            progress.progress(min(frame_idx / max(total_frames, 1), 1.0),
                              text=f"Analysing frame {frame_idx}/{total_frames}…")

        frame_idx += 1

    cap.release()
    progress.empty()
    return frame_results, annotated_frames, fps / sample_rate


# ─────────────────────────────────────────────
# Render results panel
# ─────────────────────────────────────────────
def render_image_results(is_good, body_angle, head_angle, arm_angle, issues, annotated):
    col_img, col_info = st.columns([3, 2], gap="medium")

    with col_img:
        st.image(annotated, use_container_width=True, channels="RGB")

    with col_info:
        # Overall status
        if is_good:
            st.markdown('<span class="badge badge-good">✓ GOOD FORM</span>', unsafe_allow_html=True)
            st.markdown("**Your plank alignment looks solid!** Keep holding and breathe steadily.")
        else:
            st.markdown('<span class="badge badge-bad">✗ FIX FORM</span>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Angles</div>', unsafe_allow_html=True)

        # Body angle
        body_ok = BODY_ANGLE_MIN <= body_angle <= BODY_ANGLE_MAX
        body_col = "#00e5a0" if body_ok else "#ff4060"
        st.markdown(f"""
        <div class="angle-block">
            <span class="angle-name">🔵 Body line<br>(shoulder→hip→ankle)</span>
            <span class="angle-val" style="color:{body_col}">{body_angle:.1f}°</span>
        </div>
        """, unsafe_allow_html=True)

        # Head angle
        head_ok  = head_angle >= HEAD_ANGLE_MIN
        head_col = "#00e5a0" if head_ok else "#ffd166"
        st.markdown(f"""
        <div class="angle-block">
            <span class="angle-name">🟡 Head angle<br>(ear→shoulder→hip)</span>
            <span class="angle-val" style="color:{head_col}">{head_angle:.1f}°</span>
        </div>
        """, unsafe_allow_html=True)

        # Arm angle
        if arm_angle:
            st.markdown(f"""
            <div class="angle-block">
                <span class="angle-name">⚪ Arm angle<br>(shoulder→elbow→wrist)</span>
                <span class="angle-val" style="color:#aaa">{arm_angle:.1f}°</span>
            </div>
            """, unsafe_allow_html=True)

        # Issues
        if issues:
            st.markdown('<div class="section-title">What to fix</div>', unsafe_allow_html=True)
            for issue in issues:
                st.markdown(f'<div class="issue-item">⚠ {issue}</div>', unsafe_allow_html=True)

        # Target range reminder
        st.markdown("""
        <div class="section-title">Target range</div>
        <div style="font-size:0.82rem; color:#6a6a8a; line-height:1.8;">
        Body line: <strong style="color:#e8e8f0">160° – 195°</strong><br>
        Head angle: <strong style="color:#e8e8f0">&gt; 145°</strong><br>
        Ideal body line: <strong style="color:#00e5a0">~175°–180°</strong>
        </div>
        """, unsafe_allow_html=True)


def render_video_results(frame_results, annotated_frames, effective_fps):
    good_frames = sum(1 for r in frame_results if r[0])
    total       = len(frame_results)
    good_pct    = (good_frames / total * 100) if total else 0

    # Calculate hold time = consecutive good frames
    max_streak = cur_streak = 0
    for r in frame_results:
        if r[0]:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    good_hold_secs  = good_frames / effective_fps
    max_streak_secs = max_streak  / effective_fps

    # Body angles over time
    body_angles = [r[1] for r in frame_results if r[1] > 0]
    avg_angle   = np.mean(body_angles) if body_angles else 0

    # ── Metrics row ──
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{good_hold_secs:.1f}s</div>
            <div class="metric-label">Good-form time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{max_streak_secs:.1f}s</div>
            <div class="metric-label">Longest streak</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{good_pct:.0f}%</div>
            <div class="metric-label">Frames in form</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_angle:.0f}°</div>
            <div class="metric-label">Avg body angle</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Verdict
    if good_pct >= 80:
        st.markdown('<span class="badge badge-good">✓ EXCELLENT PLANK</span>', unsafe_allow_html=True)
    elif good_pct >= 50:
        st.markdown('<span class="badge badge-warn">~ NEEDS IMPROVEMENT</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-bad">✗ POOR FORM</span>', unsafe_allow_html=True)

    # Progress bar — good form %
    bar_color = "#00e5a0" if good_pct >= 80 else "#ffd166" if good_pct >= 50 else "#ff4060"
    st.markdown(f"""
    <div style="margin:0.8rem 0 0.3rem 0; font-size:0.75rem; color:#6a6a8a; text-transform:uppercase; letter-spacing:0.1em;">
        Good-form ratio
    </div>
    <div class="prog-wrap">
        <div class="prog-fill" style="width:{good_pct:.1f}%; background:{bar_color};"></div>
    </div>
    """, unsafe_allow_html=True)

    # Body angle chart using st.line_chart
    if body_angles:
        import pandas as pd
        t = np.arange(len(body_angles)) / effective_fps
        df = pd.DataFrame({"Body angle (°)": body_angles,
                           "Good min (160°)": [BODY_ANGLE_MIN]*len(body_angles),
                           "Good max (195°)": [BODY_ANGLE_MAX]*len(body_angles)},
                          index=t.round(1))
        st.markdown('<div class="section-title">Body angle over time</div>', unsafe_allow_html=True)
        st.line_chart(df, use_container_width=True, height=180)

    # Frame gallery — show a sample of annotated frames
    st.markdown('<div class="section-title">Frame samples</div>', unsafe_allow_html=True)
    n_show  = min(6, len(annotated_frames))
    indices = np.linspace(0, len(annotated_frames)-1, n_show, dtype=int)
    cols    = st.columns(n_show)
    for i, idx in enumerate(indices):
        is_g = frame_results[idx][0]
        label = "✓ Good" if is_g else "✗ Fix"
        with cols[i]:
            st.image(annotated_frames[idx], caption=label, use_container_width=True, channels="RGB")

    # Common issues
    all_issues = []
    for r in frame_results:
        all_issues.extend(r[3])
    if all_issues:
        from collections import Counter
        common = Counter(all_issues).most_common(3)
        st.markdown('<div class="section-title">Most common issues</div>', unsafe_allow_html=True)
        for issue, count in common:
            pct_frames = count / total * 100
            st.markdown(f'<div class="issue-item">⚠ {issue} &nbsp;·&nbsp; {pct_frames:.0f}% of frames</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    # ── Hero ──
    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
        <div class="hero-title">PLANK COACH</div>
        <div class="hero-sub">AI Form Checker · SMAI Assignment 3 · T6.5</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    try:
        detector = load_pose_model()
    except Exception as e:
        st.error(f"Could not load pose model: {e}\n\nMake sure you have internet access on first run to download the model (~3 MB).")
        st.stop()

    # ── Tabs ──
    tab_image, tab_video, tab_guide = st.tabs(["📸  Image Check", "🎬  Video Analysis", "📖  Form Guide"])

    # ── IMAGE TAB ──
    with tab_image:
        st.markdown("""
        <div class="tip-card">
            <span class="tip-icon">📸</span>
            Upload a photo of yourself (or anyone) in a plank position.
            Works best with a <strong>side-view</strong> so the full body line is visible.
        </div>
        """, unsafe_allow_html=True)

        uploaded_img = st.file_uploader(
            "Drop your plank photo here",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_upload"
        )

        if uploaded_img:
            pil_img = Image.open(uploaded_img)
            with st.spinner("Analysing pose…"):
                annotated, is_good, body_angle, head_angle, arm_angle, issues, side = process_image(pil_img, detector)

            if body_angle is None:
                st.warning("⚠ " + issues[0])
            else:
                render_image_results(is_good, body_angle, head_angle, arm_angle, issues, annotated)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 0; color:#3a3a5a;">
                <div style="font-size:3rem;">🏋️</div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:0.1em; margin-top:0.5rem;">
                    Upload an image to get started
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── VIDEO TAB ──
    with tab_video:
        st.markdown("""
        <div class="tip-card">
            <span class="tip-icon">🎬</span>
            Upload a short video (5–60 seconds). The app analyses every frame,
            measures how long you held <strong>good form</strong>, and shows
            a body-angle chart over time.
        </div>
        """, unsafe_allow_html=True)

        uploaded_vid = st.file_uploader(
            "Drop your plank video here",
            type=["mp4", "mov", "avi", "webm"],
            key="vid_upload"
        )

        if uploaded_vid:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            with st.spinner("Processing video — this may take 10–30 seconds…"):
                try:
                    frame_results, annotated_frames, eff_fps = process_video(tmp_path, detector)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Video processing error: {e}")
                    st.stop()

            if not frame_results:
                st.warning("No frames could be analysed. Make sure the video contains a visible person.")
            else:
                render_video_results(frame_results, annotated_frames, eff_fps)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 0; color:#3a3a5a;">
                <div style="font-size:3rem;">🎬</div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:0.1em; margin-top:0.5rem;">
                    Upload a video to analyse your hold time
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── GUIDE TAB ──
    with tab_guide:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="section-title">What the app measures</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="tip-card">
                <span class="tip-icon">🔵</span>
                <strong>Body line angle</strong> (Shoulder → Hip → Ankle)<br>
                The most important measurement. A perfect plank is a straight
                line — roughly 175°–180°. We allow 160°–195° as the acceptable
                range. Below 160° = hip sag. Above 195° = hip pike.
            </div>

            <div class="tip-card">
                <span class="tip-icon">🟡</span>
                <strong>Head angle</strong> (Ear → Shoulder → Hip)<br>
                Your neck should be a natural extension of your spine.
                If your head drops too low (angle &lt; 145°), you're straining
                your neck. Look at the floor, about 30 cm in front of your hands.
            </div>

            <div class="tip-card">
                <span class="tip-icon">⚪</span>
                <strong>Arm angle</strong> (Shoulder → Elbow → Wrist)<br>
                Reported for reference. For a high-plank: ~180° (straight arms).
                For a forearm plank: ~90°.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title">Tips for a better plank</div>', unsafe_allow_html=True)

            tips = [
                ("💪", "Squeeze your glutes and quads", "Actively contracting these muscles prevents hip sag without any conscious effort."),
                ("🫁", "Breathe through your diaphragm", "Holding your breath spikes blood pressure and shortens how long you can hold."),
                ("👀", "Gaze at the floor, not forward", "Looking forward strains the cervical spine. Keep neutral neck alignment."),
                ("🤚", "Hands under shoulders", "Stack wrists directly below shoulders to avoid wrist strain and keep your body line straight."),
                ("📱", "Film from the side", "A side-view lets the app see the full shoulder→hip→ankle line clearly."),
                ("🔄", "Build up gradually", "Target 30-second holds with perfect form before increasing duration."),
            ]
            for icon, title, body in tips:
                st.markdown(f"""
                <div class="tip-card">
                    <span class="tip-icon">{icon}</span>
                    <strong>{title}</strong><br>
                    <span style="color:#8a8aaa">{body}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">How the detection works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-card">
        This app uses <strong>Google MediaPipe Pose Landmarker</strong> — a lightweight neural network that
        detects 33 body keypoints in real time. No training data was needed; the model is pre-trained
        on millions of images. The plank-form rules (angle thresholds) are geometry-based calculations
        applied on top of those keypoints, inspired by physiotherapy guidelines for neutral spine alignment.
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1a1a26;
                text-align:center; color:#3a3a4a; font-size:0.75rem; letter-spacing:0.1em;">
        SMAI Assignment 3 · T6.5 · IIIT Hyderabad 2025-26 · Built with MediaPipe + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
