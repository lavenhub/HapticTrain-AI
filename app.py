import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from datetime import date, datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HapticTrain AI",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    div.stButton > button { background-color: #00FF00; color: black; border-radius: 10px; font-weight: bold;}
    div[data-testid="stMetricValue"] { color: #00FF00; }
    .haptic-alert { background-color: #FF4B4B; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; color: white; animation: blinker 1s linear infinite;}
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle between three joints."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦾 HapticTrain AI")
    username = st.text_input("Athlete Name", "Alex Runner")
    st.success("System Connected")
    st.info("Hardware Simulation: ON")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📅 Smart Schedule (Excel)", "👁️ Haptic Coach", "📈 Multi-Metric Analytics"])

# ==========================================
# FEATURE 1: SMART SCHEDULE (EXCEL UPLOAD)
# ==========================================
with tab1:
    st.header("Smart Recovery Scheduler")
    st.write("Upload your training plan to analyze fatigue risks.")
    
    col_up, col_res = st.columns([1, 2])
    
    with col_up:
        uploaded_file = st.file_uploader("Upload Schedule (.xlsx or .csv)", type=['xlsx', 'csv'])
        
        # DEMO DATA GENERATOR (If user has no file)
        if st.button("Generate Demo Template"):
            demo_data = {
                "Date": [date.today() + timedelta(days=i) for i in range(10)],
                "Event_Type": ["Training", "Training", "Rest", "Training", "Competition", "Recovery", "Training", "Training", "Rest", "Training"],
                "Intensity": ["High", "High", "Low", "Medium", "MAX", "Low", "Medium", "High", "Low", "Medium"]
            }
            df_demo = pd.DataFrame(demo_data)
            df_demo.to_csv("demo_schedule.csv", index=False)
            st.success("Template Downloaded! Upload 'demo_schedule.csv' to test.")

    with col_res:
        if uploaded_file is not None:
            try:
                # Determine file type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Show Data
                st.dataframe(df.head(7), use_container_width=True)
                
                # LOGIC: Find next "Competition"
                # We assume the excel has a column 'Event_Type' or similar
                if 'Event_Type' in df.columns and 'Date' in df.columns:
                    # Convert date column to datetime
                    df['Date'] = pd.to_datetime(df['Date'])
                    today = pd.Timestamp.now()
                    
                    # Filter future competitions
                    future_comps = df[(df['Event_Type'].str.contains('Competition', case=False)) & (df['Date'] >= today)]
                    
                    if not future_comps.empty:
                        next_comp = future_comps.iloc[0]
                        days_left = (next_comp['Date'] - today).days
                        
                        st.metric("Days until Next Competition", f"{days_left} Days", delta_color="inverse")
                        
                        # LOGIC ENGINE
                        if days_left < 3:
                            st.error("🔴 CRITICAL PHASE: TAPER WEEK")
                            st.write("**AI Prescription:** Reduce volume by 50%. Focus on sleep and mobility.")
                        elif days_left < 7:
                            st.warning("🟡 PREP PHASE")
                            st.write("**AI Prescription:** Maintain intensity but drop duration.")
                        else:
                            st.success("🟢 BUILDING PHASE")
                            st.write("**AI Prescription:** High volume training authorized.")
                    else:
                        st.info("No upcoming competitions found in file.")
                else:
                    st.error("Excel must have columns: 'Date' and 'Event_Type'.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ==========================================
# FEATURE 2: HAPTIC COACH (UNCHANGED)
# ==========================================
with tab2:
    st.header("Real-Time Form Correction")
    run_camera = st.checkbox("Activate Camera & Haptic Loop")
    kpi1, kpi2 = st.columns(2)
    st_frame = st.empty()
    
    if run_camera:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and run_camera:
                ret, frame = cap.read()
                if not ret: break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Hip, Knee, Ankle
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    angle = calculate_angle(hip, knee, ankle)
                    
                    # HAPTIC LOGIC (Squat Depth)
                    if angle < 70:
                        cv2.putText(image, "BAD FORM - BUZZING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        kpi2.markdown('<div class="haptic-alert">⚠️ MOTOR ACTIVE</div>', unsafe_allow_html=True)
                    else:
                        kpi2.success("Motor Idle - Form Good")

                    kpi1.metric("Knee Angle", f"{int(angle)}°")
                    
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    st_frame.image(image, channels="BGR", use_container_width=True)
                except: pass
        cap.release()

# ==========================================
# FEATURE 3: MULTI-METRIC ANALYTICS
# ==========================================
with tab3:
    st.header("Performance Analytics & Mental Check")
    
    # Initialize separate histories
    if 'sprint_hist' not in st.session_state: st.session_state['sprint_hist'] = [12.2, 12.1, 12.0]
    if 'squat_hist' not in st.session_state: st.session_state['squat_hist'] = [45, 48, 50] # Reps per min
    if 'jump_hist' not in st.session_state: st.session_state['jump_hist'] = [60, 62, 61] # cm height

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Log New Data")
        metric_type = st.selectbox("Select Drill Type", ["100m Sprint", "Squats (Reps/Min)", "Vertical Jump (cm)"])
        
        # Dynamic Input based on selection
        if metric_type == "100m Sprint":
            val = st.number_input("Time (seconds)", 9.0, 20.0, 12.0)
            hist_key = 'sprint_hist'
            is_lower_better = True # Lower time is better
        elif metric_type == "Squats (Reps/Min)":
            val = st.number_input("Reps Count", 10, 100, 45)
            hist_key = 'squat_hist'
            is_lower_better = False # Higher reps is better
        else: # Jump
            val = st.number_input("Jump Height (cm)", 20, 120, 60)
            hist_key = 'jump_hist'
            is_lower_better = False # Higher height is better
            
        if st.button("Log Workout"):
            st.session_state[hist_key].append(val)
            
            # --- SHOWSTOPPER LOGIC (Generic for all metrics) ---
            history = st.session_state[hist_key]
            avg = sum(history[:-1]) / len(history[:-1])
            
            # Check for decline
            decline_detected = False
            if is_lower_better:
                if val > avg * 1.05: decline_detected = True # 5% Slower
            else:
                if val < avg * 0.95: decline_detected = True # 5% Weaker
            
            if decline_detected:
                st.error(f"⚠️ Performance Drop Detected! (Avg: {avg:.1f} vs Now: {val})")
                st.session_state['trigger_mental_check'] = True
            else:
                st.success("Performance Stable/Improving! Keep pushing.")
                st.session_state['trigger_mental_check'] = False

    with col2:
        st.subheader("Trend Analysis")
        # Display graph for currently selected metric
        chart_df = pd.DataFrame(st.session_state[hist_key], columns=[metric_type])
        st.line_chart(chart_df)

    # --- MENTAL CHECK MODAL ---
    if st.session_state.get('trigger_mental_check', False):
        st.markdown("---")
        with st.expander("🔻 DIAGNOSTIC REQUIRED: Open Mental Check", expanded=True):
            st.write("### Root Cause Analysis")
            stress = st.slider("Current Stress Level (1-10)", 1, 10, 5)
            sleep = st.slider("Hours of Sleep", 0, 12, 6)
            
            if st.button("Run AI Diagnosis"):
                if stress >= 7 or sleep < 5:
                    st.warning("🧠 DIAGNOSIS: CNS/Mental Burnout")
                    st.write("The drop in physical output is linked to high cognitive load.")
                    st.write("👉 **Protocol:** 20min Meditation + 8hr Sleep. NO Heavy training.")
                else:
                    st.warning("💪 DIAGNOSIS: Muscular Fatigue")
                    st.write("Mental state is clear. Muscles are simply overworked.")
                    st.write(f"👉 **Protocol:** Targeted Massage for {metric_type} muscle groups + Protein intake.")