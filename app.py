import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="HeartAssistX",
    page_icon="❤️",
    layout="centered"
)

def load_assets():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        info = joblib.load('model_info.pkl')
        return model, scaler, info
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def apply_custom_css(theme):
    if theme == "Dark":
        bg_color = "#121212"
        text_color = "#e0e0e0"
        card_bg = "#1e1e1e"
        accent_red = "#ff4d4d"
        secondary_red = "#8b0000"
        border_color = "#333333"
    else:
        bg_color = "#ffffff"
        text_color = "#333333"
        card_bg = "#fcfcfc"
        accent_red = "#d90429"
        secondary_red = "#ef233c"
        border_color = "#eeeeee"

    st.markdown(f"""
        <style>
        /* Modern iOS-like font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: {bg_color};
            color: {text_color};
        }}

        .stApp {{
            background-color: {bg_color};
        }}
        
        /* Form & Input containers */
        div[data-testid="stForm"] {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}

        /* Buttons */
        .stButton>button {{
            width: 100%;
            border-radius: 12px;
            height: 3.5em;
            background-color: {accent_red};
            color: white !important;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {secondary_red};
            box-shadow: 0 4px 15px {accent_red}44;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            color: {accent_red} !important;
            font-weight: 700;
        }}

        /* Assessment Boxes */
        .risk-card {{
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            border-left: 6px solid;
        }}
        .high-risk {{
            background-color: {accent_red}15;
            border-color: {accent_red};
            color: {accent_red};
        }}
        .low-risk {{
            background-color: #2b934815;
            border-color: #2b9348;
            color: #2b9348;
        }}

        /* Typography */
        h1 {{ font-weight: 700; color: {text_color} !important; }}
        h3 {{ font-weight: 600; color: {text_color} !important; }}
        label {{ color: {text_color} !important; font-weight: 500 !important; }}

        /* Shorten slider via CSS if possible, but columns are more robust */
        div[data-testid="stSelectSlider"] {{
            max-width: 120px; /* Force a shorter width */
        }}
        </style>
    """, unsafe_allow_html=True)

def main():
    model, scaler, info = load_assets()
    if not model:
        return

    # Sidebar - Reverted with Shortened Slider
    with st.sidebar:
        st.image("https://img.icons8.com/plasticine/100/000000/heart-with-pulse.png", width=80)
        st.title("Settings")
        st.write("---")
        
        # Shortening the toggle slider
        st.write("🌓 **Theme**")
        # Use columns to center/shorten the slider
        s_col1, s_col2 = st.columns([1, 1.2]) 
        with s_col1:
            theme = st.select_slider("", options=["Light", "Dark"], value="Light", label_visibility="collapsed")
        
        st.write("---")
        st.markdown(f"**Model:** {info['model_name']}")
        st.markdown(f"**Test Performance:** `{info['test_accuracy']:.2%}`")
        st.caption("Score based on independent 20% test data split.")

    # Apply Aesthetics
    apply_custom_css(theme)

    # Main UI
    st.title("HeartAssistX")
    st.markdown(f"##### Precise Patient Risk Assessment using Advanced Machine Learning")
    
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("clinical_form"):
        st.markdown("### 📋 Clinical Indicators")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 1, 115, 50)
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", 50, 220, 120)
            chol = st.number_input("Cholesterol", 100, 550, 220)
            fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1])
            
        with col2:
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
            oldpeak = st.number_input("ST Depression", 0.0, 8.0, 1.0, step=0.1)
            slope = st.selectbox("ST Slope", options=[0, 1, 2])
            ca = st.selectbox("Major Vessels Count", options=[0, 1, 2, 3, 4])
            thal = st.selectbox("Thal Status", options=[1, 2, 3])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("GENERATE RISK PROFILE")

    if submitted:
        # Data Preparation
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ]], columns=info['feature_names'])

        # Scaling
        input_processed = scaler.transform(input_data) if info['use_scaler'] else input_data

        # Prediction
        prob = model.predict_proba(input_processed)[0][1]
        prediction = 1 if prob >= 0.5 else 0
        confidence = prob if prediction == 1 else (1 - prob)

        # UI Response
        st.divider()
        st.subheader("Diagnostic Output")
        
        m1, m2 = st.columns(2)
        m1.metric("Risk Probability", f"{prob:.1%}")
        m2.metric("Intelligence Confidence", f"{confidence:.1%}")

        if prediction == 1:
            st.markdown(f"""
                <div class="risk-card high-risk">
                    <h3>⚠️ HIGH RISK ALERT</h3>
                    <p>Clinical data indicates high probability of heart disease. Immediate cardiological review is recommended.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="risk-card low-risk">
                    <h3>✅ LOW RISK OBSERVED</h3>
                    <p>Current patient metrics align with low cardiovascular risk. Continue preventative care.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
