import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings for cleaner execution
warnings.filterwarnings('ignore')

# --- 1. BRUTALIST UI CONFIGURATION ---
st.set_page_config(page_title="CHURN_PREDICTOR", layout="wide")

st.markdown("""
    <style>
    /* High-contrast, brutalist aesthetic */
    .stApp { background-color: #ffffff; color: #000000; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { text-transform: uppercase; border-bottom: 4px solid #000; padding-bottom: 5px; font-weight: 900; letter-spacing: -1px; }
    .stButton>button { 
        background-color: #000000; color: #ffffff; border: 3px solid #000000; 
        font-weight: 900; text-transform: uppercase; border-radius: 0; 
        box-shadow: 6px 6px 0px #000000; transition: all 0.1s ease; width: 100%; height: 60px; font-size: 1.2rem;
    }
    .stButton>button:active { box-shadow: 0px 0px 0px #000000; transform: translate(6px, 6px); }
    div[data-testid="stSidebar"] { background-color: #f4f4f4; border-right: 4px solid #000; }
    .stProgress > div > div > div { background-color: #000000; }
    </style>
    """, unsafe_allow_html=True)


# --- 2. LOAD THE ML BRAIN ---
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_pipeline.pkl")


pipeline = load_model()

# --- 3. LAYOUT & INPUTS ---
st.title("SYSTEM // CHURN_PREDICTOR")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("INPUT_PARAMS")

    tenure = st.slider("TENURE (MONTHS)", 0, 72, 12)
    monthly_charges = st.number_input("MONTHLY CHARGES ($)", value=50.0)
    contract = st.selectbox("CONTRACT TYPE", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("INTERNET SERVICE", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("TECH SUPPORT", ["Yes", "No", "No internet service"])

    # Calculate engineered features dynamically based on user input
    est_ltv = tenure * monthly_charges
    eng_score = 1 if internet_service != "No" else 0 + (1 if tech_support == "Yes" else 0)

with col2:
    st.header("SYSTEM_OUTPUT")

    if st.button("EXECUTE_PREDICTION"):
        # Construct the exact dataframe structure the pipeline expects
        input_data = pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': est_ltv,
            'Contract': contract,
            'InternetService': internet_service,
            'TechSupport': tech_support,
            'Estimated_LTV': est_ltv,
            'Service_Engagement_Score': eng_score,
            # Hardcoded defaults for unselected fields to keep UI clean
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No',
            'OnlineBackup': 'No', 'DeviceProtection': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check'
        }])

        # Get probability of class 1 (Churn)
        proba = pipeline.predict_proba(input_data)[0][1]

        st.markdown(f"### CHURN_PROBABILITY: {proba * 100:.1f}%")
        st.progress(float(proba))

        # Binary Risk Classification
        if proba > 0.6:
            st.error("⚠️ HIGH RISK // IMMEDIATE INTERVENTION REQUIRED")
        elif proba > 0.4:
            st.warning("⚠️ MEDIUM RISK // MONITOR ACCOUNT")
        else:
            st.success("✔️ LOW RISK // STABLE ACCOUNT")

st.markdown("---")
st.markdown("**METRICS:** XGBOOST PIPELINE | OPTIMIZED FOR RECALL | MODEL V2.0")