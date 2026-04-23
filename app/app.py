import streamlit as st
import pandas as pd
import joblib
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- 1. NUVIA-STYLE UI CONFIGURATION ---
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.markdown("""
    <style>
    /* FIX 1: Hide the default Streamlit top black bar */
    header { visibility: hidden !important; }

    /* Soft global background */
    .stApp { background-color: #F8F9FA; color: #1F2937; font-family: 'Inter', sans-serif; }

    /* Clean headers */
    h1 { font-weight: 600; color: #111827; margin-bottom: 0px; }

    /* Input Container Styling */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background: #FFFFFF; border-radius: 16px; padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Custom Gradient HTML Cards */
    .metric-card {
        padding: 20px; border-radius: 16px; color: white; box-shadow: 0 10px 20px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    .card-green { background: linear-gradient(135deg, #84CC16 0%, #4D7C0F 100%); }
    .card-pink { background: linear-gradient(135deg, #EC4899 0%, #9D174D 100%); }
    .card-blue { background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); }
    .metric-title { font-size: 14px; opacity: 0.9; margin-bottom: 5px; font-weight: 500;}
    .metric-value { font-size: 28px; font-weight: 700; margin: 0; }

    /* Primary Button */
    .stButton>button { 
        background-color: #111827; color: white; border-radius: 8px; border: none;
        padding: 10px; font-weight: 600; width: 100%;
    }
    .stButton>button:hover { background-color: #374151; color: white; }

    /* FIX 2: Force all alert text to be dark and readable */
    div[data-testid="stAlert"] * {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. LOAD THE ML BRAIN ---
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_pipeline.pkl")


pipeline = load_model()

# --- 3. DASHBOARD LAYOUT ---
# FIX 3: Updated wording for B2B Telecom
st.title("Telecom B2B Churn Predictor")
st.markdown("Predictive Account Risk & Retention Analysis")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.8])

with col1:
    st.markdown("### Client Parameters")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", value=50.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    est_ltv = tenure * monthly_charges
    eng_score = 1 if internet_service != "No" else 0 + (1 if tech_support == "Yes" else 0)

with col2:
    card1, card2, card3 = st.columns(3)
    card1.markdown(
        f'<div class="metric-card card-green"><div class="metric-title">Expected LTV</div><div class="metric-value">${est_ltv:,.0f}</div></div>',
        unsafe_allow_html=True)
    card2.markdown(
        f'<div class="metric-card card-pink"><div class="metric-title">Monthly Revenue</div><div class="metric-value">${monthly_charges:,.2f}</div></div>',
        unsafe_allow_html=True)
    card3.markdown(
        f'<div class="metric-card card-blue"><div class="metric-title">Engagement Score</div><div class="metric-value">{eng_score} / 2</div></div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Generate Risk Report"):
        input_data = pd.DataFrame([{
            'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': est_ltv,
            'Contract': contract, 'InternetService': internet_service, 'TechSupport': tech_support,
            'Estimated_LTV': est_ltv, 'Service_Engagement_Score': eng_score,
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'PhoneService': 'Yes', 'MultipleLines': 'No', 'OnlineSecurity': 'No',
            'OnlineBackup': 'No', 'DeviceProtection': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check'
        }])

        proba = pipeline.predict_proba(input_data)[0][1] * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            title={'text': "Flight Risk Probability", 'font': {'size': 18, 'color': '#111827'}},
            number={'suffix': "%", 'font': {'size': 40, 'color': '#111827', 'weight': 'bold'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#111827"},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': "#D1FAE5"},
                    {'range': [40, 60], 'color': "#FEF3C7"},
                    {'range': [60, 100], 'color': "#FEE2E2"}
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if proba > 60:
            st.error("🚨 High risk of churn. Immediate intervention required.")
        elif proba > 40:
            st.warning("⚠️ Elevated risk detected. Monitor account health metrics closely.")
        else:
            st.success("✅ Expected retention rate is high. Account is stable.")