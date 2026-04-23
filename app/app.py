import streamlit as st
import pandas as pd
import joblib
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- 1. UI CONFIGURATION & THEME TOGGLE ---
st.set_page_config(page_title="Churn Prediction", layout="wide")

theme_choice = st.radio("UI THEME", ["Wireframe (Light)", "Cyber-Slate (Neon Dark)"], horizontal=True)

if theme_choice == "Wireframe (Light)":
    st.markdown("""
        <style>
        header { visibility: hidden !important; }
        .stApp { 
            background-color: #FAFAFA; 
            background-image: radial-gradient(#CBD5E1 1px, transparent 1px);
            background-size: 20px 20px;
            color: #1F2937; 
            font-family: 'Inter', sans-serif; 
        }
        h1, h2, h3, p, label, span { color: #111827 !important; }
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
            background: #FFFFFF; border-radius: 12px; padding: 20px;
            box-shadow: 4px 4px 0px rgba(0,0,0,0.1); border: 2px solid #E5E7EB;
        }
        .metric-card { padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px; border: 2px solid #000; box-shadow: 4px 4px 0px #000;}
        .card-green { background: #84CC16; color: #000; }
        .card-pink { background: #EC4899; color: #000; }
        .card-blue { background: #3B82F6; color: #000; }
        .metric-title { font-size: 14px; font-weight: 700; text-transform: uppercase; color: #000 !important; }
        .metric-value { font-size: 28px; font-weight: 900; margin: 0; color: #000 !important; }

        /* FIX: Lock the button text to white, preventing global text color override */
        .stButton>button { background-color: #111827 !important; color: #FFFFFF !important; border-radius: 8px; border: none; padding: 12px; font-weight: 800; text-transform: uppercase; width: 100%; box-shadow: 4px 4px 0px #000; transition: 0.1s;}
        .stButton>button * { color: #FFFFFF !important; }
        .stButton>button:hover { background-color: #374151 !important; color: #FFFFFF !important; }
        .stButton>button:hover * { color: #FFFFFF !important; }
        .stButton>button:active { box-shadow: 0px 0px 0px #000 !important; transform: translate(4px, 4px) !important; }

        .action-plan { background-color: #F8FAFC; border-left: 5px solid #111827; padding: 15px; margin-top: 20px; font-family: 'Courier New', Courier, monospace; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)
    gauge_font_color = "#000000"
    gauge_bg_color = "white"
    gauge_bar_color = "#111827"

else:
    st.markdown("""
        <style>
        header { visibility: hidden !important; }
        .stApp { 
            background-color: #0F172A; color: #F8FAFC; font-family: 'Inter', sans-serif; 
        }
        h1, h2, h3, p, label, span { color: #F8FAFC !important; }
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
            background: #1E293B; border-radius: 12px; padding: 20px;
            border: 1px solid #334155; box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
        }
        .metric-card { 
            padding: 20px; border-radius: 12px; margin-bottom: 20px; 
            background: #1E293B; border: 1px solid rgba(57, 255, 20, 0.3); 
            box-shadow: 0px 4px 15px rgba(57, 255, 20, 0.05);
        }
        .card-green, .card-pink, .card-blue { background: #1E293B; }
        .metric-title { font-size: 14px; font-weight: 600; color: #94A3B8 !important; text-transform: uppercase; }
        .metric-value { font-size: 28px; font-weight: 900; margin: 0; color: #39FF14 !important; text-shadow: 0px 0px 10px rgba(57, 255, 20, 0.4);}

        /* Neon Accent Button and text lock */
        .stButton>button { 
            background-color: #1E293B !important; color: #39FF14 !important; 
            border: 2px solid #39FF14 !important; border-radius: 8px; padding: 12px; 
            font-weight: 800; width: 100%; transition: 0.2s; text-transform: uppercase;
        }
        .stButton>button * { color: #39FF14 !important; }
        .stButton>button:hover { background-color: #39FF14 !important; color: #0F172A !important; box-shadow: 0px 0px 15px rgba(57,255,20,0.4) !important; }
        .stButton>button:hover * { color: #0F172A !important; }

        /* Glowing Red Slider Thumb in Dark Mode */
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #FF3366 !important;
            box-shadow: 0px 0px 12px rgba(255, 51, 102, 0.9) !important;
            border: 2px solid #FFF !important;
        }

        div[data-baseweb="select"] > div, input { border: 1px solid #334155 !important; background-color: #0F172A !important; color: #F8FAFC !important;}
        div[data-testid="stAlert"] { background-color: #1E293B !important; border: 1px solid #39FF14 !important; }
        div[data-testid="stAlert"] * { color: #39FF14 !important; font-weight: 800 !important; }

        .action-plan { background-color: #0F172A; border-left: 5px solid #39FF14; padding: 15px; margin-top: 20px; color: #39FF14; font-family: 'Courier New', Courier, monospace; font-weight: bold; border: 1px solid #334155;}
        </style>
    """, unsafe_allow_html=True)
    gauge_font_color = "#F8FAFC"
    gauge_bg_color = "#0F172A"
    gauge_bar_color = "#39FF14"


# --- 2. LOAD THE ML BRAIN ---
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_pipeline.pkl")


pipeline = load_model()

# --- 3. DASHBOARD LAYOUT ---
st.title("Telecom B2B Churn Predictor")
st.markdown("Predictive Account Risk & Retention Analysis")
st.markdown("---")

col1, col2 = st.columns([1, 1.8])

# LEFT COLUMN: Inputs and Action Button
with col1:
    st.markdown("### Client Parameters")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", value=50.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    est_ltv = tenure * monthly_charges
    eng_score = 1 if internet_service != "No" else 0 + (1 if tech_support == "Yes" else 0)

    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.button("GENERATE RISK REPORT")

# RIGHT COLUMN: Metrics and Visuals
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

    if run_prediction:
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

        # Dynamic Gauge Chart with darkened ticks
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            title={'text': "Flight Risk Probability", 'font': {'size': 18, 'color': gauge_font_color}},
            number={'suffix': "%", 'font': {'size': 40, 'color': gauge_font_color, 'weight': 'bold'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': gauge_font_color,
                         'tickfont': {'color': gauge_font_color, 'size': 14, 'weight': 'bold'}},
                'bar': {'color': gauge_bar_color},
                'bgcolor': gauge_bg_color,
                'borderwidth': 1,
                'bordercolor': gauge_font_color,
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0,0,0,0)"},
                    {'range': [40, 60], 'color': "rgba(0,0,0,0)"},
                    {'range': [60, 100], 'color': "rgba(0,0,0,0)"}
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if proba > 60:
            st.error("🚨 HIGH RISK OF CHURN. IMMEDIATE INTERVENTION REQUIRED.")
            action = f"**AI Prescriptive Action:** Account is highly unstable. Immediately offer a 15% discount to transition from {contract} to a One-year contract and throw in free Tech Support to increase ecosystem lock-in."
        elif proba > 40:
            st.warning("⚠️ ELEVATED RISK DETECTED. MONITOR ACCOUNT METRICS CLOSELY.")
            action = f"**AI Prescriptive Action:** Engagement score is {eng_score}/2. Schedule a customer success call to upsell Tech Support or Fiber optic services to deepen account dependency."
        else:
            st.success("✅ EXPECTED RETENTION RATE IS HIGH. ACCOUNT IS STABLE.")
            action = "**AI Prescriptive Action:** Account is healthy. Eligible for standard automated marketing and cross-selling campaigns."

        st.markdown(f'<div class="action-plan">{action}</div>', unsafe_allow_html=True)