# 📉 Telecom B2B Churn: Predictive Retention System

**Live Application:** [View the Deployed System Here](https://churn-predictor-abil.streamlit.app)

![Light Mode UI](assets/light_mode.png)

## 🎯 Executive Summary
Customer churn is a silent revenue killer. Acquiring a new customer costs up to 5x more than retaining an existing one. This project delivers a production-ready Machine Learning pipeline that predicts customer churn probability with a focus on **high recall**, wrapped in a custom dual-theme SaaS dashboard for executive decision-making.

**Business Impact:** By aggressively tuning the model's `scale_pos_weight` to handle class imbalance, the system achieved a **0.80 Recall** for churning customers. In a corporate setting, false positives (giving a discount to a safe customer) cost significantly less than false negatives (losing the recurring revenue of a high-value customer entirely). The dashboard also features an **AI Prescriptive Action Engine** that instantly recommends retention strategies (like contract shifts or tech support upsells) based on dynamic risk factors.

## 🏗️ Architecture & Tech Stack
This is not just a Jupyter Notebook; it is a scalable, multi-tenant-ready architecture.
* **Backend ML Pipeline:** Python, Scikit-Learn (`Pipeline`, `ColumnTransformer`), XGBoost, Pandas.
* **Infrastructure:** Modular design, Config-driven execution (`config.yaml`), Joblib serialization.
* **Frontend:** Streamlit with custom CSS injection (Wireframe & Cyber-Slate themes), integrated with Plotly for dynamic data visualization.

## 🚀 Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/AbilAlgorithms/customer-churn-prediction.git](https://github.com/AbilAlgorithms/customer-churn-prediction.git)
   cd customer-churn-prediction
   
![Dark Mode UI](assets/dark_mode.png)