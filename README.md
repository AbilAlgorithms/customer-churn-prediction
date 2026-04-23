# 📉 Telco Customer Churn: Predictive Retention System

## 🎯 Executive Summary
Customer churn is a silent revenue killer. Acquiring a new customer costs up to 5x more than retaining an existing one. This project delivers a production-ready Machine Learning pipeline that predicts customer churn probability with a focus on **high recall**, wrapped in a brutalist, high-contrast UI for executive decision-making.

**Business Impact:** By aggressively tuning the model's `scale_pos_weight` to handle class imbalance, the system achieved a **0.80 Recall** for churning customers. In a corporate setting, false positives (giving a discount to a safe customer) cost significantly less than false negatives (losing the recurring revenue of a high-value customer entirely). This model protects the bottom line.

## 🏗️ Architecture & Tech Stack
This is not just a Jupyter Notebook; it is a scalable, multi-tenant-ready architecture.
* **Backend ML Pipeline:** Python, Scikit-Learn (`Pipeline`, `ColumnTransformer`), XGBoost, Pandas.
* **Infrastructure:** Modular design, Config-driven execution (`config.yaml`), Joblib serialization.
* **Frontend:** Streamlit with custom CSS injection for a minimalist, brutalist aesthetic.

## 🚀 Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/churn-prediction-system.git](https://github.com/yourusername/churn-prediction-system.git)
   cd churn-prediction-system