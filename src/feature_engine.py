import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates smart business features to improve model signal."""
    print("Engineering advanced features...")
    df_engineered = df.copy()

    # Feature 1: Estimated Lifetime Value (LTV)
    # How much money has this customer given us over their total tenure?
    df_engineered['Estimated_LTV'] = df_engineered['tenure'] * df_engineered['MonthlyCharges']

    # Feature 2: Service Engagement Score
    # The more services a customer uses, the harder it is for them to leave (ecosystem lock-in).
    services = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

    # Count how many of these services the user has active
    df_engineered['Service_Engagement_Score'] = df_engineered[services].apply(
        lambda x: sum(x.astype(str).str.contains('Yes')), axis=1
    )

    # Target Encoding: Change 'Yes'/'No' churn into 1s and 0s for the math model
    df_engineered['Churn'] = df_engineered['Churn'].map({'Yes': 1, 'No': 0})

    print("Feature engineering complete!")
    return df_engineered