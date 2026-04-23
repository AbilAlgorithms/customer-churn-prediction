import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads raw data and handles initial structural issues."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # The 'TotalCharges' column comes in as text because of blank spaces.
    # We force it into decimal numbers here.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Any blank spaces become 'NaN' (Not a Number). We fill those with 0.0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # We drop the customer ID because it is useless for making predictions
    df.drop('customerID', axis=1, inplace=True, errors='ignore')

    print("Data loaded and cleaned successfully!")
    return df