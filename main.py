import yaml
from src.data_loader import load_and_clean_data
from src.feature_engine import engineer_features
from src.model_pipeline import build_and_train_pipeline

if __name__ == "__main__":
    # 1. Load the configuration settings
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    print("\n=== CHURN PREDICTION PIPELINE STARTING ===")

    # 2. Ingest Data
    raw_df = load_and_clean_data(config['data']['raw_path'])

    # 3. Engineer Features
    processed_df = engineer_features(raw_df)

    # Save a copy of the cleaned data (good practice)
    processed_df.to_csv(config['data']['processed_path'], index=False)
    print(f"Processed data saved to {config['data']['processed_path']}")

    # 4. Train, Evaluate, and Save the Model
    build_and_train_pipeline(processed_df, config)

    print("=== PIPELINE EXECUTION COMPLETE ===\n")