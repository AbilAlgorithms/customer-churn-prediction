import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


def build_and_train_pipeline(df: pd.DataFrame, config: dict):
    """Builds, trains, evaluates, and saves the ML pipeline."""

    # Separate the target (Churn) from the features
    X = df.drop(config['data']['target'], axis=1)
    y = df[config['data']['target']]

    # Split into training data (80%) and testing data (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'],
        random_state=config['model']['random_state'], stratify=y
    )

    # Sort columns by data type so we can process them correctly
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessor rules: Standardize numbers, One-Hot Encode text categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Set up the XGBoost model using the settings from our config.yaml
    xgb_params = config['model']['xgb_params']
    classifier = XGBClassifier(**xgb_params, random_state=42)

    # Tie it all together into one automated pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    print("\nTraining XGBoost Pipeline (Optimized for Recall)...")
    pipeline.fit(X_train, y_train)

    # Test the model to see how well it learned
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Save the trained brain so our web app can use it later
    joblib.dump(pipeline, config['model']['save_path'])
    print(f"\nPipeline successfully saved to {config['model']['save_path']}")