import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_data, save_data
from logging_utils import setup_logging

# Setup logging
logger = setup_logging()

def split_data(df, target_column='class', test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets and removes non-numeric features."""
    logger.info("Splitting dataset into train and test sets...")

    # Drop datetime columns
    df = df.drop(columns=['signup_time', 'purchase_time'], errors='ignore')

    # Ensure only numeric columns are used for training
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type="logistic"):
    """Trains a model based on the specified type."""
    logger.info(f"Training {model_type} model...")

    if model_type == "logistic":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'logistic' or 'random_forest'.")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs performance metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Model Performance: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

if __name__ == "__main__":
    # Load the processed dataset
    df = load_data(r"D:\Kifya_training\Week 8\-AI-Powered-Fraud-Detection-System\data\processed\final_fraud_data.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train models
    logistic_model = train_model(X_train, y_train, model_type="logistic")
    rf_model = train_model(X_train, y_train, model_type="random_forest")

    # Evaluate models
    logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # Choose best model (based on F1-score)
    best_model = logistic_model if logistic_metrics["f1_score"] > rf_metrics["f1_score"] else rf_model
    best_model_name = "logistic_model.pkl" if logistic_metrics["f1_score"] > rf_metrics["f1_score"] else "random_forest_model.pkl"

    # Save best model
    joblib.dump(best_model, f"models/{best_model_name}")
    logger.info(f"Best model saved as {best_model_name}!")
