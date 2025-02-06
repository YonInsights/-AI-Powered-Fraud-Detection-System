import os
import sys
import pandas as pd
import numpy as np
import logging
from utils import load_data, save_data
from logging_utils import setup_logging

# Setup logging
logger = setup_logging()

def create_transaction_features(df):
    """Generates transaction frequency and time-based features."""
    logger.info("Generating transaction-based features...")

    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Compute transaction count per user correctly
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')

    # Fill NaN transaction counts with 1
    df['transaction_count'].fillna(1, inplace=True)

    # Compute time since signup
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

    return df

def create_time_features(df):
    """Extracts hour and day features from timestamps."""
    logger.info("Generating time-based features...")

    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_day'] = df['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6

    return df

def create_categorical_encoding(df):
    """Encodes categorical features into numerical values."""
    logger.info("Encoding categorical variables...")

    categorical_cols = ['source', 'browser', 'sex']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def normalize_features(df):
    """Scales numerical features for better model performance."""
    logger.info("Normalizing numerical values...")

    numerical_cols = ['purchase_value', 'transaction_count', 'time_since_signup']
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    return df

if __name__ == "__main__":
    # Load the cleaned dataset
    df = load_data("data/processed/cleaned_fraud_data.csv")

    # Generate new features
    df = create_transaction_features(df)
    df = create_time_features(df)
    df = create_categorical_encoding(df)
    df = normalize_features(df)

    # Save the final processed dataset
    save_data(df,r"D:\Kifya_training\Week 8\-AI-Powered-Fraud-Detection-System\data\processed/final_fraud_data.csv")
    logger.info("Feature engineering completed successfully!")
