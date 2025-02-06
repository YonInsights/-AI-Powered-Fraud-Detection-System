import os
import sys
import pandas as pd
import numpy as np
import logging
from utils import load_data, save_data
from logging_utils import setup_logging

# Setup logging
logger = setup_logging()

def clean_data(df):
    """Handles missing values and duplicates."""
    logger.info("Cleaning data: Handling missing values and duplicates...")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

def convert_datetime(df, columns):
    """Converts specified columns to datetime format, handling errors."""
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def merge_ip_data(transaction_df, ip_df):
    """Merges fraud transaction data with IP geolocation data."""
    logger.info("Merging IP address data...")
    transaction_df['ip_address'] = transaction_df['ip_address'].astype(int)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

    # Merge using IP range
    merged_df = transaction_df.merge(
        ip_df, how="left",
        left_on="ip_address",
        right_on="lower_bound_ip_address"
    )
    return merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])

if __name__ == "__main__":
    # Load datasets
    fraud_data = load_data("data/raw/Fraud_Data.csv")
    ip_data = load_data("data/raw/IpAddress_to_Country.csv")

    # Clean fraud data
    fraud_data = clean_data(fraud_data)
    fraud_data = convert_datetime(fraud_data, ['signup_time', 'purchase_time'])

    # Merge with IP geolocation data
    processed_data = merge_ip_data(fraud_data, ip_data)

    # Save cleaned data
    save_data(processed_data, "data/processed/cleaned_fraud_data.csv")
    logger.info("Data preprocessing complete!")
