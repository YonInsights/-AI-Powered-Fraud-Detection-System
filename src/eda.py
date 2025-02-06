import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data
from logging_utils import setup_logging
from visualization import plot_distribution, plot_fraud_trends

# Setup logging
logger = setup_logging()

def analyze_missing_values(df):
    """Checks for missing values in the dataset."""
    missing_values = df.isnull().sum()
    logger.info(f"Missing values:\n{missing_values}")
    return missing_values

def analyze_fraud_distribution(df):
    """Computes fraud case distribution."""
    fraud_counts = df['class'].value_counts()
    logger.info(f"Fraud vs Non-Fraud distribution:\n{fraud_counts}")
    return fraud_counts

def analyze_time_based_trends(df):
    """Analyzes fraud cases over time."""
    df['purchase_hour'] = df['purchase_time'].dt.hour
    fraud_by_hour = df[df['class'] == 1]['purchase_hour'].value_counts().sort_index()
    
    logger.info("Fraud cases distribution by hour computed.")
    return fraud_by_hour

if __name__ == "__main__":
    # Load the cleaned dataset
    df = load_data("data/processed/cleaned_fraud_data.csv")

    # Perform EDA
    missing_values = analyze_missing_values(df)
    fraud_distribution = analyze_fraud_distribution(df)
    fraud_by_hour = analyze_time_based_trends(df)

    # Plot fraud trends
    plot_distribution(df, 'purchase_value', 'Distribution of Purchase Value')
    plot_fraud_trends(fraud_by_hour, 'Fraud Cases by Hour')
    
    logger.info("EDA completed successfully!")
