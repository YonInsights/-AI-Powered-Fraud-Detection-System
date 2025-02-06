import pandas as pd

def load_data(filepath):
    """Loads a CSV file into a Pandas DataFrame."""
    return pd.read_csv(filepath)

def save_data(df, filepath):
    """Saves a Pandas DataFrame to CSV format."""
    df.to_csv(filepath, index=False)
