import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, column, title):
    """Plots distribution of a numerical column."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

def plot_fraud_trends(fraud_by_hour, title):
    """Plots fraud cases over different time periods."""
    plt.figure(figsize=(8, 5))
    fraud_by_hour.plot(kind='bar', color='red')
    plt.title(title)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Fraud Cases")
    plt.show()
