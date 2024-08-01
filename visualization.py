import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def plot_feature_distributions(data):
    """
    Plots distributions of features.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        None
    """
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()

def plot_correlation_matrix(data):
    """
    Plots the correlation matrix of the data.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        None
    """
    corr = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_model_performance(conf_matrix):
    """
    Plots the confusion matrix of the model performance.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()