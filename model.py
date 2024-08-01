import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def prepare_data(data):
    """
    Prepares the data for training by splitting it into features and target.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y


def train_model(X, y):
    """
    Trains a RandomForest model on the provided data.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.

    Returns:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_test, y_test