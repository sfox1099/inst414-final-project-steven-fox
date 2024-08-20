import pandas as pd
import os

def transform(data):
    """
    Transforms the data by performing assertion tests to ensure data cleanliness.

    Args:
        data (pd.DataFrame): Input data to be transformed.

    Returns:
        pd.DataFrame: The transformed data after assertion checks.
    """
    assert not data.isnull().values.any()
    assert data['UDI'].dtype == 'int64'
    assert data['Product ID'].dtype == 'object'
    assert data['Type'].dtype == 'object'
    assert data['Rotational speed [rpm]'].dtype == 'int64'
    assert data['Torque [Nm]'].dtype == 'float64'
    assert data['Tool wear [min]'].dtype == 'int64'
    assert data['Target'].dtype == 'int64'
    
    return data