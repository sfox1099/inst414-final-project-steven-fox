import pandas as pd


def extract(file_path):
    """
    Extracts data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Data extracted from the CSV file.
    """
    data = pd.read_csv(file_path) 
    return data