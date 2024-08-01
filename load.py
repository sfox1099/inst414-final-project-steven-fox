import pandas as pd

def load(data, file_path):
    """
    Loads data into a CSV file.

    Args:
        data (pd.DataFrame): Data to be saved.
        file_path (str): Path to the CSV file.

    Returns:
        None
    """
    data.to_csv(file_path, index=False)
    print(f"Data successfully loaded to {file_path}")