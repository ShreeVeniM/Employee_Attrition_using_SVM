import pandas as pd


def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None