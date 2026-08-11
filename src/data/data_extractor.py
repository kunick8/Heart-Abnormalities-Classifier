import pandas as pd

def extract_data(pathname) -> pd.DataFrame:
    return pd.read_csv(pathname)
