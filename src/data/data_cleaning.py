import pandas as pd

def join_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame.join(df1, df2)
