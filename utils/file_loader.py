import pandas as pd

def load_and_align(file_old, file_new, id_col):
    """
    Load old and new CSV files, and perform inner join on ID column.
    Returns a DataFrame with suffixes _old and _new for each column.
    """
    df_old = pd.read_csv(file_old)
    df_new = pd.read_csv(file_new)
    
    merged_df = pd.merge(df_old, df_new, on=id_col, suffixes=('_old', '_new'))
    return merged_df
