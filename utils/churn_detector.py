def detect_churn_rows(df, threshold=0.9):
    """
    Detect rows where the overall similarity is below threshold.
    Returns the indices of churned rows.
    """
    churn_indices = df[df['overall_similarity'] < threshold].index.tolist()
    return churn_indices
