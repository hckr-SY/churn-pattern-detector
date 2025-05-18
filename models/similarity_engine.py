import pandas as pd

from models.embedder import Embedder
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_score(texts_old, texts_new, embedder):
    emb_old = embedder.encode(texts_old)
    emb_new = embedder.encode(texts_new)
    emb_old = emb_old.cpu().numpy()
    emb_new = emb_new.cpu().numpy()
    scores = [cosine_similarity([o], [n])[0][0] for o, n in zip(emb_old, emb_new)]
    return scores

def compute_column_similarities(merged_df, config):
    """
    For each relevant column pair (old and new), compute similarity scores.
    Returns a DataFrame where each column is the similarity score of one column.
    """
    id_col = config['id_column']
    columns = config['columns_to_compare']  # List of column base names e.g. ['address', 'city', 'pincode']
    model_name = config['model_name']
    device = config['device']

    embedder = Embedder(model_name=model_name, device=device)
    similarity_data = {}

    for col in columns:
        col_old = f"{col}_old"
        col_new = f"{col}_new"
        texts_old = merged_df[col_old].astype(str).tolist()
        texts_new = merged_df[col_new].astype(str).tolist()
        scores = compute_similarity_score(texts_old, texts_new, embedder)
        similarity_data[f"{col}_similarity"] = scores

    return pd.DataFrame(similarity_data)
