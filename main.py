import os
import pandas as pd
import yaml
from models.embedder import Embedder
from models.similarity_engine import compute_column_similarities
from utils.file_loader import load_and_align
from utils.churn_detector import detect_churn_rows
from utils.llm_insights import generate_insights_with_ollama

# Load configuration
with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

id_col = config['id_column']
threshold = config['threshold']
column_weights = config['column_weights']
similarity_methods = config['similarity_methods']
model_name = config['model_name']
device = config['device']

# Load and align datasets
merged_df = load_and_align("data/dataset_old.csv", "data/dataset_new.csv", id_col)

# Compute column-wise similarities
similarity_df = compute_column_similarities(merged_df, config)

# Merge similarity scores into dataframe
merged_df = pd.concat([merged_df, similarity_df], axis=1)

# Compute overall weighted similarity score per row
weights_series = pd.Series(column_weights)
merged_df['overall_similarity'] = similarity_df.mul(weights_series).sum(axis=1)

# Detect churn rows based on threshold
merged_df['churn_flag'] = merged_df['overall_similarity'] < threshold

# Save output files
os.makedirs("output", exist_ok=True)
merged_df.to_csv("output/_merged_score.csv", index=False)
merged_df[merged_df['churn_flag']].to_csv("output/churn_results.csv", index=False)

# Generate AI-based insights on churned rows using Ollama
churned_df = merged_df[merged_df['churn_flag']]
insights = generate_insights_with_ollama(churned_df, similarity_methods)

with open("output/summary.txt", "w") as f:
    f.write(insights)

print("✅ Churn detection and insight generation complete.")
