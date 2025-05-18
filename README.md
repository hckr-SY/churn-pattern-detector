# AI Churn Detector

This tool compares two datasets and detects row-level changes (churn) using semantic similarity.

## Features
- Sentence Transformer-based embeddings
- Cosine similarity comparison
- Optional address normalization with Libpostal

## Setup
```bash
pip install -r requirements.txt
```

## Run
Place your two datasets in `data/` as `dataset_old.csv` and `dataset_new.csv`, then run:
```bash
python main.py
```

Output will be saved to `output/churn_results.csv`.

## Config
Modify settings in `config/settings.yaml` to change model, device, or data columns.