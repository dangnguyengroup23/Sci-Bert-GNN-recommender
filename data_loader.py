import pandas as pd
import ast
import random
import logging
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []

def generate_synthetic_data(max_samples):
    logger.warning("Generating synthetic data (synthetic mode).")
    cats = [f"cs.CV" if i % 5 == 0 else f"cat.{i%20}" for i in range(50)]
    authors = [f"Author_{j}" for j in range(200)]
    data = {
        'id': [str(i) for i in range(max_samples)],
        'title': [f"Research on topic {i} in deep learning and vision" for i in range(max_samples)],
        'abstract': [f"This work studies model {i}, presenting experiments and results that illustrate patterns. More descriptive text to give SciBERT something to learn." for i in range(max_samples)],
        'authors_parsed': [[(random.choice(authors), "", "")] for _ in range(max_samples)],
        'categories': [random.choice(cats) for _ in range(max_samples)],
    }
    return pd.DataFrame(data)

def load_data(max_samples=12000):
    try:
        path = "data/arxiv-metadata-oai-snapshot.json"
        df = pd.read_json(path, lines=True, nrows=max_samples)
        logger.info(f"Loaded {len(df)} rows from arXiv.")
    except Exception as e:
        logger.warning(f"Failed to load real data: {e}. Using synthetic data.")
        df = generate_synthetic_data(max_samples)

    df.fillna({'abstract': "No abstract", 'title': "No title"}, inplace=True)
    df['id'] = df['id'].astype(str)
    df['text'] = df['title'] + "\n\n" + df['abstract']
    df['authors_parsed'] = df.get('authors_parsed', pd.Series([[]]*len(df))).apply(safe_literal_eval)
    df['authors_set'] = df['authors_parsed'].apply(lambda x: {a[0] for a in x if a})
    df['categories'] = df['categories'].apply(lambda x: x.split() if isinstance(x, str) else (x if isinstance(x, list) else []))
    df['label'] = df['categories'].apply(lambda x: x[0] if x else 'unknown')

    keep = df['label'].value_counts()[lambda x: x >= 5].index
    df = df[df['label'].isin(keep)].reset_index(drop=True)

    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label']).astype(int)
    return df, le