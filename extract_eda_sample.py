import sys, os, pandas as pd, numpy as np
sys.path.insert(0, 'src')
from features import extract_features_batch, FEATURE_NAMES

df = pd.read_csv('data/processed/known/train.csv')
parts = [g.sample(min(200, len(g)), random_state=42) for _, g in df.groupby('class_label')]
sample = pd.concat(parts).reset_index(drop=True)
print(f'Sample: {len(sample)} rows, {sample["class_label"].nunique()} classes')

feats = extract_features_batch(sample['domain'].tolist())
feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
feat_df.insert(0, 'domain', sample['domain'].values)
feat_df.insert(1, 'class_label', sample['class_label'].values)

os.makedirs('data/eda', exist_ok=True)
feat_df.to_csv('data/eda/sample_features.csv', index=False)
sz = os.path.getsize('data/eda/sample_features.csv') / 1024
print(f'Saved data/eda/sample_features.csv  ({sz:.0f} KB, shape={feat_df.shape})')
