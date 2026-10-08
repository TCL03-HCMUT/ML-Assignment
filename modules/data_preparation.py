"""Export row-aligned preprocessed text and labels for downstream models."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from modules.config import LABEL_NAMES, PreprocessingConfig
from modules.preprocessing import preprocess_dataset


def prepare_splits(splits, config=None, data_dir='data/processed', features_dir='features'):
    config = config or PreprocessingConfig()
    processed = {name: preprocess_dataset(df, config) for name, df in splits.items()}
    data_dir = Path(data_dir)
    features_dir = Path(features_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    impact = []
    for name, df in processed.items():
        df[['id', 'text', 'processed_text', 'tokens', 'labels']].to_json(
            data_dir / f'{name}.jsonl', orient='records', lines=True, force_ascii=False)
        y = df[LABEL_NAMES].to_numpy(dtype=np.uint8)
        np.save(features_dir / f'{name}_labels.npy', y, allow_pickle=False)
        impact.append({
            'split': name, 'rows': len(df),
            'mean_tokens': float(df.tokens.map(len).mean()),
            'empty_fallback_rows': int(df.processed_text.eq(config.empty_token).sum()),
        })
        assert len(df) == len(splits[name]) and df.id.equals(splits[name].id)
        assert np.array_equal(y, splits[name][LABEL_NAMES].to_numpy())
    metadata = {
        'label_names': LABEL_NAMES, 'config': asdict(config),
        'split_policy': 'official; repeated text retained and audited',
        'impact': impact,
        'artifacts_are': 'raw/processed text, tokens and multi-hot labels; model tokenization/padding handled downstream',
    }
    (features_dir / 'preprocessing_metadata.json').write_text(json.dumps(metadata, indent=2))
    return processed, pd.DataFrame(impact), metadata


def compare_preprocessing(train):
    """Data-impact comparison only; no model-quality claims without classifiers."""
    profiles={'emotion_preserving':PreprocessingConfig(),
              'stopwords_ablation':PreprocessingConfig(remove_stopwords=True),
              'punctuation_ablation':PreprocessingConfig(remove_punctuation=True),
              'embedding_minimal':PreprocessingConfig(lowercase=False,replace_urls=False,replace_users=False)}
    rows=[]
    for name,config in profiles.items():
        df=preprocess_dataset(train,config)
        rows.append({'profile':name,'mean_tokens':float(df.tokens.map(len).mean()),
            'vocabulary':len({t for tokens in df.tokens for t in tokens}),
            'changed_comments':int(df.processed_text.ne(df.text).sum()),
            'empty_fallback_rows':int(df.processed_text.eq(config.empty_token).sum())})
    return pd.DataFrame(rows)
