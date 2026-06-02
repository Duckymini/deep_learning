"""
Shared training and evaluation utilities for the hate speech RAC pipeline.

Functions here are used across multiple training and evaluation notebooks.
Import via:
    from training_utils import compute_metrics, tokenize_augmented, ...
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    """
    HuggingFace Trainer callback — macro F1, Precision, Recall.
    Accepts an EvalPrediction object (attribute access).
    """
    preds  = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids
    return {
        'macro_f1': f1_score(labels, preds, average='macro',  zero_division=0),
        'macro_p':  precision_score(labels, preds, average='macro', zero_division=0),
        'macro_r':  recall_score(labels, preds, average='macro',    zero_division=0),
    }


def tokenize_augmented(records, tokenizer, max_length=256, query_key='query'):
    """
    Build augmented input strings and tokenize for the RAC classifier.

    Format: query [SEP] neighbor1 [SEP] neighbor2 ...

    Parameters
    ----------
    records : list[dict]
        Each dict must have keys: query_key (str), 'neighbors' (list[str]), 'label' (int).
    query_key : str
        Key for the query text in each record. Default 'query'.
    """
    sep = tokenizer.sep_token
    texts = [
        f' {sep} '.join([r[query_key]] + r['neighbors'])
        for r in records
    ]
    encoded = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )
    encoded['labels'] = [r['label'] for r in records]
    return Dataset.from_dict(encoded)


def tokenize_plain(hf_dataset, tokenizer, text_col='post', max_length=256):
    """
    Plain tokenization for baseline / no-retrieval evaluation.

    Parameters
    ----------
    text_col : str
        Column name for input text. Default 'post' (IHC convention).
    """
    encoded = tokenizer(
        list(hf_dataset[text_col]),
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )
    encoded['labels'] = list(hf_dataset['label'])
    return Dataset.from_dict(encoded)


def filter_records(cached, k, threshold):
    """
    Filter cached (text, score) neighbor pairs to a specific (k, threshold).

    Used after augment_split_cached to apply model-specific hyperparameters
    without re-encoding.
    """
    return [
        {
            'query':     r['query'],
            'neighbors': [text for text, score in r['neighbors'] if score >= threshold][:k],
            'label':     r['label'],
        }
        for r in cached
    ]


def strip_label_prefix(text):
    """Remove [hate] or [not hate] prefix from a chunk text."""
    return text.replace('[hate] ', '', 1).replace('[not hate] ', '', 1)


def set_seed(seed):
    """Set random seeds for full reproducibility (Python, NumPy, PyTorch, CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
