import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    """
    HuggingFace Trainer callback returning macro F1, Precision and Recall.

    Input:
        eval_pred : EvalPrediction — object with .predictions and .label_ids

    Output:
        dict with keys 'macro_f1', 'macro_p', 'macro_r'
    """
    preds  = np.argmax(eval_pred.predictions, axis=-1)  # convert raw logits to predicted class indices
    labels = eval_pred.label_ids
    return {
        'macro_f1': f1_score(labels, preds, average='macro',  zero_division=0),   # zero_division=0 returns 0 instead of raising a warning when a class has no predicted samples
        'macro_p':  precision_score(labels, preds, average='macro', zero_division=0),
        'macro_r':  recall_score(labels, preds, average='macro',    zero_division=0),
    }


def tokenize_augmented(records, tokenizer, max_length=256, query_key='query'):
    """
    Build augmented input strings and tokenize for the RAC classifier.
    Format: query [SEP] neighbor1 [SEP] neighbor2 ...

    Input:
        records    : list[dict] — each dict must have query_key (str),
                     'neighbors' (list[str]) and 'label' (int)
        tokenizer  : HuggingFace tokenizer
        max_length : int — max token length
        query_key  : str — key for the query text in each record

    Output:
        HuggingFace Dataset ready for the Trainer
    """
    sep = tokenizer.sep_token  # use the model's actual separator token (e.g. "[SEP]" for BERT, "</s>" for RoBERTa)
    texts = [
        f' {sep} '.join([r[query_key]] + r['neighbors'])  # concatenate query and all neighbors with SEP in between
        for r in records
    ]
    encoded = tokenizer(
        texts,
        truncation=True,
        padding='max_length',  # pad every sequence to the same length so they can be stacked into a uniform batch tensor
        max_length=max_length,
    )
    encoded['labels'] = [r['label'] for r in records]  # attach labels so HF Trainer can compute the loss without extra arguments
    return Dataset.from_dict(encoded)


def tokenize_plain(hf_dataset, tokenizer, text_col='post', max_length=256):
    """
    Plain tokenization for baseline / no-retrieval evaluation.

    Input:
        hf_dataset : HuggingFace Dataset with a 'label' column
        tokenizer  : HuggingFace tokenizer
        text_col   : str — column name for the input text
        max_length : int — max token length

    Output:
        HuggingFace Dataset ready for the Trainer
    """
    encoded = tokenizer(
        list(hf_dataset[text_col]),  # HF Dataset columns are Arrow arrays; convert to a plain list as the tokenizer expects
        truncation=True,
        padding='max_length',
        max_length=max_length,
    )
    encoded['labels'] = list(hf_dataset['label'])
    return Dataset.from_dict(encoded)


def filter_records(cached, k, threshold):
    """
    Filter cached (text, score) neighbor pairs to a specific (k, threshold).
    Used after augment_split_cached to apply model-specific hyperparameters without re-encoding.

    Input:
        cached    : list[dict] — records with 'neighbors' as list of (text, score) pairs
        k         : int — max neighbors to keep
        threshold : float — minimum cosine similarity to keep

    Output:
        list[dict] — same structure with 'neighbors' as list[str]
    """
    return [
        {
            'query':     r['query'],
            'neighbors': [text for text, score in r['neighbors'] if score >= threshold][:k],  # apply threshold first, then cap at k; order matters
            'label':     r['label'],
        }
        for r in cached
    ]


def strip_label_prefix(text):
    """Remove [hate] or [not hate] prefix from a chunk text."""
    return text.replace('[hate] ', '', 1).replace('[not hate] ', '', 1)  # the third argument 1 limits replacement to the first occurrence only


def set_seed(seed):
    """
    Set random seeds for full reproducibility across Python, NumPy, PyTorch and CUDA.

    Input:
        seed : int — seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # seed every GPU, not just the default one
