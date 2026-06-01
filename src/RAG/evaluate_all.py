"""
Evaluates all 36 trained models (9 no-index + 27 RAC) and saves
full-precision F1/P/R results to ../results/all_results.json.

Run on RCP:
    cd /scratch/deep_learning/RAG && python evaluate_all.py
"""

import os, json
from pathlib import Path
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import urllib.request, zipfile, shutil, pandas as pd
import warnings
warnings.filterwarnings('ignore')
from src.RAG.rag import retrieve_top_k_above_threshold

# ── Paths ─────────────────────────────────────────────────────────────────────
RAG_DIR         = Path('.')
INDEX_DIR       = RAG_DIR / 'index'
WEIGHTS_DIR     = Path('..') / 'weights'
WEIGHTS_RAG_DIR = Path('..') / 'weights_rag_best_hp'
RESULTS_FILE    = Path('..') / 'results' / 'all_results.json'

RETRIEVER_HF_ID = 'sentence-transformers/all-mpnet-base-v2'

BEST_PARAMS = {
    'bert':     {'k': 3, 'threshold': 0.4},
    'hatebert': {'k': 3, 'threshold': 0.5},
    'roberta':  {'k': 5, 'threshold': 0.4},
}

MAX_K         = max(p['k']         for p in BEST_PARAMS.values())
MIN_THRESHOLD = min(p['threshold'] for p in BEST_PARAMS.values())
MAX_LENGTH    = 256
BATCH_SIZE    = 64

MODEL_MAP = {
    'bert':     'bert-base-uncased',
    'hatebert': 'GroNLP/hateBERT',
    'roberta':  'roberta-base',
}

BASELINE_WEIGHT_MAP = {
    'bert':     'bert-base-uncased',
    'hatebert': 'hateBERT',
    'roberta':  'roberta-base',
}

INDEX_TYPES = ['training', 'documents', 'full']
DATASETS    = ['IHC', 'ISHate', 'Vicomtech']
MODELS      = ['bert', 'hatebert', 'roberta']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load test datasets ────────────────────────────────────────────────────────
print('\nLoading datasets ...')
raw_ihc = load_dataset('tasksource/implicit-hate-stg1', split='train')
splits  = raw_ihc.train_test_split(test_size=0.10, seed=42)

def add_binary_label_ihc(example):
    example['label'] = 0 if example['class'] == 'not_hate' else 1
    return example

test_ihc = splits['test'].map(add_binary_label_ihc)

ishate_raw = load_dataset('BenjaminOcampo/ISHate')

def add_binary_label_ishate(example):
    example['label'] = 0 if example['hateful_layer'] == 'Non-HS' else 1
    return example

test_ishate = ishate_raw['test'].map(add_binary_label_ishate)

_repo_dir      = str(RAG_DIR / 'data' / 'hate-speech-dataset')
_metadata_path = f'{_repo_dir}/annotations_metadata.csv'
_test_dir      = f'{_repo_dir}/sampled_test'

if not all(os.path.exists(p) for p in [_metadata_path, _test_dir]):
    if os.path.isdir(_repo_dir):
        shutil.rmtree(_repo_dir)
    os.makedirs(str(RAG_DIR / 'data'), exist_ok=True)
    _zip_url  = 'https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip'
    _zip_path = str(RAG_DIR / 'data' / 'hate-speech-dataset.zip')
    urllib.request.urlretrieve(_zip_url, _zip_path)
    with zipfile.ZipFile(_zip_path, 'r') as zf:
        zf.extractall(str(RAG_DIR / 'data'))
    os.rename(str(RAG_DIR / 'data' / 'hate-speech-dataset-master'), _repo_dir)
    os.remove(_zip_path)

_metadata = pd.read_csv(_metadata_path).set_index('file_id')

def _load_vicomtech_test(split_dir):
    rows = []
    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith('.txt'):
            continue
        file_id = fname[:-4]
        if file_id not in _metadata.index:
            continue
        label_str = _metadata.loc[file_id, 'label']
        if label_str not in ('hate', 'noHate'):
            continue
        with open(os.path.join(split_dir, fname), encoding='utf-8') as f:
            text = f.read().strip()
        rows.append({'text': text, 'label': 1 if label_str == 'hate' else 0})
    return Dataset.from_list(rows)

test_vicomtech = _load_vicomtech_test(_test_dir)

DATASET_MAP = {
    'IHC':       {'test': test_ihc,       'text_col': 'post'},
    'ISHate':    {'test': test_ishate,     'text_col': 'text'},
    'Vicomtech': {'test': test_vicomtech,  'text_col': 'text'},
}
print(f'  IHC test: {len(test_ihc):,}  ISHate test: {len(test_ishate):,}  Vicomtech test: {len(test_vicomtech):,}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_metrics_from_arrays(preds, labels):
    return {
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'macro_p':  precision_score(labels, preds, average='macro', zero_division=0),
        'macro_r':  recall_score(labels, preds, average='macro', zero_division=0),
    }

def evaluate_model(model, tokenizer, hf_dataset):
    """Run inference and return full-precision metrics."""
    texts  = [ex[DATASET_MAP[ds_name]['text_col']] for ex in hf_dataset]
    labels = [ex['label'] for ex in hf_dataset]

    all_preds = []
    model.eval()
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch_texts, truncation=True, padding=True,
                           max_length=MAX_LENGTH, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return compute_metrics_from_arrays(all_preds, labels)

def augment_test(hf_dataset, text_col, ret_model, ret_tokenizer, ret_index, ret_documents, k, threshold):
    """Augment test split (no self-exclusion needed)."""
    records = []
    for example in tqdm(hf_dataset, desc='augment', leave=False):
        tweet = example[text_col]
        neighbors = retrieve_top_k_above_threshold(
            tweet, threshold, ret_model, ret_tokenizer, ret_index, ret_documents,
            chunk_id=None, k=k, use_mean_pool=True,
        )
        records.append({
            'query':     tweet,
            'neighbors': [t for t, _ in neighbors],
            'label':     example['label'],
        })
    return records

def tokenize_augmented(records, tokenizer):
    sep = tokenizer.sep_token
    texts  = [f' {sep} '.join([r['query']] + r['neighbors']) for r in records]
    labels = [r['label'] for r in records]
    encoded = tokenizer(texts, truncation=True, padding='max_length',
                        max_length=MAX_LENGTH)
    encoded['labels'] = labels
    return Dataset.from_dict(encoded)

def evaluate_augmented(clf_model, clf_tokenizer, records):
    """Evaluate a RAC model on pre-augmented records."""
    tok = tokenize_augmented(records, clf_tokenizer)
    clf_model.eval()
    all_preds = []
    for i in range(0, len(tok), BATCH_SIZE):
        batch = tok[i:i+BATCH_SIZE]
        input_ids      = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        with torch.no_grad():
            logits = clf_model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    labels = tok['labels']
    return compute_metrics_from_arrays(all_preds, labels)


# ── Evaluate no-index baselines ───────────────────────────────────────────────
results = {}

print('\n' + '='*60)
print('EVALUATING NO-INDEX BASELINES')
print('='*60)

for model_key in MODELS:
    weight_name = BASELINE_WEIGHT_MAP[model_key]
    for ds_name in DATASETS:
        weight_path = WEIGHTS_DIR / f'{weight_name}_{ds_name}'
        print(f'  {model_key} | no-index | {ds_name} ...', end=' ', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(weight_path))
        model     = AutoModelForSequenceClassification.from_pretrained(str(weight_path)).to(device)
        ds_cfg    = DATASET_MAP[ds_name]

        # Plain inference (no augmentation)
        texts  = [ex[ds_cfg['text_col']] for ex in ds_cfg['test']]
        labels = [ex['label'] for ex in ds_cfg['test']]
        all_preds = []
        model.eval()
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_texts, truncation=True, padding=True,
                               max_length=MAX_LENGTH, return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        metrics = compute_metrics_from_arrays(all_preds, labels)
        results[(model_key, 'no-index', ds_name)] = metrics
        print(f'F1={metrics["macro_f1"]:.4f}')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# ── Load retriever once ───────────────────────────────────────────────────────
print('\n' + '='*60)
print('EVALUATING RAC MODELS')
print('='*60)
print(f'\nLoading retriever: {RETRIEVER_HF_ID} ...')
ret_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_HF_ID)
ret_model     = AutoModel.from_pretrained(RETRIEVER_HF_ID).eval().to(device)
print('Retriever ready.\n')

for index_type in INDEX_TYPES:
    ret_index = faiss.read_index(str(INDEX_DIR / 'sbert' / f'vdb_{index_type}.faiss'))
    with open(INDEX_DIR / f'lookup_{index_type}.json') as f:
        ret_documents = json.load(f)
    print(f'\n--- Index: {index_type} ({ret_index.ntotal:,} vectors) ---')

    # Augment all datasets once per index (cached)
    aug_cache = {}
    for ds_name in DATASETS:
        ds_cfg = DATASET_MAP[ds_name]
        k, threshold = MAX_K, MIN_THRESHOLD  # cache at max, filter per model
        aug_cache[ds_name] = augment_test(
            ds_cfg['test'], ds_cfg['text_col'],
            ret_model, ret_tokenizer, ret_index, ret_documents,
            k=k, threshold=threshold,
        )

    for model_key in MODELS:
        k         = BEST_PARAMS[model_key]['k']
        threshold = BEST_PARAMS[model_key]['threshold']

        for ds_name in DATASETS:
            weight_path = WEIGHTS_RAG_DIR / model_key / 'sbert' / index_type / ds_name
            print(f'  {model_key} | {index_type} | {ds_name} ...', end=' ', flush=True)

            clf_tokenizer = AutoTokenizer.from_pretrained(str(weight_path))
            clf_model     = AutoModelForSequenceClassification.from_pretrained(
                str(weight_path), num_labels=2).to(device)

            # Filter cached augmentation to model-specific (k, threshold)
            filtered = [
                {'query': r['query'],
                 'neighbors': [t for t, s in r['neighbors'] if s >= threshold][:k],
                 'label': r['label']}
                for r in aug_cache[ds_name]
            ]

            metrics = evaluate_augmented(clf_model, clf_tokenizer, filtered)
            results[(model_key, index_type, ds_name)] = metrics
            print(f'F1={metrics["macro_f1"]:.4f}')

            del clf_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

del ret_model
if device.type == 'cuda':
    torch.cuda.empty_cache()

# ── Save results ──────────────────────────────────────────────────────────────
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
serializable = {f'{m}|{it}|{ds}': v for (m, it, ds), v in results.items()}
with open(RESULTS_FILE, 'w') as f:
    json.dump(serializable, f, indent=2)
print(f'\nSaved to {RESULTS_FILE}')

# ── Print summary table ───────────────────────────────────────────────────────
print('\nSummary (Macro F1):')
header = f'{"Model":<10} {"Index":<12}  {"IHC":>6}  {"ISHate":>7}  {"Vicomtech":>10}'
print(header)
print('-' * len(header))
for model_key in MODELS:
    for index_type in ['no-index'] + INDEX_TYPES:
        row = f'{model_key:<10} {index_type:<12}'
        for ds_name in DATASETS:
            v = results.get((model_key, index_type, ds_name), {}).get('macro_f1', float('nan'))
            row += f'  {v:>6.4f}'
        print(row)
    print()
