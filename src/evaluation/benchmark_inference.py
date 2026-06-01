"""
Benchmark inference throughput of the full RAC pipeline:
  SBERT encode query → FAISS lookup → RoBERTa forward pass

Reports samples/sec and GPU memory, matching the format of
Cheremetiev et al. Table 8 for a fair comparison.

Usage:
    cd /scratch/deep_learning/RAG
    python benchmark_inference.py
"""

import time
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset
from src.rag import retrieve_top_k_above_threshold

# ── Config ────────────────────────────────────────────────────────────────────
RAG_DIR         = Path('.')
INDEX_DIR       = RAG_DIR / 'index'
CLASSIFIER_DIR  = Path('..') / 'weights_rag' / 'roberta' / 'sbert' / 'full' / 'IHC'
RETRIEVER_HF_ID = 'sentence-transformers/all-mpnet-base-v2'

K          = 5
THRESHOLD  = 0.4
MAX_LENGTH = 256
BATCH_SIZE = 16       # match Cheremetiev et al. eval batch size
N_WARMUP   = 50       # samples to discard before timing
N_SAMPLES  = 500      # samples to time (IHC test = 2148, we use a subset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load models ───────────────────────────────────────────────────────────────
print('\nLoading retriever ...')
ret_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_HF_ID)
ret_model     = AutoModel.from_pretrained(RETRIEVER_HF_ID).eval().to(device)

print('Loading FAISS full index ...')
ret_index = faiss.read_index(str(INDEX_DIR / 'sbert' / 'vdb_full.faiss'))
with open(INDEX_DIR / 'lookup_full.json') as f:
    ret_documents = json.load(f)
print(f'  {ret_index.ntotal:,} vectors')

print('Loading RoBERTa(E+K) classifier ...')
clf_tokenizer = AutoTokenizer.from_pretrained(str(CLASSIFIER_DIR))
clf_model     = AutoModelForSequenceClassification.from_pretrained(
    str(CLASSIFIER_DIR), num_labels=2
).eval().to(device)

# ── Load test data ────────────────────────────────────────────────────────────
print('\nLoading IHC test split ...')
raw_ihc = load_dataset('tasksource/implicit-hate-stg1', split='train')
splits  = raw_ihc.train_test_split(test_size=0.10, seed=42)
test_texts = [ex['post'] for ex in splits['test']]
print(f'  {len(test_texts):,} test samples, using {N_WARMUP + N_SAMPLES}')

samples = test_texts[:N_WARMUP + N_SAMPLES]

# ── Full pipeline: augment + classify one sample at a time ────────────────────
def infer_single(text):
    """Full RAC inference for a single query."""
    # Step 1: retrieve
    neighbors = retrieve_top_k_above_threshold(
        text, THRESHOLD, ret_model, ret_tokenizer,
        ret_index, ret_documents,
        chunk_id=None, k=K, use_mean_pool=True,
    )
    neighbor_texts = [t for t, _ in neighbors]

    # Step 2: build augmented input and classify
    sep = clf_tokenizer.sep_token
    augmented = f' {sep} '.join([text] + neighbor_texts)
    inputs = clf_tokenizer(
        augmented,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        logits = clf_model(**inputs).logits
    return int(torch.argmax(logits, dim=-1).item())


# ── Warm up ───────────────────────────────────────────────────────────────────
print(f'\nWarming up ({N_WARMUP} samples) ...')
for text in samples[:N_WARMUP]:
    infer_single(text)
if device.type == 'cuda':
    torch.cuda.synchronize()

# ── Measure GPU memory after warmup ──────────────────────────────────────────
if device.type == 'cuda':
    mem_gb = torch.cuda.memory_allocated(device) / 1024**3
    mem_reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
    print(f'\nGPU memory allocated : {mem_gb:.2f} GB')
    print(f'GPU memory reserved  : {mem_reserved_gb:.2f} GB')
else:
    print('\n(CPU only — no GPU memory to report)')

# ── Timed run ─────────────────────────────────────────────────────────────────
print(f'\nTiming {N_SAMPLES} samples ...')
timed_samples = samples[N_WARMUP:]

if device.type == 'cuda':
    torch.cuda.synchronize()
t0 = time.perf_counter()

for text in timed_samples:
    infer_single(text)

if device.type == 'cuda':
    torch.cuda.synchronize()
t1 = time.perf_counter()

elapsed    = t1 - t0
throughput = N_SAMPLES / elapsed

print(f'\n{"="*50}')
print(f'Elapsed          : {elapsed:.2f} s')
print(f'Throughput       : {throughput:.1f} samples/sec')
print(f'Latency/sample   : {1000 * elapsed / N_SAMPLES:.1f} ms')
if device.type == 'cuda':
    print(f'GPU memory       : {mem_gb:.2f} GB allocated / {mem_reserved_gb:.2f} GB reserved')
print(f'{"="*50}')

# ── Twitter back-of-envelope ──────────────────────────────────────────────────
tweets_per_day = 500_000_000
tweets_per_sec = tweets_per_day / 86_400
gpus_needed    = tweets_per_sec / throughput

print(f'\nTwitter-scale estimate ({tweets_per_day/1e6:.0f}M tweets/day):')
print(f'  Tweets/sec     : {tweets_per_sec:.0f}')
print(f'  GPUs needed    : {gpus_needed:.1f}')
