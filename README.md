# Retrieval-Augmented Classifier (RAC) for Hate Speech Detection

A three-stage pipeline for implicit hate speech detection that augments standard transformer classifiers with FAISS-based nearest-neighbor retrieval. The retriever finds semantically similar training examples at inference time; these are prepended to the input, giving the classifier richer context without retraining the retriever.

## Architecture

```
Input tweet
    │
    ▼
[Layer 1] SBERT retriever (all-mpnet-base-v2, frozen)
    │  FAISS cosine search → top-k neighbors above threshold
    ▼
Augmented input: "tweet [SEP] [hate] neighbor1 [SEP] [not hate] neighbor2 ..."
    │
    ▼
[Layer 2] Fine-tuned classifier (BERT / HateBERT / RoBERTa)
    │  Binary: hate / not hate
    ▼
[Layer 3] LLM explainer (optional)
           Produces a structured moderation report
```

Self-exclusion at train time: each tweet's own `chunk_id` is passed to the retriever so it cannot retrieve itself from the index, preventing data leakage.

## Datasets

| Dataset | Task | Size (train/test) | Source |
|---------|------|-------------------|--------|
| IHC (Implicit Hate Corpus) | Binary hate / not hate | 19,332 / 2,148 | ElSherief et al. 2021 — `tasksource/implicit-hate-stg1` on HuggingFace |
| ISHate | Binary + 3-class | 55,023 / 4,368 | Ocampo et al. EACL 2023 — `BenjaminOcampo/ISHate` on HuggingFace |
| Vicomtech | Binary hate / not hate | 1,914 / 478 | Vicomtech GitHub repo |

## Models

**Classifier backbone** (fine-tuned): `bert-base-uncased`, `GroNLP/hateBERT`, `roberta-base`

**Retriever** (frozen): `sentence-transformers/all-mpnet-base-v2` — mean-pooled embeddings, L2-normalized, inner product = cosine similarity

## FAISS Indices

Three indices are built from different corpora:

| Index | Vectors | Content |
|-------|---------|---------|
| `vdb_example.faiss` | ~67,864 | Training tweets from IHC + ISHate + Vicomtech |
| `vdb_knowledge.faiss` | ~40,952 | Hate symbol definitions scraped from ADL/SPLC |
| `vdb_full.faiss` | ~108,816 | Union of example + knowledge (knowledge IDs offset by 100,000) |

Each index is paired with a `lookup_*.json` mapping `chunk_id → "[hate] text"` or `"[not hate] text"`.

## Best Hyperparameters

Found by grid search (K ∈ {3, 5, 10}, threshold ∈ {0.3, 0.4, 0.5, 0.6}) on IHC with the full index:

| Model | K | Threshold |
|-------|---|-----------|
| BERT | 5 | 0.5 |
| HateBERT | 3 | 0.4 |
| RoBERTa | 3 | 0.4 |

Training config: `lr=2e-5`, `batch_size=16`, `epochs=3`, `max_length=256`.

---

## Repository Structure

```
deep_learning/
├── src/
│   ├── retriever.py              # encode(), retrieve_top_k_above_threshold(), etc.
│   ├── training_utils.py         # compute_metrics(), tokenize_augmented(), filter_records(), set_seed()
│   ├── data_loaders.py           # load_ihc_binary(), load_ishate_binary(), load_vicomtech(), …
│   │
│   ├── retriever/
│   │   ├── chunk_example.ipynb         # Build chunks_example.csv from IHC + ISHate + Vicomtech
│   │   ├── scrap_chunk_knowledge.ipynb # Scrape ADL/SPLC → chunks_knowledge.csv (needs Firecrawl API key)
│   │   ├── index.ipynb                 # Encode chunks with SBERT → build vdb_*.faiss + lookup_*.json
│   │   └── retrieval.ipynb             # Interactive retrieval analysis and verification
│   │
│   ├── training/
│   │   ├── training_baseline.ipynb         # Fine-tune BERT/HateBERT/RoBERTa without retrieval (9 runs)
│   │   ├── hyperparameter_tuning.ipynb     # Grid search K × threshold on IHC/full index (36 runs)
│   │   ├── training.ipynb                  # RAC fine-tuning: 3 models × 3 indices × 3 datasets (27 runs)
│   │   ├── training_multiple_seed.ipynb    # RoBERTa + full index, 3 seeds — run once per seed
│   │   └── training_cross_evaluation.ipynb # Train on ISHate/Vicomtech, evaluate on IHC (12 runs)
│   │
│   ├── evaluation/
│   │   ├── evaluate_all.ipynb                  # All 36 configs → results/all_results.json
│   │   ├── evaluation_IHC.ipynb                # Compare vs. ElSherief et al. 2021 on IHC
│   │   ├── evaluation_ISHate.ipynb             # Replication of Ocampo et al. EACL 2023 (3-class)
│   │   ├── evaluation_rac_without_retriever.ipynb  # RAC weights, plain-text inference (ablation)
│   │   └── inference_latency.ipynb             # End-to-end throughput and latency benchmark
│   │
│   └── LLM/
│       ├── llm_explainer.py    # LLM-based moderation report generator (Layer 3)
│       └── llm_demo.ipynb      # Full 3-layer pipeline demo → HTML report
│
├── corpus/
│   ├── chunks/
│   │   ├── chunks_example.csv          # 67,864 labeled tweets (IHC + ISHate + Vicomtech train)
│   │   ├── chunks_example_noihc.csv    # Same minus IHC — used for cross-dataset study
│   │   └── chunks_knowledge.csv        # ~41k hate symbol definitions
│   └── index/
│       ├── vdb_example.faiss / lookup_example.json
│       ├── vdb_knowledge.faiss / lookup_knowledge.json
│       └── vdb_full.faiss / lookup_full.json
│
├── weigths/                              # Saved model checkpoints
│   ├── weights_baseline/                 # {model}/{dataset}/
│   ├── weights_rac_best_hyperparameters/ # {model}/sbert/{index_type}/{dataset}/
│   ├── weights_rac_hyperparameter_tuning/
│   ├── weights_rac_multiseed/
│   ├── weights_rac_cross_evaluation/
│   └── weights_rac_noihc_index/
│
├── results/                  # Executed notebooks and JSON outputs
└── ee559_docker_env/         # Dockerfile and requirements for the RCP cluster image
```

---

## Running the Pipeline

### Prerequisites

All notebooks are designed to run inside the pre-built Docker image `registry.rcp.epfl.ch/ee-559-potocnik/my-toolbox:v0.1` on the RCP cluster. See `ee559_docker_env/` for the Dockerfile and pinned dependencies. There is no supported local installation path.

The corpus data (`corpus/chunks/` and `corpus/index/`) must exist before running any training notebook. `chunks_knowledge.csv` requires re-running `scrap_chunk_knowledge.ipynb` with a Firecrawl API key if not already present. All other chunk and index files can be rebuilt from scratch with the steps below.

### Execution Order

The pipeline has four sequential stages. Within each stage, notebooks can run in parallel.

#### Stage 1 — Build the corpus (if not already present)

```
src/retriever/chunk_example.ipynb     → corpus/chunks/chunks_example.csv
src/retriever/index.ipynb             → corpus/index/vdb_*.faiss + lookup_*.json
```

#### Stage 2 — Training (all parallel)

```
src/training/training_baseline.ipynb        → weigths/weights_baseline/
src/training/hyperparameter_tuning.ipynb    → weigths/weights_rac_hyperparameter_tuning/
src/training/training.ipynb                 → weigths/weights_rac_best_hyperparameters/
src/training/training_multiple_seed.ipynb   → weigths/weights_rac_multiseed/   (run 3×: TRAINING_SEED=0,1,2)
src/training/training_cross_evaluation.ipynb → weigths/weights_rac_cross_evaluation/
src/evaluation/evaluation_ISHate.ipynb      → trains and evaluates internally
```

#### Stage 3 — Evaluation (all parallel, needs Stage 2)

```
src/evaluation/evaluate_all.ipynb                  → results/all_results.json
src/evaluation/evaluation_IHC.ipynb                → comparison table vs. ElSherief 2021
src/evaluation/evaluation_rac_without_retriever.ipynb → ablation tables
src/evaluation/inference_latency.ipynb             → latency/throughput report
```

### RCP Cluster (RunAI)

Each notebook maps to a `runai submit` job. Key parameters:

```bash
runai submit --name <job-name> \
  --run-as-uid 258393 \
  --image registry.rcp.epfl.ch/ee-559-potocnik/my-toolbox:v0.1 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-559-scratch-g44,path=/scratch \
  --existing-pvc claimname=home,path=/home/potocnik \
  --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
  --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
  -- bash -lc "set -euo pipefail; mkdir -p /scratch/project/deep_learning/results; \
    cd /scratch/project/deep_learning/src/<folder>; \
    jupyter nbconvert --to notebook --execute <notebook>.ipynb \
      --output <notebook>.executed.ipynb \
      --output-dir /scratch/project/deep_learning/results"
```

Use `--gpu 1` for all jobs. The multi-seed notebook takes a `TRAINING_SEED` environment variable; submit three separate jobs with `TRAINING_SEED=0`, `TRAINING_SEED=1`, `TRAINING_SEED=2`.

### LLM Demo (local)

`src/LLM/llm_demo.ipynb` is designed to run locally, not on the cluster. It requires one extra package not in the Docker image:

```bash
pip install groq
```

It also needs a free [Groq API key](https://console.groq.com) set in the notebook.

---

## Results

Executed notebooks and JSON result files are stored in `results/`. Key outputs:

| File | Contents |
|------|----------|
| `results/all_results.json` | Macro F1/P/R for all 36 model × index × dataset configurations |
| `results/multiseed_results_s{0,1,2}.json` | Per-seed F1 for RoBERTa + full index across 3 datasets |
| `results/hyperparameter_tuning.executed.ipynb` | HP grid heatmaps per model |
| `results/evaluate_all.executed.ipynb` | Full summary table |
| `results/evaluation_IHC.executed.ipynb` | Comparison vs. ElSherief et al. 2021 |
| `results/evaluation_ISHate.executed.ipynb` | Comparison vs. Ocampo et al. EACL 2023 |
| `results/inference_latency.executed.ipynb` | Throughput (samples/sec) and GPU memory |

---

## Key Implementation Details

**Augmented input format**

```
tweet_text [SEP] [hate] similar_tweet_1 [SEP] [not hate] similar_tweet_2 ...
```

Label prefixes (`[hate]` / `[not hate]`) are prepended to each neighbor so the classifier can use the retrieved label as a signal, not just the text.

**Self-exclusion**

At train time, the tweet's own `chunk_id` (from `chunks_example.csv`) is passed to `retrieve_top_k_above_threshold`. The function fetches `k+1` candidates and discards the self-match by ID. At test time, `chunk_id=None` and no exclusion is applied.

**Caching**

`training.ipynb` and `evaluate_all.ipynb` augment the full split once at `(MAX_K, MIN_THRESHOLD)` and call `filter_records()` to apply per-model `(k, threshold)` without re-encoding. This avoids redundant SBERT forward passes when comparing multiple models on the same index.

**Retriever**

`src/retriever.py` exposes three retrieval functions:

- `retrieve_top_k` — exactly k neighbors, no threshold
- `retrieve_by_threshold` — all neighbors above a threshold, no cap
- `retrieve_top_k_above_threshold` — up to k neighbors above a threshold (used in training)
