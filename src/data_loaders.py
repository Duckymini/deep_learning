"""
Dataset loaders for the hate speech detection pipeline.

Three datasets, each with multiple loading modes:
  - IHC        (tasksource/implicit-hate-stg1)     — HuggingFace
  - ISHate     (BenjaminOcampo/ISHate)             — HuggingFace
  - Vicomtech  (Vicomtech/hate-speech-dataset)     — GitHub ZIP, loaded in-memory

All functions return HuggingFace Dataset objects or list[dict], ready to use.
No local data folder is created or required.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
import zipfile
from typing import Literal

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# IHC — tasksource/implicit-hate-stg1
# ---------------------------------------------------------------------------

def load_ihc_binary(seed: int = 42):
    """
    IHC binary classification, 90/10 train/test split.

    label : 0 = not_hate, 1 = hate (explicit + implicit collapsed)
    text_col : 'post'
    seed : controls the random train/test split — use seed=42 for the
           canonical split shared across all experiments.

    Returns (train, test) as HF Datasets.
    """
    raw = load_dataset("tasksource/implicit-hate-stg1", split="train")
    splits = raw.train_test_split(test_size=0.10, seed=seed)

    def _label(x):
        x["label"] = 0 if x["class"] == "not_hate" else 1
        return x

    return splits["train"].map(_label), splits["test"].map(_label)


def load_ihc_implicit_only(seed: int = 42):
    """
    IHC test split with explicit_hate rows removed — for evaluation_IHC.ipynb.
    Replicates the binary not_hate vs implicit_hate setup from ElSherief et al.

    label : 0 = not_hate, 1 = implicit_hate
    text_col : 'post'

    Returns test HF Dataset.
    """
    raw = load_dataset("tasksource/implicit-hate-stg1", split="train")
    test = raw.train_test_split(test_size=0.10, seed=seed)["test"]
    test = test.filter(lambda x: x["class"] != "explicit_hate")
    return test.map(lambda x: {"label": 0 if x["class"] == "not_hate" else 1})


def load_ihc_chunks(seed: int = 42):
    """
    IHC training split formatted for the chunking pipeline.

    label : string "hate" / "not_hate" (kept as text for the chunk prefix)

    Returns list[{"text": str, "label": str}].
    """
    raw = load_dataset("tasksource/implicit-hate-stg1", split="train")
    train = raw.train_test_split(test_size=0.10, seed=seed)["train"]
    return [
        {
            "text": row["post"],
            "label": "not_hate" if row["class"] == "not_hate" else "hate",
        }
        for row in train
    ]


# ---------------------------------------------------------------------------
# ISHate — BenjaminOcampo/ISHate
# ---------------------------------------------------------------------------

def load_ishate_binary():
    """
    ISHate binary classification using hateful_layer column.
    Uses the pre-made train/test splits.

    label : 0 = Non-HS, 1 = hate
    text_col : 'text'

    Returns (train, test) as HF Datasets.
    """
    raw = load_dataset("BenjaminOcampo/ISHate")

    def _label(x):
        x["label"] = 0 if x["hateful_layer"] == "Non-HS" else 1
        return x

    return raw["train"].map(_label), raw["test"].map(_label)


def load_ishate_multiclass():
    """
    ISHate 3-class classification for Ocampo et al. (EACL 2023) Tasks A and B.
    Uses the pre-made train/validation/test splits.

    Task A (label_a) : 0=Non-HS, 1=Explicit HS, 2=Implicit HS
    Task B (label_b) : 0=Non-HS, 1=Non-Subtle HS, 2=Subtle HS
    text_col : 'text'

    Returns (train, val, test) as HF Datasets.
    """
    raw = load_dataset("BenjaminOcampo/ISHate")

    def _label(x):
        if x["hateful_layer"] == "Non-HS":
            x["label_a"], x["label_b"] = 0, 0
        else:
            x["label_a"] = 1 if x["implicit_layer"] == "Explicit HS" else 2
            x["label_b"] = 1 if x["subtlety_layer"] == "Non-Subtle HS" else 2
        return x

    return (
        raw["train"].map(_label),
        raw["validation"].map(_label),
        raw["test"].map(_label),
    )


def load_ishate_chunks():
    """
    ISHate training split formatted for the chunking pipeline.
    Uses hf_hub_download (parquet) to avoid streaming the full dataset.

    label : string "hate" / "not_hate"

    Returns list[{"text": str, "label": str}].
    """
    path = hf_hub_download(
        repo_id="BenjaminOcampo/ISHate",
        filename="ishate_train.parquet.gzip",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    return [
        {
            "text": row["text"],
            "label": "not_hate" if row["hateful_layer"] == "Non-HS" else "hate",
        }
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Vicomtech — Vicomtech/hate-speech-dataset (GitHub)
# ---------------------------------------------------------------------------

_VICOMTECH_URL = (
    "https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip"
)


def _load_vicomtech_raw() -> dict:
    """
    Download the Vicomtech ZIP from GitHub into a temporary directory,
    parse both splits, and return the rows as plain lists.

    The temporary directory is deleted automatically after this function returns.
    No persistent storage is created.

    Returns {"train": list[dict], "test": list[dict]} with label as int 0/1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")
        print("Downloading Vicomtech dataset from GitHub...")
        urllib.request.urlretrieve(_VICOMTECH_URL, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        repo = os.path.join(tmpdir, "hate-speech-dataset-master")
        meta = pd.read_csv(
            os.path.join(repo, "annotations_metadata.csv")
        ).set_index("file_id")

        result = {}
        for split_key, dir_name in [("train", "sampled_train"), ("test", "sampled_test")]:
            split_dir = os.path.join(repo, dir_name)
            rows = []
            for fname in sorted(os.listdir(split_dir)):
                if not fname.endswith(".txt"):
                    continue
                file_id = fname[:-4]
                if file_id not in meta.index:
                    continue
                label_str = meta.loc[file_id, "label"]
                if label_str not in ("hate", "noHate"):
                    continue
                with open(os.path.join(split_dir, fname), encoding="utf-8") as f:
                    text = f.read().strip()
                if not text:
                    continue
                rows.append({"text": text, "label": 1 if label_str == "hate" else 0})
            result[split_key] = rows

        return result  # temp dir cleaned up here


def load_vicomtech(split: Literal["train", "test", "both"] = "both"):
    """
    Vicomtech binary classification.
    Downloads the dataset from GitHub on each call (no local storage).

    label : 0 = not hate, 1 = hate
    text_col : 'text'
    split : "train" | "test" | "both"

    Returns a single HF Dataset (train or test) or a (train, test) tuple.
    """
    raw = _load_vicomtech_raw()
    train_ds = Dataset.from_list(raw["train"])
    test_ds  = Dataset.from_list(raw["test"])

    if split == "train":
        return train_ds
    if split == "test":
        return test_ds
    return train_ds, test_ds


def load_vicomtech_chunks():
    """
    Vicomtech training split formatted for the chunking pipeline.
    Downloads from GitHub on each call (no local storage).

    label : string "hate" / "not_hate"

    Returns list[{"text": str, "label": str}].
    """
    raw = _load_vicomtech_raw()
    return [
        {
            "text": row["text"],
            "label": "hate" if row["label"] == 1 else "not_hate",
        }
        for row in raw["train"]
    ]
