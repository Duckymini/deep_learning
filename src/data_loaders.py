import os
import tempfile
import urllib.request
import zipfile
from typing import Literal

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


def load_ihc_binary(seed: int = 42):
    """
    IHC binary classification with a 90/10 train/test split.

    Input:
        seed : int — controls the random train/test split;
               use 42 for the canonical split shared across all experiments

    Output:
        (train, test) as HF Datasets — label 0=not_hate 1=hate, text_col='post'
    """
    raw = load_dataset("tasksource/implicit-hate-stg1", split="train")
    splits = raw.train_test_split(test_size=0.10, seed=seed)  # seed ensures the same rows go to train vs test across all experiments

    def _label(x):
        x["label"] = 0 if x["class"] == "not_hate" else 1
        return x

    return splits["train"].map(_label), splits["test"].map(_label)


def load_ihc_implicit_only(seed: int = 42):
    """
    IHC test split with explicit_hate rows removed.
    Replicates the not_hate vs implicit_hate binary setup from ElSherief et al.

    Input:
        seed : int — controls the train/test split

    Output:
        HF Dataset — label 0=not_hate 1=implicit_hate, text_col='post'
    """
    raw = load_dataset("tasksource/implicit-hate-stg1", split="train")
    test = raw.train_test_split(test_size=0.10, seed=seed)["test"]
    test = test.filter(lambda x: x["class"] != "explicit_hate")  # keep only not_hate and implicit_hate to replicate the binary task from the paper
    return test.map(lambda x: {"label": 0 if x["class"] == "not_hate" else 1})


def load_ihc_chunks(seed: int = 42):
    """
    IHC training split formatted for the chunking pipeline.

    Input:
        seed : int — controls the train/test split

    Output:
        list[{"text": str, "label": str}] — label is "hate" or "not_hate"
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


def load_ishate_binary():
    """
    ISHate binary classification using the hateful_layer column.
    Uses the pre-made train/test splits.

    Output:
        (train, test) as HF Datasets — label 0=Non-HS 1=hate, text_col='text'
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

    Output:
        (train, val, test) as HF Datasets with label_a and label_b columns, text_col='text'
        label_a: 0=Non-HS  1=Explicit HS  2=Implicit HS
        label_b: 0=Non-HS  1=Non-Subtle HS  2=Subtle HS
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

    Output:
        list[{"text": str, "label": str}] — label is "hate" or "not_hate"
    """
    path = hf_hub_download(
        repo_id="BenjaminOcampo/ISHate",
        filename="ishate_train.parquet.gzip",
        repo_type="dataset",  # tells hf_hub_download to look in the datasets namespace, not models
    )
    df = pd.read_parquet(path)
    return [
        {
            "text": row["text"],
            "label": "not_hate" if row["hateful_layer"] == "Non-HS" else "hate",
        }
        for _, row in df.iterrows()
    ]


_VICOMTECH_URL = (
    "https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip"
)


def _load_vicomtech_raw() -> dict:
    """
    Download the Vicomtech ZIP from GitHub into a temp directory, parse both splits,
    and return the rows. The temp directory is deleted automatically on return.

    Output:
        dict with keys "train" and "test", each a list[{"text": str, "label": int}]
    """
    with tempfile.TemporaryDirectory() as tmpdir:  # temp directory is deleted automatically when the with-block exits, even on error
        zip_path = os.path.join(tmpdir, "dataset.zip")
        print("Downloading Vicomtech dataset from GitHub...")
        urllib.request.urlretrieve(_VICOMTECH_URL, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        repo = os.path.join(tmpdir, "hate-speech-dataset-master")
        meta = pd.read_csv(
            os.path.join(repo, "annotations_metadata.csv")
        ).set_index("file_id")  # index by file_id for O(1) lookup instead of filtering rows on every iteration

        result = {}
        for split_key, dir_name in [("train", "sampled_train"), ("test", "sampled_test")]:
            split_dir = os.path.join(repo, dir_name)
            rows = []
            for fname in sorted(os.listdir(split_dir)):
                if not fname.endswith(".txt"):  # skip non-text files (e.g. .DS_Store)
                    continue
                file_id = fname[:-4]  # strip the .txt extension to get the file_id used in the metadata CSV
                if file_id not in meta.index:  # guard against files with no annotation entry
                    continue
                label_str = meta.loc[file_id, "label"]
                if label_str not in ("hate", "noHate"):  # the dataset also contains "relation" labels — skip them for the binary task
                    continue
                with open(os.path.join(split_dir, fname), encoding="utf-8") as f:
                    text = f.read().strip()
                if not text:  # skip empty files
                    continue
                rows.append({"text": text, "label": 1 if label_str == "hate" else 0})
            result[split_key] = rows

        return result


def load_vicomtech(split: Literal["train", "test", "both"] = "both"):
    """
    Vicomtech binary classification. Downloads from GitHub on each call (no local storage).

    Input:
        split : "train" | "test" | "both"

    Output:
        single HF Dataset if split is "train" or "test",
        (train, test) tuple if split is "both" — label 0=not hate 1=hate, text_col='text'
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

    Output:
        list[{"text": str, "label": str}] — label is "hate" or "not_hate"
    """
    raw = _load_vicomtech_raw()
    return [
        {
            "text": row["text"],
            "label": "hate" if row["label"] == 1 else "not_hate",
        }
        for row in raw["train"]
    ]
