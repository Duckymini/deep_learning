import numpy as np
import torch
import tqdm
import faiss

BATCH_SIZE = 64
MAX_LENGTH = 64
K = 4

# --- Encoding functions ---

def encode(texts, model, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_vecs.append(cls_vecs)
    return np.vstack(all_vecs).astype("float32")

def encode_and_search(tweet_text, fetch, model, tokenizer, index):
    vec = encode([tweet_text], model, tokenizer, batch_size=1)
    faiss.normalize_L2(vec)
    scores, ids = index.search(vec, fetch)
    return scores[0], ids[0]

# ---- Retrieving functions ----

def retrieve_top_k(tweet_text, documents, chunk_id=None, k=K):
    """
    Always returns exactly k neighbors.

    tweet_text : raw tweet string (no [hate]/[not hate] prefix).
    chunk_id   : the tweet's own chunk_id (int) to exclude self at train time.
                 Pass None at test time — the tweet is not in the index.
    Returns    : list of k (text, score) tuples.
    """
    fetch = k + 1 if chunk_id is not None else k
    scores, ids = encode_and_search(tweet_text, fetch)

    results = []
    for score, cid in zip(scores, ids):
        if cid == -1:
            continue
        if chunk_id is not None and cid == chunk_id:  # exclude by id, NOT by score
            continue
        results.append((documents[str(cid)], float(score)))
        if len(results) == k:
            break
    return results


def retrieve_by_threshold(tweet_text, threshold, index, documents, chunk_id=None):
    """
    Returns all neighbors whose similarity score is >= threshold, with no upper limit.

    Since k is unbounded, we fetch all vectors in one shot (index.ntotal) when a
    chunk_id is given, so the self-match can always be filtered out regardless of rank.

    tweet_text : raw tweet string (no [hate]/[not hate] prefix).
    threshold  : minimum cosine similarity score (float, e.g. 0.95).
    chunk_id   : the tweet's own chunk_id (int) to exclude self at train time.
                 Pass None at test time.
    Returns    : list of (text, score) tuples with score >= threshold, ordered by score desc.
    """
    fetch = index.ntotal if chunk_id is not None else index.ntotal
    scores, ids = encode_and_search(tweet_text, fetch)

    results = []
    for score, cid in zip(scores, ids):
        if cid == -1 or score < threshold:
            break  # FAISS returns results sorted by score desc — safe to stop early
        if chunk_id is not None and cid == chunk_id:
            continue
        results.append((documents[str(cid)], float(score)))
    return results


def retrieve_top_k_above_threshold(tweet_text, threshold, documents, chunk_id=None, k=K):
    """
    Returns neighbors with score >= threshold, capped at k results.

    Combines the two strategies: quality filter (threshold) + quantity cap (k).
    Useful when you want at most k results but only if they are genuinely similar.

    tweet_text : raw tweet string (no [hate]/[not hate] prefix).
    threshold  : minimum cosine similarity score (float, e.g. 0.95).
    chunk_id   : the tweet's own chunk_id (int) to exclude self at train time.
                 Pass None at test time.
    k          : maximum number of results to return.
    Returns    : list of (text, score) tuples with score >= threshold, len <= k.
    """
    fetch = k + 1 if chunk_id is not None else k
    scores, ids = encode_and_search(tweet_text, fetch)

    results = []
    for score, cid in zip(scores, ids):
        if cid == -1 or score < threshold:
            break
        if chunk_id is not None and cid == chunk_id:
            continue
        results.append((documents[str(cid)], float(score)))
        if len(results) == k:
            break
    return results