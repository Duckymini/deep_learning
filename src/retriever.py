import numpy as np
import torch
from tqdm import tqdm
import faiss

BATCH_SIZE = 64
MAX_LENGTH = 64
K = 4


def encode(texts, model, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
           use_mean_pool=False):
    """
    Encode a list of texts into fixed-size float32 vectors.

    Input:
        texts        : list[str] — texts to encode
        model        : HuggingFace model
        tokenizer    : corresponding tokenizer
        batch_size   : number of texts per forward pass
        max_length   : max token length per text
        use_mean_pool: True for SBERT mean pooling, False for CLS token

    Output:
        np.ndarray of shape (N, D), float32
    """
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
        inputs = {k: v.to(device) for k, v in inputs.items()}  # move all input tensors to the same device as the model
        with torch.no_grad():  # disable gradient tracking — inference only, no backprop needed
            outputs = model(**inputs)
        if use_mean_pool:
            hidden = outputs.last_hidden_state                          # (B, T, D)
            mask   = inputs["attention_mask"].unsqueeze(-1).float()     # (B, T, 1) — 1 for real tokens, 0 for padding; unsqueeze adds a dim for broadcasting over D
            vecs   = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # average token embeddings over real tokens only; clamp prevents division by zero on empty sequences
            vecs   = vecs.cpu().contiguous().numpy()  # contiguous() ensures the tensor's memory layout is compatible with numpy
        else:
            vecs = outputs.last_hidden_state[:, 0, :].cpu().contiguous().numpy()  # CLS token sits at position 0 and summarises the whole sequence
        all_vecs.append(vecs)
    return np.vstack(all_vecs).astype("float32")  # FAISS only accepts float32 arrays


def encode_and_search(tweet_text, fetch, model, tokenizer, index, use_mean_pool=False):
    """
    Encode one text and run a FAISS search.

    Input:
        tweet_text   : str — text to encode
        fetch        : int — number of candidates to retrieve from the index
        model        : HuggingFace model
        tokenizer    : corresponding tokenizer
        index        : FAISS IndexIDMap
        use_mean_pool: True for SBERT mean pooling, False for CLS token

    Output:
        (scores, ids) — 1-D arrays of length fetch
    """
    vec = encode([tweet_text], model, tokenizer, batch_size=1, use_mean_pool=use_mean_pool)
    faiss.normalize_L2(vec)  # cosine similarity equals dot product on unit-norm vectors; the index uses inner product (IndexFlatIP)
    scores, ids = index.search(vec, fetch)
    return scores[0], ids[0]  # index.search returns 2-D arrays (one row per query); unwrap the single-query batch dimension


def retrieve_top_k(tweet_text, model, tokenizer, index, documents, chunk_id=None, k=K,
                   use_mean_pool=False):
    """
    Return exactly k nearest neighbors regardless of similarity score.

    Input:
        tweet_text   : str — raw tweet (no label prefix)
        model        : HuggingFace model
        tokenizer    : corresponding tokenizer
        index        : FAISS IndexIDMap
        documents    : dict mapping str(chunk_id) → text
        chunk_id     : int or None — the tweet's own chunk_id to exclude at train time;
                       pass None at test time
        k            : number of neighbors to return
        use_mean_pool: True for SBERT mean pooling

    Output:
        list of (text, score) tuples, length k
    """
    fetch = k + 1 if chunk_id is not None else k  # request one extra so we still have k results after discarding the self-match
    scores, ids = encode_and_search(tweet_text, fetch, model, tokenizer, index, use_mean_pool)

    results = []
    for score, cid in zip(scores, ids):
        if cid == -1:  # FAISS pads with -1 when the index has fewer entries than requested
            continue
        if chunk_id is not None and cid == chunk_id:  # at train time, exclude the query's own chunk to avoid leakage
            continue
        results.append((documents[str(cid)], float(score)))
        if len(results) == k:
            break
    return results


def retrieve_by_threshold(tweet_text, threshold, model, tokenizer, index, documents,
                          chunk_id=None, use_mean_pool=False):
    """
    Return all neighbors with similarity >= threshold (no upper limit on count).

    When chunk_id is given, fetches all index.ntotal vectors to guarantee the
    self-match can be excluded regardless of its rank.

    Input:
        tweet_text   : str — raw tweet (no label prefix)
        threshold    : float — minimum cosine similarity (e.g. 0.95)
        model        : HuggingFace model
        tokenizer    : corresponding tokenizer
        index        : FAISS IndexIDMap
        documents    : dict mapping str(chunk_id) → text
        chunk_id     : int or None — own chunk_id to exclude at train time
        use_mean_pool: True for SBERT mean pooling

    Output:
        list of (text, score) tuples with score >= threshold, ordered by score desc
    """
    fetch = index.ntotal  # fetch every vector so the self-match is guaranteed to appear and can be excluded regardless of its score rank
    scores, ids = encode_and_search(tweet_text, fetch, model, tokenizer, index, use_mean_pool)

    results = []
    for score, cid in zip(scores, ids):
        if cid == -1 or score < threshold:
            break  # FAISS returns results sorted by score descending, so all remaining entries will also be below the threshold
        if chunk_id is not None and cid == chunk_id:
            continue
        results.append((documents[str(cid)], float(score)))
    return results


def retrieve_top_k_above_threshold(tweet_text, threshold, model, tokenizer, index, documents,
                                   chunk_id=None, k=K, use_mean_pool=False):
    """
    Return up to k neighbors with similarity >= threshold.

    Combines quality filtering (threshold) with a quantity cap (k).

    Input:
        tweet_text   : str — raw tweet (no label prefix)
        threshold    : float — minimum cosine similarity (e.g. 0.4)
        model        : HuggingFace model
        tokenizer    : corresponding tokenizer
        index        : FAISS IndexIDMap
        documents    : dict mapping str(chunk_id) → text
        chunk_id     : int or None — own chunk_id to exclude at train time
        k            : maximum number of results
        use_mean_pool: True for SBERT mean pooling

    Output:
        list of (text, score) tuples with score >= threshold, length <= k
    """
    fetch = k + 1 if chunk_id is not None else k  # request one extra so we still have k results after discarding the self-match
    scores, ids = encode_and_search(tweet_text, fetch, model, tokenizer, index, use_mean_pool)

    results = []
    for score, cid in zip(scores, ids):
        if cid == -1 or score < threshold:
            break  # results are sorted descending — safe to stop as soon as we cross below the threshold
        if chunk_id is not None and cid == chunk_id:
            continue
        results.append((documents[str(cid)], float(score)))
        if len(results) == k:
            break
    return results
