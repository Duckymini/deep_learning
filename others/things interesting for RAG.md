Question 2 — A Retriever That Outputs a Variable Number of Results
Yes, this is entirely possible and actually an active research direction. Instead of always returning a fixed top-k, you design the retriever to be adaptive.
Approaches
1. Similarity threshold (simplest)
for each candidate document:
    if similarity_score > threshold τ:
        include it
    else:
        discard
→ output: 0 to N documents depending on what passes
You set τ based on validation performance. Could return 0 documents if nothing is relevant enough.
2. Confidence-based retrieval
The retriever outputs a relevance score per document. You only include documents above a learned or tuned cutoff. This is essentially what rerankers (cross-encoders) already do — they score each candidate and you threshold.
3. Learned adaptive-k
Train a small auxiliary model that takes the query and the score distribution as input and predicts how many documents to retrieve. More complex but fully learnable.
4. Attention-weighted retrieval (softest version)
Don't retrieve discrete documents at all. Instead, compute a weighted sum of all document embeddings, weighted by relevance score. The model effectively "sees" all documents but pays more attention to relevant ones. Output is always one vector — no concatenation problem at all.