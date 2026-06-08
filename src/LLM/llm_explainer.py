from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import faiss
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    from retriever import encode, retrieve_top_k, retrieve_top_k_above_threshold
except ImportError:
    pass  # heavy deps only available in the notebook environment


@dataclass
class Layer2Output:
    """Output contract from the BERT-based RAG classifier (Layer 2)."""
    original_text: str
    label: str                          # "hate" | "not hate"
    confidence: float                   # fraction in [0, 1]
    hate_category: str                  # "racism" | "sexism" | "antisemitism" | "unknown"
    retrieved: List[Tuple[str, float]]  # [(text_with_label_prefix, cosine_score), ...]


@dataclass
class ExplainerOutput:
    """Structured explanation produced by Layer 3 for website moderation."""
    summary: str
    evidence_used: List[int]            # 1-based indices into retrieved passages
    target_groups: List[str]
    severity: str                       # "low" | "medium" | "high"
    recommended_action: str             # "auto-block" | "human-review" | "allow"
    moderator_note: Optional[str]       # populated when action is human-review or validation fails
    validation_passed: bool


_LABEL_RE = re.compile(r"^\[(hate|not hate)\]\s*:?\s*", re.IGNORECASE)  # matches "[hate]" or "[not hate]" at the start of a chunk string, with optional colon and surrounding whitespace


def _strip_label(text: str) -> str:
    return _LABEL_RE.sub("", text).strip()


strip_label = _strip_label  # public alias used by the demo notebook


def _extract_label(text: str) -> str:
    m = _LABEL_RE.match(text)
    return m.group(1).lower() if m else "unknown"  # group(1) captures the label word inside the brackets


RETRIEVER_HF_ID = "sentence-transformers/all-mpnet-base-v2"
CLF_HF_IDS = {
    "bert":    "bert-base-uncased",
    "roberta": "roberta-base",
}


_SYSTEM_PROMPT = (
    "You are a content moderation assistant. Your task is to explain a hate speech "
    "classification decision in clear, professional language for a website moderation team. "
    "You must ground every claim in the provided evidence passages using their index numbers. "
    "Do not introduce information that is absent from the evidence. "
    "Respond with valid JSON only — no markdown, no prose outside the JSON object."
)

_JSON_SCHEMA = """{
  "summary": "<1-2 sentence plain-English explanation of why this content was flagged or allowed>",
  "evidence_used": [<1-based integer indices of passages you relied on, e.g. [1, 3]>],
  "target_groups": ["<group1>", "<group2>"],
  "severity": "<low | medium | high>",
  "recommended_action": "<auto-block | human-review | allow>",
  "moderator_note": "<brief note for the human reviewer, or null if none needed>"
}"""


def build_prompt(
    original_text: str,
    label: str,
    confidence: float,
    hate_category: str,
    retrieved: List[Tuple[str, float]],
) -> str:
    """
    Build the user-turn prompt sent to the LLM.

    Input:
        original_text : str — the post to moderate
        label         : str — "hate" or "not hate"
        confidence    : float — classifier confidence in [0, 1]
        hate_category : str — detected hate category (e.g. "racism")
        retrieved     : list of (text_with_label_prefix, cosine_score) pairs

    Output:
        str — formatted prompt string
    """
    passages_block = "\n".join(
        f"[{i + 1}] \"{_strip_label(text)}\"  (similarity: {score:.4f})"
        for i, (text, score) in enumerate(retrieved)
    )
    return (
        f"TEXT TO MODERATE:\n\"{original_text}\"\n\n"
        f"CLASSIFICATION DECISION:\n"
        f"  label     : {label}\n"
        f"  confidence: {confidence:.2f}\n"
        f"  category  : {hate_category}\n\n"
        f"RETRIEVED EVIDENCE PASSAGES:\n{passages_block}\n\n"
        f"Produce a JSON response matching this exact schema:\n{_JSON_SCHEMA}"
    )


def call_llm(prompt: str, client, model: str) -> dict:
    """
    Call any OpenAI-compatible client and return parsed JSON.
    Falls back to plain completion if json_object mode is unsupported (e.g. Ollama).

    Input:
        prompt : str — formatted user-turn prompt
        client : OpenAI-compatible client instance
        model  : str — model identifier

    Output:
        dict — parsed JSON response from the LLM
    """
    base_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    try:
        response = client.chat.completions.create(
            **base_kwargs, response_format={"type": "json_object"}  # forces the model to output valid JSON; not all backends support this parameter
        )
    except Exception:
        response = client.chat.completions.create(**base_kwargs)  # fallback for backends that don't support response_format (e.g. Ollama)

    content = response.choices[0].message.content.strip()

    # some models wrap their JSON in markdown code fences despite the system prompt; strip them before parsing
    content = re.sub(r"^```(?:json)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content)

    return json.loads(content)


_REQUIRED_FIELDS = {"summary", "evidence_used", "target_groups", "severity", "recommended_action"}
_VALID_SEVERITIES = {"low", "medium", "high"}
_VALID_ACTIONS = {"auto-block", "human-review", "allow"}


def validate_output(raw: dict, n_passages: int) -> bool:
    """
    Check LLM output structure before building an ExplainerOutput.
    A False return triggers a human-review override in explain().

    Input:
        raw        : dict — parsed LLM JSON response
        n_passages : int — number of retrieved passages (upper bound for evidence indices)

    Output:
        bool — True if all required fields are present, non-empty, and within expected ranges
    """
    if not _REQUIRED_FIELDS.issubset(raw.keys()):
        return False
    if not isinstance(raw["evidence_used"], list) or not raw["evidence_used"]:
        return False
    for idx in raw["evidence_used"]:
        if not isinstance(idx, int) or not (1 <= idx <= n_passages):  # indices must be 1-based integers within the actual passage count
            return False
    if raw.get("severity") not in _VALID_SEVERITIES:
        return False
    if raw.get("recommended_action") not in _VALID_ACTIONS:
        return False
    if not raw.get("summary", "").strip():
        return False
    return True


def explain(layer2_output: Layer2Output, client, model: str) -> ExplainerOutput:
    """
    Generate a structured moderation explanation for a Layer 2 classification.
    The LLM does NOT re-classify — it only explains Layer 2's decision.

    Input:
        layer2_output : Layer2Output — classification result from the BERT-RAG layer
        client        : OpenAI-compatible client — Groq, openai.OpenAI, or Ollama-backed
        model         : str — model identifier, e.g. "llama3-8b-8192" or "mistral"

    Output:
        ExplainerOutput — structured explanation; recommended_action is overridden to
        "human-review" and validation_passed is False if LLM output fails validation
    """
    prompt = build_prompt(
        layer2_output.original_text,
        layer2_output.label,
        layer2_output.confidence,
        layer2_output.hate_category,
        layer2_output.retrieved,
    )

    raw: dict = {}  # pre-initialise so the fallback branch can safely call .get() even if call_llm raises before assigning
    valid = False
    try:
        raw = call_llm(prompt, client, model)
        valid = validate_output(raw, len(layer2_output.retrieved))
    except Exception:
        pass  # any LLM or JSON parsing failure falls through to the invalid-output branch below

    if not valid:
        return ExplainerOutput(
            summary=raw.get("summary", "Explanation could not be generated."),
            evidence_used=raw.get("evidence_used", []),
            target_groups=raw.get("target_groups", []),
            severity=raw.get("severity", "unknown"),
            recommended_action="human-review",  # always escalate to human review when the LLM output cannot be trusted
            moderator_note="LLM output failed evidence validation — manual review required.",
            validation_passed=False,
        )

    return ExplainerOutput(
        summary=raw["summary"],
        evidence_used=raw["evidence_used"],
        target_groups=raw["target_groups"],
        severity=raw["severity"],
        recommended_action=raw["recommended_action"],
        moderator_note=raw.get("moderator_note") or None,  # convert empty string "" to None; LLMs sometimes return "" instead of null
        validation_passed=True,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers — used by llm_demo.ipynb
# ---------------------------------------------------------------------------

def load_pipeline(model_family, index_split, dataset, index_dir, weights_rac_dir):
    """Load the SBERT retriever, FAISS index, and RAC classifier.

    Returns (ret_model, ret_tokenizer, index, documents, clf_model, clf_tokenizer, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading retriever: {RETRIEVER_HF_ID} ...")
    ret_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_HF_ID)
    ret_model     = AutoModel.from_pretrained(RETRIEVER_HF_ID).eval().to(device)

    index_path  = Path(index_dir) / f"vdb_{index_split}.faiss"
    lookup_path = Path(index_dir) / f"lookup_{index_split}.json"
    print(f"Loading index: {index_path} ...")
    faiss_index = faiss.read_index(str(index_path))
    with open(lookup_path) as f:
        documents = json.load(f)
    print(f"  Index size: {faiss_index.ntotal:,} vectors")

    clf_path = str(Path(weights_rac_dir) / model_family / "sbert" / index_split / dataset)
    print(f"Loading RAC classifier: {clf_path} ...")
    clf_tokenizer = AutoTokenizer.from_pretrained(clf_path)
    clf_model     = AutoModelForSequenceClassification.from_pretrained(clf_path).eval().to(device)

    print(f"\nReady: {model_family.upper()} | index=sbert/{index_split} | trained_on={dataset}")
    return ret_model, ret_tokenizer, faiss_index, documents, clf_model, clf_tokenizer, device


def retrieve_numpy(text, threshold, k, model, tokenizer, xb, id_map, documents):
    """Encode with mean pooling then score via numpy dot product — no FAISS at query time."""
    vec   = encode([text], model, tokenizer, batch_size=1, use_mean_pool=True)
    vec_n = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)

    sims    = (xb @ vec_n.T).squeeze()
    top_pos = np.argsort(sims)[::-1][:k]
    top_ids = id_map[top_pos]
    scores  = sims[top_pos]

    retrieved = [
        (documents[str(int(cid))], float(sc))
        for cid, sc in zip(top_ids, scores)
        if sc >= threshold
    ][:k]
    if not retrieved:
        retrieved = [
            (documents[str(int(cid))], float(sc))
            for cid, sc in zip(top_ids, scores)
        ][:k]

    del vec, vec_n, sims, top_pos, top_ids, scores
    return retrieved


def run_pipeline(text, ret_model, ret_tokenizer, index, documents,
                 clf_model, clf_tokenizer, device,
                 llm_client, llm_model, k=5, threshold=0.3):
    """Run Layer 2 (retrieval + classification) and Layer 3 (LLM explanation) for one text.

    Returns (Layer2Output, ExplainerOutput).
    """
    retrieved = retrieve_top_k_above_threshold(
        text, threshold, ret_model, ret_tokenizer, index, documents, chunk_id=None, k=k
    )
    if not retrieved:
        retrieved = retrieve_top_k(
            text, ret_model, ret_tokenizer, index, documents, chunk_id=None, k=k
        )

    sep       = clf_tokenizer.sep_token or "[SEP]"
    augmented = f" {sep} ".join([text] + [t for t, _ in retrieved])
    inputs    = clf_tokenizer(
        augmented, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    inputs = {key: v.to(device) for key, v in inputs.items()}
    with torch.no_grad():
        logits = clf_model(**inputs).logits[0]
    probs      = F.softmax(logits, dim=-1)
    pred_idx   = torch.argmax(probs).item()
    label      = "hate" if pred_idx == 1 else "not hate"
    confidence = probs[pred_idx].item()

    l2 = Layer2Output(
        original_text=text, label=label, confidence=confidence,
        hate_category="unknown", retrieved=retrieved,
    )
    explanation = explain(l2, llm_client, llm_model)
    return l2, explanation


def display_result(example_id, text, l2, explanation):
    """Print a formatted single-result block to stdout."""
    w = 80
    print("=" * w)
    print(f"[{example_id}]  TEXT : {text}")
    print("-" * w)
    print(f"LAYER 2  : {l2.label.upper()}  ({l2.confidence:.1%} confidence)")
    print()
    print(f"RETRIEVED NEIGHBORS ({len(l2.retrieved)}):")
    for i, (txt, score) in enumerate(l2.retrieved, 1):
        print(f"  [{i}] {score:.4f}  {_strip_label(txt)[:100]}")
    print()
    print("LAYER 3 EXPLANATION:")
    print(f"  Summary   : {explanation.summary}")
    print(f"  Severity  : {explanation.severity}")
    print(f"  Action    : {explanation.recommended_action}")
    print(f"  Targets   : {', '.join(explanation.target_groups) if explanation.target_groups else chr(8212)}")
    print(f"  Evidence  : {explanation.evidence_used}")
    if explanation.moderator_note:
        print(f"  Note      : {explanation.moderator_note}")
    valid_str = "✓ passed" if explanation.validation_passed else "✗ FAILED (forced human-review)"
    print(f"  Validation: {valid_str}")
    print("=" * w)
    print()


def wrap(text, width=80, indent=14):
    """Wrap text for fixed-width terminal output with hanging indent."""
    return textwrap.fill(str(text), width=width, subsequent_indent=" " * indent)
