"""
Layer 3: LLM Explanation Layer for the hate speech detection pipeline.

Receives Layer 2's classification decision (label, confidence, retrieved passages)
and produces a structured, auditable explanation suited for website moderation.

The LLM does NOT re-classify — that is Layer 2's job. It only explains.

Supported backends (all free or cheap):
  - Groq API  : free tier, fast. pip install groq + GROQ_API_KEY env var.
  - Ollama    : fully local. Install Ollama, run `ollama pull mistral`.
  - OpenAI    : paid (~$0.15/1M tokens). OPENAI_API_KEY env var.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Label prefix utilities
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^\[(hate|not hate)\]\s*:?\s*", re.IGNORECASE)


def _strip_label(text: str) -> str:
    return _LABEL_RE.sub("", text).strip()


def _extract_label(text: str) -> str:
    m = _LABEL_RE.match(text)
    return m.group(1).lower() if m else "unknown"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str, client, model: str) -> dict:
    """
    Call any OpenAI-compatible client and return parsed JSON.
    Falls back to plain completion if json_object mode is unsupported (e.g. Ollama).
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
            **base_kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        response = client.chat.completions.create(**base_kwargs)

    content = response.choices[0].message.content.strip()

    # Strip markdown code fences that some models add despite instructions
    content = re.sub(r"^```(?:json)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content)

    return json.loads(content)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"summary", "evidence_used", "target_groups", "severity", "recommended_action"}
_VALID_SEVERITIES = {"low", "medium", "high"}
_VALID_ACTIONS = {"auto-block", "human-review", "allow"}


def validate_output(raw: dict, n_passages: int) -> bool:
    """
    Returns True only if:
    - all required fields are present and non-empty
    - evidence_used contains only integers in [1, n_passages]
    - severity and recommended_action are recognized values

    A False return triggers a human-review override in explain().
    """
    if not _REQUIRED_FIELDS.issubset(raw.keys()):
        return False
    if not isinstance(raw["evidence_used"], list) or not raw["evidence_used"]:
        return False
    for idx in raw["evidence_used"]:
        if not isinstance(idx, int) or not (1 <= idx <= n_passages):
            return False
    if raw.get("severity") not in _VALID_SEVERITIES:
        return False
    if raw.get("recommended_action") not in _VALID_ACTIONS:
        return False
    if not raw.get("summary", "").strip():
        return False
    return True


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def explain(layer2_output: Layer2Output, client, model: str) -> ExplainerOutput:
    """
    Generate a structured moderation explanation for a Layer 2 classification.

    Parameters
    ----------
    layer2_output : Layer2Output
        Classification result from the BERT-RAG layer.
    client :
        OpenAI-compatible client — Groq, openai.OpenAI, or Ollama-backed.
    model : str
        Model identifier, e.g. "llama3-8b-8192" (Groq) or "mistral" (Ollama).

    Returns
    -------
    ExplainerOutput
        Structured explanation. recommended_action is overridden to "human-review"
        and validation_passed is False if the LLM output fails validation.
    """
    prompt = build_prompt(
        layer2_output.original_text,
        layer2_output.label,
        layer2_output.confidence,
        layer2_output.hate_category,
        layer2_output.retrieved,
    )

    raw: dict = {}
    valid = False
    try:
        raw = call_llm(prompt, client, model)
        valid = validate_output(raw, len(layer2_output.retrieved))
    except Exception:
        pass

    if not valid:
        return ExplainerOutput(
            summary=raw.get("summary", "Explanation could not be generated."),
            evidence_used=raw.get("evidence_used", []),
            target_groups=raw.get("target_groups", []),
            severity=raw.get("severity", "unknown"),
            recommended_action="human-review",
            moderator_note="LLM output failed evidence validation — manual review required.",
            validation_passed=False,
        )

    return ExplainerOutput(
        summary=raw["summary"],
        evidence_used=raw["evidence_used"],
        target_groups=raw["target_groups"],
        severity=raw["severity"],
        recommended_action=raw["recommended_action"],
        moderator_note=raw.get("moderator_note") or None,
        validation_passed=True,
    )


# ---------------------------------------------------------------------------
# KNN Layer 2 — lightweight majority-vote alternative to the full RAG classifier
# ---------------------------------------------------------------------------

def knn_layer2(
    original_text: str,
    retrieved: List[Tuple[str, float]],
    hate_category: str = "unknown",
) -> Layer2Output:
    """
    Classifies via majority vote on the [hate]/[not hate] prefixes of retrieved neighbors.
    Confidence = fraction of neighbors that agree with the majority label.

    Simpler and faster than the full RAG classifier — useful for quick tests,
    ablation studies, or as a KNN baseline to compare against the trained model.
    """
    labels = [_extract_label(text) for text, _ in retrieved]
    hate_count = labels.count("hate")
    majority_label = "hate" if hate_count >= len(labels) / 2 else "not hate"
    agreement = hate_count if majority_label == "hate" else (len(labels) - hate_count)
    confidence = agreement / len(labels) if labels else 0.0

    return Layer2Output(
        original_text=original_text,
        label=majority_label,
        confidence=confidence,
        hate_category=hate_category,
        retrieved=retrieved,
    )


# ---------------------------------------------------------------------------
# Quick demo — run as: python layer3_explainer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Pick your backend ---
    # Option A: Groq (free, recommended) — set GROQ_API_KEY in your environment
    # Option B: Ollama (local, free)     — run `ollama serve` and `ollama pull mistral`
    # Option C: OpenAI (paid)            — set OPENAI_API_KEY in your environment

    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        model = "llama3-8b-8192"
        print("Backend: Groq (llama3-8b-8192)")
    elif openai_key:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        model = "gpt-4o-mini"
        print("Backend: OpenAI (gpt-4o-mini)")
    else:
        import openai
        client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model = "mistral"
        print("Backend: Ollama (mistral) — make sure `ollama serve` is running")

    # --- Example 1: Implicit hate speech (great replacement dog-whistle) ---
    text1 = "These people are replacing us, look at the crime statistics"
    retrieved1 = [
        ("[hate] great replacement theory promotes the idea that white populations are being replaced by immigrants", 0.984),
        ("[hate] crime statistics are frequently misused to spread racist narratives targeting immigrant communities", 0.981),
        ("[not hate] immigration policy is a legitimate subject of democratic debate", 0.963),
    ]
    l2 = knn_layer2(text1, retrieved1, hate_category="racism")
    print(f"\n--- Example 1 ---")
    print(f"Text: {text1}")
    print(f"Layer 2: {l2.label} ({l2.confidence:.0%} confidence)")
    result = explain(l2, client, model)
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))

    # --- Example 2: Counter-speech (should route to human-review or allow)
    text2 = "Stop spreading hatred against immigrants, they are humans too"
    retrieved2 = [
        ("[not hate] counter-speech directly challenges hate speech and promotes inclusion", 0.977),
        ("[not hate] expressions of solidarity with marginalized groups are protected discourse", 0.971),
        ("[hate] immigrants are used as scapegoats in racist political rhetoric", 0.955),
    ]
    l2b = knn_layer2(text2, retrieved2)
    print(f"\n--- Example 2 (counter-speech) ---")
    print(f"Text: {text2}")
    print(f"Layer 2: {l2b.label} ({l2b.confidence:.0%} confidence)")
    result2 = explain(l2b, client, model)
    print(json.dumps(result2.__dict__, indent=2, ensure_ascii=False))
