"""
Text attribution methods for radiology reports and tweets.

Three methods share a common interface:
    explain(text, tokenizer, model, target_class) → List[Tuple[str, float]]

Methods:
    SHAPAttributor          — token masking via SHAP
    IntegratedGradientsAttributor — Captum IG with inputs_embeds
    LIMEAttributor          — LIME with [MASK] token perturbations
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str, tokenizer: AutoTokenizer, max_length: int = 128) -> dict:
    """Tokenize text, return dict with input_ids and attention_mask tensors."""
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc


def _tokens_from_ids(input_ids: torch.Tensor, tokenizer: AutoTokenizer) -> List[str]:
    """Convert input_ids to readable token strings."""
    return tokenizer.convert_ids_to_tokens(input_ids[0].tolist())


def _get_logit_for_class(
    model_forward: Callable,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> float:
    """Run model forward and return scalar logit for target_class."""
    ids = input_ids.to(device)
    mask = attention_mask.to(device)
    with torch.no_grad():
        out = model_forward(input_ids=ids, attention_mask=mask)
    if isinstance(out, dict):
        logits = out.get("logits", out.get("binary_conf"))
    else:
        logits = out
    if logits.ndim == 1:
        return logits[target_class].item()
    return logits[0, target_class].item()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP — token masking
# ─────────────────────────────────────────────────────────────────────────────

class SHAPAttributor:
    """
    Token-level SHAP attribution via brute-force masking.

    For each non-special token, replaces it with the tokenizer's [MASK] (or
    <mask>) token and measures the change in target class logit. Attribution
    score = original_logit - masked_logit.

    Args:
        model:       Text encoder (RadBERTEncoder or TwitterRoBERTaEncoder).
        tokenizer:   Matching HuggingFace tokenizer.
        device:      Torch device.
        max_length:  Tokenizer max length.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.max_length = max_length

        # Determine mask token ID
        mask_token = getattr(tokenizer, "mask_token", "[MASK]")
        self.mask_id: int = tokenizer.convert_tokens_to_ids(mask_token)
        if self.mask_id == tokenizer.unk_token_id:
            logger.warning(
                "Tokenizer does not have a [MASK] token; using UNK as substitute."
            )

    def explain(
        self,
        text: str,
        target_class: int = 0,
    ) -> List[Tuple[str, float]]:
        """
        Compute SHAP-style token attributions.

        Args:
            text:         Input text string.
            target_class: Class index to attribute.

        Returns:
            List of (token_string, attribution_score) sorted by |score| descending.
        """
        enc = _tokenize(text, self.tokenizer, self.max_length)
        input_ids = enc["input_ids"]       # (1, seq)
        attention_mask = enc["attention_mask"]

        baseline_logit = _get_logit_for_class(
            self.model, input_ids, attention_mask, target_class, self.device
        )

        tokens = _tokens_from_ids(input_ids, self.tokenizer)
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        attributions: List[Tuple[str, float]] = []
        for i, token in enumerate(tokens):
            tid = input_ids[0, i].item()
            if tid in special_ids or tid == 0:
                continue

            masked = input_ids.clone()
            masked[0, i] = self.mask_id
            masked_logit = _get_logit_for_class(
                self.model, masked, attention_mask, target_class, self.device
            )
            score = baseline_logit - masked_logit
            attributions.append((token, float(score)))

        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return attributions


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients — Captum
# ─────────────────────────────────────────────────────────────────────────────

class IntegratedGradientsAttributor:
    """
    Integrated Gradients token attribution using Captum.

    Routes through the model's inputs_embeds pathway so gradients flow
    w.r.t. the embedding space. Baseline is the zero embedding tensor.

    Args:
        model:       Text encoder with forward_for_ig(inputs_embeds, mask) method.
        tokenizer:   Matching tokenizer.
        device:      Torch device.
        n_steps:     IG approximation steps (default 50).
        max_length:  Tokenizer max length.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        n_steps: int = 50,
        max_length: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.n_steps = n_steps
        self.max_length = max_length

    def explain(
        self,
        text: str,
        target_class: int = 0,
    ) -> List[Tuple[str, float]]:
        """
        Compute Integrated Gradients token attributions.

        Args:
            text:         Input text string.
            target_class: Class index to attribute (used as output index).

        Returns:
            List of (token_string, attribution_score) sorted by |score| descending.
        """
        from captum.attr import IntegratedGradients

        enc = _tokenize(text, self.tokenizer, self.max_length)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            embeddings = embed_layer(input_ids)  # (1, seq, hidden)

        baseline = torch.zeros_like(embeddings)

        # Wrapper: embeddings → scalar logit for target_class
        def forward_fn(embeds: torch.Tensor) -> torch.Tensor:
            logits = self.model.forward_for_ig(embeds, attention_mask)
            if logits.ndim == 1:
                return logits[target_class].unsqueeze(0)
            return logits[:, target_class]

        ig = IntegratedGradients(forward_fn)
        # Gradients must flow through embeddings
        embeddings_input = embeddings.requires_grad_(True)
        attrs, _ = ig.attribute(
            embeddings_input,
            baselines=baseline,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        # attrs: (1, seq, hidden)
        # Aggregate across embedding dimension: L2 norm
        token_scores = attrs.norm(dim=-1).squeeze(0)  # (seq,)
        token_scores = token_scores.detach().cpu().numpy()

        tokens = _tokens_from_ids(input_ids.cpu(), self.tokenizer)
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        attributions: List[Tuple[str, float]] = []
        for i, token in enumerate(tokens):
            tid = input_ids[0, i].item()
            if tid in special_ids or tid == 0:
                continue
            attributions.append((token, float(token_scores[i])))

        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return attributions


# ─────────────────────────────────────────────────────────────────────────────
# LIME — local linear model
# ─────────────────────────────────────────────────────────────────────────────

class LIMEAttributor:
    """
    LIME text attributor using lime.lime_text.LimeTextExplainer.

    Perturbs the input by masking random subsets of words with [MASK]
    and fits a local linear model to predict the target class logit.

    Args:
        model:       Text encoder.
        tokenizer:   Matching tokenizer.
        device:      Torch device.
        n_samples:   Number of LIME perturbation samples.
        max_length:  Tokenizer max length.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
        n_samples: int = 100,
        max_length: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.n_samples = n_samples
        self.max_length = max_length

        mask_token = getattr(tokenizer, "mask_token", "[MASK]")
        self.mask_token_str = mask_token

    def _predict_fn(self, texts: list, target_class: int) -> np.ndarray:
        """
        Batch prediction function for LIME.
        Replaces masked-out words with the mask token and runs inference.

        Args:
            texts:        List of perturbed text strings.
            target_class: Class to score.

        Returns:
            2D numpy array of shape (n_texts, 2) — [1-prob, prob] for binary.
        """
        results = []
        for text in texts:
            enc = _tokenize(text, self.tokenizer, self.max_length)
            logit = _get_logit_for_class(
                self.model,
                enc["input_ids"],
                enc["attention_mask"],
                target_class,
                self.device,
            )
            prob = float(torch.sigmoid(torch.tensor(logit)))
            results.append([1 - prob, prob])
        return np.array(results)

    def explain(
        self,
        text: str,
        target_class: int = 0,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Compute LIME token attributions.

        Args:
            text:         Input text string.
            target_class: Class index to attribute.
            top_n:        Maximum number of tokens to return.

        Returns:
            List of (word, attribution_score) sorted by |score| descending.
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError as exc:
            raise ImportError(
                "LIME is required: pip install lime"
            ) from exc

        explainer = LimeTextExplainer(class_names=["neg", "pos"])

        def predict_wrapper(texts: list) -> np.ndarray:
            return self._predict_fn(texts, target_class)

        explanation = explainer.explain_instance(
            text,
            predict_wrapper,
            num_features=top_n,
            num_samples=self.n_samples,
            labels=[1],
        )

        raw = explanation.as_list(label=1)
        # raw: list of (word, weight)
        raw.sort(key=lambda x: abs(x[1]), reverse=True)
        return [(word, float(weight)) for word, weight in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Attention weight extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    tokenizer: AutoTokenizer,
) -> dict:
    """
    Extract last-layer attention weights averaged across all heads.

    Args:
        model:          RadBERTEncoder or TwitterRoBERTaEncoder.
        input_ids:      (1, seq) LongTensor.
        attention_mask: (1, seq) LongTensor.
        device:         Torch device.
        tokenizer:      Matching tokenizer.

    Returns:
        Dict mapping token_string → average attention weight (float).
    """
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )

    attention_weights = out.get("attention_weights")
    if attention_weights is None or len(attention_weights) == 0:
        return {}

    # Take last layer: (1, n_heads, seq, seq)
    last_layer = attention_weights[-1].squeeze(0)  # (heads, seq, seq)
    # Average across heads and take CLS → all tokens
    avg_attn = last_layer.mean(dim=0)[0]  # (seq,)
    avg_attn = avg_attn.cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }

    result = {}
    for i, token in enumerate(tokens):
        if input_ids[0, i].item() in special_ids:
            continue
        result[token] = float(avg_attn[i])

    return result
