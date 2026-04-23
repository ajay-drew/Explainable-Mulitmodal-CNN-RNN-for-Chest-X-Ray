"""
Text encoders for Mode A (RadBERT) and Mode B (TwitterRoBERTa).

Both encoders are FULLY FROZEN — zero gradient updates.
Text in Mode A is used ONLY for XAI attribution, not for predictions.
inputs_embeds support is included for Captum Integrated Gradients.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RadBERT — Mode A (radiology report XAI)
# ─────────────────────────────────────────────────────────────────────────────

class RadBERTEncoder(nn.Module):
    """
    Frozen RadBERT encoder for extracting radiology report features.

    Used ONLY for XAI attribution (SHAP, Integrated Gradients, attention).
    Does NOT contribute to image predictions.

    Args:
        model_name:  HuggingFace model identifier.
        projection_dim: Output projection dimension (default 512).
    """

    def __init__(
        self,
        model_name: str = "StanfordAIMI/RadBERT",
        projection_dim: int = 512,
    ) -> None:
        super().__init__()
        logger.info(f"Loading RadBERT from {model_name} …")
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # typically 768

        # Frozen projection: CLS → 512
        self.projection = nn.Linear(self.hidden_size, projection_dim, bias=False)

        # Freeze everything
        self.freeze()

        logger.info(
            f"RadBERTEncoder ready. hidden_size={self.hidden_size}, "
            f"projection_dim={projection_dim}. "
            f"Trainable params: {self.count_trainable_parameters()} (should be 0)."
        )

    def freeze(self) -> None:
        """Freeze all parameters including the projection layer."""
        for param in self.parameters():
            param.requires_grad = False

    def count_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the word embedding layer (used by Captum IG)."""
        return self.bert.embeddings.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting both token IDs and raw embeddings.

        Args:
            input_ids:      (B, seq_len) LongTensor. Mutually exclusive with
                            inputs_embeds.
            attention_mask: (B, seq_len) LongTensor of 1s and 0s.
            inputs_embeds:  (B, seq_len, hidden_size) float tensor. When
                            provided, input_ids is ignored — required for
                            Captum Integrated Gradients.

        Returns:
            dict:
              'projected'       — (B, 512) frozen projection of CLS token
              'cls_vector'      — (B, 768) raw CLS hidden state
              'attention_weights' — tuple of per-layer attention tensors
        """
        if inputs_embeds is not None:
            outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        elif input_ids is not None:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        cls_vector = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        projected = self.projection(cls_vector)           # (B, 512)

        return {
            "projected": projected,
            "cls_vector": cls_vector,
            "attention_weights": outputs.attentions,      # tuple of (B, heads, seq, seq)
        }

    def forward_for_ig(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simplified forward for Captum IG — returns CLS projected features.
        Accepts raw embeddings so IG can differentiate w.r.t. input space.

        Args:
            inputs_embeds:  (B, seq_len, hidden_size) float tensor.
            attention_mask: (B, seq_len) mask tensor.

        Returns:
            (B, 512) projected CLS features.
        """
        out = self.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out["projected"]


# ─────────────────────────────────────────────────────────────────────────────
# TwitterRoBERTa — Mode B (sentiment classification)
# ─────────────────────────────────────────────────────────────────────────────

class TwitterRoBERTaEncoder(nn.Module):
    """
    Frozen TwitterRoBERTa sentiment classifier.

    Uses cardiffnlp/twitter-roberta-base-sentiment which already has a
    pretrained 3-class head (negative / neutral / positive).
    No head replacement, no fine-tuning, pure inference.

    Neutral predictions are resolved to binary using confidence:
    if max-confidence class is neutral, the binary label is determined
    by comparing negative vs positive logit.

    Args:
        model_name: HuggingFace model identifier.
    """

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ) -> None:
        super().__init__()
        logger.info(f"Loading TwitterRoBERTa from {model_name} …")
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(
            f"TwitterRoBERTaEncoder ready. "
            f"Trainable params: {self.count_trainable_parameters()} (should be 0)."
        )

    def count_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the word embedding layer (used by Captum IG)."""
        return self.model.roberta.embeddings.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning 3-class logits and binary prediction.

        Args:
            input_ids:      (B, seq_len) LongTensor.
            attention_mask: (B, seq_len) LongTensor.
            inputs_embeds:  (B, seq_len, hidden) — used by Captum IG.

        Returns:
            dict:
              'logits'        — (B, 3) raw logits [neg, neu, pos]
              'probs'         — (B, 3) softmax probabilities
              'binary_pred'   — (B,) LongTensor, 0=negative, 1=positive
              'binary_conf'   — (B,) float confidence for binary prediction
        """
        if inputs_embeds is not None:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        elif input_ids is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        logits = outputs.logits  # (B, 3)
        probs = torch.softmax(logits, dim=-1)

        # Binary resolution: negative(0) vs positive(2); neutral(1) → compare neg/pos
        neg_prob = probs[:, 0]
        pos_prob = probs[:, 2]
        binary_pred = (pos_prob >= neg_prob).long()
        binary_conf = torch.where(
            pos_prob >= neg_prob, pos_prob, neg_prob
        )

        return {
            "logits": logits,
            "probs": probs,
            "binary_pred": binary_pred,
            "binary_conf": binary_conf,
            "attention_weights": outputs.attentions,
        }

    def forward_for_ig(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simplified forward for Captum IG — returns full 3-class logits.

        Args:
            inputs_embeds:  (B, seq_len, hidden) float tensor.
            attention_mask: (B, seq_len) mask.

        Returns:
            (B, 3) logits.
        """
        out = self.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out["logits"]


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_radbert(
    model_name: str = "StanfordAIMI/RadBERT",
    device: Optional[torch.device] = None,
) -> RadBERTEncoder:
    """Load RadBERTEncoder on device in eval mode."""
    dev = device or torch.device("cpu")
    enc = RadBERTEncoder(model_name=model_name)
    enc.to(dev)
    enc.eval()
    return enc


def load_twitter_roberta(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    device: Optional[torch.device] = None,
) -> TwitterRoBERTaEncoder:
    """Load TwitterRoBERTaEncoder on device in eval mode."""
    dev = device or torch.device("cpu")
    enc = TwitterRoBERTaEncoder(model_name=model_name)
    enc.to(dev)
    enc.eval()
    return enc
