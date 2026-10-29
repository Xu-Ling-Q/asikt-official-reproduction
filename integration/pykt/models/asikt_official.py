"""Bridge the released ASIKT source into pyKT's native data protocol.

The model, attention, and Mamba-2 modules under ``asikt_official_source`` are
byte-for-byte copies of the ASIKT release. This file only adapts pyKT tensors
to the released constructor and forward signature.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .asikt import build_difficulty_masks


SOURCE_DIR = Path(__file__).with_name("asikt_official_source")
SOURCE_FILES = (
    "ASIKT.py",
    "data_pre.py",
    "LICENSE",
    "main.py",
    "mamba2.py",
    "README.md",
    "transformer.py",
    "run.py",
)
_OFFICIAL_MODULES = None


def _load_official_modules():
    """Load the release under its original absolute imports, then restore sys.modules."""
    global _OFFICIAL_MODULES
    if _OFFICIAL_MODULES is not None:
        return _OFFICIAL_MODULES

    module_names = ("run", "mamba2", "transformer", "ASIKT")
    saved = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    source = str(SOURCE_DIR.resolve())
    sys.path.insert(0, source)
    try:
        importlib.invalidate_caches()
        asikt_module = importlib.import_module("ASIKT")
        transformer_module = importlib.import_module("transformer")
        mamba_module = importlib.import_module("mamba2")
        run_module = importlib.import_module("run")
        official_modules = {
            name: sys.modules[name] for name in module_names
        }
    finally:
        try:
            sys.path.remove(source)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    _OFFICIAL_MODULES = SimpleNamespace(
        ASIKT=asikt_module.ASIKT,
        TransformerLayer=transformer_module.TransformerLayer,
        Mamba2=mamba_module.Mamba2,
        Mamba2Config=mamba_module.Mamba2Config,
        modules=official_modules,
    )
    return _OFFICIAL_MODULES


def official_source_hashes():
    """Return SHA256 values for the preserved release model files."""
    result = {}
    for name in SOURCE_FILES:
        path = SOURCE_DIR / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class ASIKTOfficial(nn.Module):
    """pyKT adapter for the unmodified official ASIKT model."""

    def __init__(
        self,
        num_concepts,
        num_questions,
        priors,
        d_model=128,
        dropout=0.2,
        n_blocks=2,
        rasch=1,
        final_fc_dim=512,
        n_heads=8,
        d_ff=1024,
        theta=0.5,
        emb_type="qid",
    ):
        super().__init__()
        official = _load_official_modules()
        self.model_name = "asikt_official"
        self.emb_type = emb_type
        self.num_q = int(num_questions)
        self.num_c = int(num_concepts)
        self.theta = float(theta)

        self.model = official.ASIKT(
            n_kc=self.num_c,
            n_q=self.num_q,
            d_model=int(d_model),
            dropout=float(dropout),
            n_blocks=int(n_blocks),
            rasch=int(rasch),
            final_fc_dim=int(final_fc_dim),
            n_heads=int(n_heads),
            d_ff=int(d_ff),
        )

        self.register_buffer(
            "question_bacc", priors["question_bacc"].float()
        )
        self.register_buffer(
            "question_frequency",
            priors["question_frequency"].float(),
        )
        self.register_buffer(
            "concept_bacc", priors["concept_bacc"].float()
        )
        self.register_buffer(
            "concept_frequency", priors["concept_frequency"].float()
        )

    def _prepare(self, data):
        device = self.question_bacc.device
        q = torch.cat(
            [data["qseqs"][:, :1], data["shft_qseqs"]], dim=1
        ).to(device)
        c = torch.cat(
            [data["cseqs"][:, :1], data["shft_cseqs"]], dim=1
        ).to(device)
        r = torch.cat(
            [data["rseqs"][:, :1], data["shft_rseqs"]], dim=1
        ).to(device)
        valid = torch.cat(
            [data["validseqs"][:, :1], data["shft_validseqs"]], dim=1
        ).to(device).bool()

        # The released loader reserves index zero for padding.
        q = torch.where(valid, q.long() + 1, torch.zeros_like(q.long()))
        c = torch.where(valid, c.long() + 1, torch.zeros_like(c.long()))
        r = torch.where(valid, r.long(), torch.zeros_like(r.long()))
        return q, c, r, valid

    def forward(self, data, return_loss=False):
        q, c, r, valid = self._prepare(data)
        question_mask, interaction_mask = build_difficulty_masks(
            q,
            c,
            r,
            valid,
            self.question_bacc,
            self.question_frequency,
            self.concept_bacc,
            self.concept_frequency,
            self.theta,
        )

        # This target shift is copied from run.py in the official release.
        target_current = torch.where(
            c > 0, r, torch.full_like(r, -1)
        )
        target_shifted = torch.zeros_like(target_current)
        target_shifted[:, :-1] = target_current[:, 1:]
        target_shifted[:, -1] = -1

        loss_sum, flat_predictions, _ = self.model(
            q,
            c,
            r,
            target_shifted,
            question_mask,
            interaction_mask,
        )
        predictions = flat_predictions.reshape(q.shape[0], q.shape[1])
        aligned_predictions = predictions[:, :-1]
        if return_loss:
            return aligned_predictions, loss_sum
        return aligned_predictions
