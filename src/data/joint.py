"""Joint MTL dataset and NLLB-aware collator for translation + supertagging.

For supertagging, both src_lang and tgt_lang are eus_Latn — NLLB has no
supertag language code, so we reuse the treebank's target language.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import Dataset

TASK_TRANSLATE = "translate"
TASK_SUPERTAG = "supertag"

SPA = "spa_Latn"
EUS = "eus_Latn"


@dataclass
class JointExample:
    source: str
    target: str
    task: str
    src_lang: str
    tgt_lang: str


class JointMTLDataset(Dataset):
    """Deterministically interleaves translation and supertagging examples at a configurable ratio."""

    def __init__(
        self,
        translation: Dataset,
        supertagging: Dataset,
        translate_weight: float = 0.8,
        seed: int = 0,
    ) -> None:
        assert 0.0 < translate_weight < 1.0
        self.translation = translation
        self.supertagging = supertagging
        self.translate_weight = translate_weight
        self._rng = random.Random(seed)
        self._virtual_len = len(translation) + len(supertagging)

        # Pre-compute a task schedule for reproducibility across epochs.
        self._schedule: list[tuple[str, int]] = []
        t_idx = p_idx = 0
        for _ in range(self._virtual_len):
            pick_translate = self._rng.random() < translate_weight
            if pick_translate and t_idx < len(translation):
                self._schedule.append((TASK_TRANSLATE, t_idx % len(translation)))
                t_idx += 1
            elif not pick_translate and p_idx < len(supertagging):
                self._schedule.append((TASK_SUPERTAG, p_idx % len(supertagging)))
                p_idx += 1
            elif t_idx < len(translation):
                self._schedule.append((TASK_TRANSLATE, t_idx % len(translation)))
                t_idx += 1
            else:
                self._schedule.append((TASK_SUPERTAG, p_idx % len(supertagging)))
                p_idx += 1

    def __len__(self) -> int:
        return self._virtual_len

    def __getitem__(self, idx: int) -> JointExample:
        task, inner = self._schedule[idx]
        if task == TASK_TRANSLATE:
            row = self.translation[inner]
            return JointExample(
                source=row["source"], target=row["target"], task=TASK_TRANSLATE,
                src_lang=SPA, tgt_lang=EUS,
            )
        row = self.supertagging[inner]
        return JointExample(
            source=row["source"], target=row["target"], task=TASK_SUPERTAG,
            src_lang=EUS, tgt_lang=EUS,
        )


def model_inputs(batch: dict) -> dict:
    """Strip collator-only keys (task, forced_bos_token_id) before model(**batch)."""
    return {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}


def build_joint_collator(tokenizer, max_length: int = 256):
    """NLLB-aware collator: groups by src_lang before tokenizing, returns input_ids /
    attention_mask / labels plus task and forced_bos_token_id for training-loop use.
    """

    pad_id = tokenizer.pad_token_id

    def _tokenise_group(examples: list[JointExample], src_lang: str) -> dict:
        tokenizer.src_lang = src_lang
        sources = [ex.source for ex in examples]
        targets = [ex.target for ex in examples]
        enc = tokenizer(
            sources, text_target=targets, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        return enc

    def collate(batch: list[JointExample]) -> dict:
        # Partition by source language (supertag=eus, translate=spa).
        groups: dict[str, list[JointExample]] = {}
        for ex in batch:
            groups.setdefault(ex.src_lang, []).append(ex)

        input_ids_parts = []
        attn_parts = []
        labels_parts = []
        tasks: list[str] = []
        forced_bos: list[int] = []

        for src_lang, exs in groups.items():
            enc = _tokenise_group(exs, src_lang)
            input_ids_parts.append(enc["input_ids"])
            attn_parts.append(enc["attention_mask"])
            labels_parts.append(enc["labels"])
            for ex in exs:
                tasks.append(ex.task)
                forced_bos.append(tokenizer.convert_tokens_to_ids(ex.tgt_lang))

        def _pad(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
            max_len = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[1] < max_len:
                    pad = torch.full(
                        (t.shape[0], max_len - t.shape[1]), pad_value, dtype=t.dtype
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            return torch.cat(padded, dim=0)

        input_ids = _pad(input_ids_parts, pad_id)
        attention_mask = _pad(attn_parts, 0)
        # HuggingFace convention: labels use -100 for ignored (padding) tokens.
        labels = _pad(labels_parts, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task": tasks,
            "forced_bos_token_id": torch.tensor(forced_bos, dtype=torch.long),
        }

    return collate


def infinite_iter(loader) -> Iterator:
    """Yield batches from a DataLoader forever (for step-based training loops)."""
    while True:
        for batch in loader:
            yield batch
