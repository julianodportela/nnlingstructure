"""Joint MTL dataset: translation + syntactic parsing.

Provides a PyTorch-compatible interleaved iterator over two datasets (translation
and parsing), each example tagged with its task. A collator tokenises using the
NLLB tokenizer and sets the correct forced_bos / src_lang for each task.

For the parsing task we keep source and target language both as Basque
(`eus_Latn`) so that the encoder sees Basque input and the decoder generates
Basque-annotated output — NLLB has no dedicated "parse" language code, so we
reuse the target language of the treebank.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import Dataset

TASK_TRANSLATE = "translate"
TASK_PARSE = "parse"

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
    """Interleave parallel translation pairs with parsing pairs.

    Sampling is weighted: at each index we pick from the translation or parsing
    pool with probability proportional to `translate_weight` / (1 - weight).
    This gives a deterministic virtual length equal to the sum of both pools,
    while letting the caller control the task mix via the weight.
    """

    def __init__(
        self,
        translation: Dataset,
        parsing: Dataset,
        translate_weight: float = 0.8,
        seed: int = 0,
    ) -> None:
        assert 0.0 < translate_weight < 1.0
        self.translation = translation
        self.parsing = parsing
        self.translate_weight = translate_weight
        self._rng = random.Random(seed)
        self._virtual_len = len(translation) + len(parsing)

        # Pre-compute a task schedule for reproducibility across epochs.
        self._schedule: list[tuple[str, int]] = []
        t_idx = p_idx = 0
        for _ in range(self._virtual_len):
            pick_translate = self._rng.random() < translate_weight
            if pick_translate and t_idx < len(translation):
                self._schedule.append((TASK_TRANSLATE, t_idx % len(translation)))
                t_idx += 1
            elif not pick_translate and p_idx < len(parsing):
                self._schedule.append((TASK_PARSE, p_idx % len(parsing)))
                p_idx += 1
            elif t_idx < len(translation):
                self._schedule.append((TASK_TRANSLATE, t_idx % len(translation)))
                t_idx += 1
            else:
                self._schedule.append((TASK_PARSE, p_idx % len(parsing)))
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
        row = self.parsing[inner]
        return JointExample(
            source=row["source"], target=row["target"], task=TASK_PARSE,
            src_lang=EUS, tgt_lang=EUS,
        )


def build_joint_collator(tokenizer, max_length: int = 256):
    """Return a collator that tokenises a batch of JointExamples.

    Groups examples by src_lang so the tokenizer's src_lang / forced_bos is set
    correctly per sub-batch. The collator returns a single padded tensor dict
    spanning the whole batch, with an extra `task` list for loss routing.
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
        # Partition by source language (parse=eus, translate=spa).
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

        # Pad groups to the max shape across groups.
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
