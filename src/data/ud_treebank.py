"""Basque Universal Dependency Treebank (UD_Basque-BDT) loader.

Produces seq2seq-style (input_text, linearized_parse_target) pairs for MTL with NLLB.
Downloads the treebank from the UniversalDependencies GitHub mirror if missing.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import conllu
from torch.utils.data import Dataset

UD_BASQUE_BDT_BASE = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Basque-BDT/master"
)
UD_FILES = {
    "train": "eu_bdt-ud-train.conllu",
    "dev": "eu_bdt-ud-dev.conllu",
    "test": "eu_bdt-ud-test.conllu",
}

ParseFormat = Literal["pos", "deprel", "pos+deprel"]


def download_ud_basque_bdt(data_dir: Path) -> Path:
    out_dir = data_dir / "ud_basque_bdt"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in UD_FILES.values():
        target = out_dir / fname
        if target.exists():
            continue
        url = f"{UD_BASQUE_BDT_BASE}/{fname}"
        print(f"[info] downloading {url}")
        urllib.request.urlretrieve(url, target)
    return out_dir


def _linearize(tokens: list[dict], fmt: ParseFormat) -> str:
    """Turn a CONLLU sentence's token list into a target string for seq2seq."""
    parts: list[str] = []
    for tok in tokens:
        if isinstance(tok["id"], tuple):
            # skip multiword ranges like "1-2"
            continue
        form = tok["form"]
        upos = tok.get("upos") or "_"
        deprel = tok.get("deprel") or "_"
        if fmt == "pos":
            parts.append(f"{form}/{upos}")
        elif fmt == "deprel":
            parts.append(f"{form}/{deprel}")
        else:  # pos+deprel
            parts.append(f"{form}/{upos}/{deprel}")
    return " ".join(parts)


@dataclass
class ParseExample:
    text: str
    target: str


class BasqueUDDataset(Dataset):
    """Basque UD Treebank formatted as seq2seq parsing pairs.

    Each item yields {"source": sentence_text, "target": linearized_parse}.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        fmt: ParseFormat = "pos+deprel",
        limit: int | None = None,
    ) -> None:
        assert split in UD_FILES, f"unknown split: {split}"
        root = download_ud_basque_bdt(data_dir)
        conllu_path = root / UD_FILES[split]
        raw = conllu_path.read_text(encoding="utf-8")
        sentences = conllu.parse(raw)

        self._examples: list[ParseExample] = []
        for sent in sentences:
            text = sent.metadata.get("text")
            if not text:
                text = " ".join(
                    t["form"] for t in sent if not isinstance(t["id"], tuple)
                )
            target = _linearize(list(sent), fmt)
            self._examples.append(ParseExample(text=text, target=target))

        if limit is not None:
            self._examples = self._examples[:limit]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        return {"source": ex.text, "target": ex.target}

    def iter_examples(self) -> Iterable[ParseExample]:
        return iter(self._examples)
