"""Tatoeba Translation Challenge: Spanish -> Basque parallel pairs.

Downloads the release tarball from the Tatoeba Challenge CSC Pouta mirror
(the canonical public source used by Tiedemann 2020). Avoids HuggingFace
gated datasets. The pair directory `eus-spa` (ISO codes in alphabetical
order) contains `*.src` = Basque and `*.trg` = Spanish; we swap at load
time to yield Spanish->Basque, matching the translation direction used in
the baseline eval.
"""
from __future__ import annotations

import gzip
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

TATOEBA_RELEASE = "v2023-09-26"
# Tatoeba Challenge orders ISO codes alphabetically in the pair dir name,
# so `eus-spa` is correct (not `spa-eus`). Inside the archive:
#   *.src = Basque (eus, first in name), *.trg = Spanish (spa, second).
# We will swap at load time to yield Spanish->Basque.
PAIR_DIR = "eus-spa"
TATOEBA_URL = (
    "https://object.pouta.csc.fi/Tatoeba-Challenge-"
    f"{TATOEBA_RELEASE}/{PAIR_DIR}.tar"
)


def download_tatoeba_es_eu(data_dir: Path) -> Path:
    out_dir = data_dir / "tatoeba_spa_eus"
    release_dir = out_dir / "data" / "release" / TATOEBA_RELEASE / PAIR_DIR
    if (release_dir / "train.src.gz").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / f"{PAIR_DIR}.tar"
    if not tarball.exists():
        print(f"[info] downloading {TATOEBA_URL}")
        urllib.request.urlretrieve(TATOEBA_URL, tarball)
    print(f"[info] extracting {tarball}")
    with tarfile.open(tarball, "r") as tf:
        tf.extractall(out_dir, filter="data")
    return out_dir


def _find_split_files(out_dir: Path, split: str) -> tuple[Path, Path]:
    """Locate src+trg files (gzipped or plain) for a split under the tarball layout."""
    candidates = [
        out_dir,
        out_dir / "data" / "release" / TATOEBA_RELEASE / PAIR_DIR,
        out_dir / PAIR_DIR,
    ]
    for base in list(candidates):
        if base.exists():
            for sub in base.iterdir():
                if sub.is_dir():
                    candidates.append(sub)
    for base in candidates:
        for ext in (".gz", ""):
            src_file = base / f"{split}.src{ext}"
            trg_file = base / f"{split}.trg{ext}"
            if src_file.exists() and trg_file.exists():
                return src_file, trg_file
    raise FileNotFoundError(
        f"Could not locate {split}.src[.gz]/{split}.trg[.gz] under {out_dir}"
    )


def _read_lines(path: Path, limit: int | None = None) -> list[str]:
    """Stream lines from a file (gzipped if `.gz`), stopping early at `limit`.

    Tatoeba train.src.gz/train.trg.gz are hundreds of MB compressed with tens
    of millions of lines, so a full read is wasteful when sampling a subset.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[str] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            out.append(line.rstrip("\n"))
    return out


@dataclass
class TranslationExample:
    src: str
    tgt: str


class TatoebaEsEuDataset(Dataset):
    """Spanish -> Basque parallel pairs from the Tatoeba Translation Challenge."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        limit: int | None = None,
        max_chars: int = 500,
    ) -> None:
        out_dir = download_tatoeba_es_eu(data_dir)
        src_file, trg_file = _find_split_files(out_dir, split)
        # Dev/test splits ship uncompressed, train splits are gzipped.
        eus_lines = _read_lines(src_file, limit)  # pair dir `eus-spa` -> src=eus
        spa_lines = _read_lines(trg_file, limit)
        assert len(eus_lines) == len(spa_lines), "src/trg must be line-aligned"

        # Swap to Spanish->Basque for training direction alignment with eval.
        pairs = [
            TranslationExample(src=spa, tgt=eus)
            for spa, eus in zip(spa_lines, eus_lines)
            if spa and eus and len(spa) <= max_chars and len(eus) <= max_chars
        ]
        if limit is not None:
            pairs = pairs[:limit]
        self._examples = pairs

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        return {"source": ex.src, "target": ex.tgt}
