"""Drop-in for BasqueUDDataset; reads JSONL from src/spinoff/annotate_tatoeba.py."""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class TatoebaAnnotatedDataset(Dataset):

    def __init__(self, path: Path, limit: int | None = None) -> None:
        path = Path(path)
        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("annotation"):
                    records.append(r)
                if limit is not None and len(records) >= limit:
                    break
        self._records = records
        print(f"[info] TatoebaAnnotatedDataset: {len(records)} pairs from {path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        return {"source": r["tgt_eu"], "target": r["annotation"]}
