"""Dry-run smoke test for the joint MTL data loader.

Validates UD supertagging load + collator + tokenizer using a tiny in-memory
Spanish->Basque translation stub, so we can exercise the full pipeline shape
without downloading the 1.1GB Tatoeba Challenge archive yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data import (
    BasqueUDDataset,
    JointMTLDataset,
    TASK_SUPERTAG,
    TASK_TRANSLATE,
    build_joint_collator,
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SPANISH_BASQUE_STUB = [
    ("Hola, ¿cómo estás?", "Kaixo, zer moduz?"),
    ("Me gusta leer libros.", "Liburuak irakurtzea gustatzen zait."),
    ("El tren sale a las diez.", "Trena hamarretan ateratzen da."),
    ("Tengo hambre.", "Goseak nago."),
    ("Gracias por tu ayuda.", "Eskerrik asko zure laguntzagatik."),
    ("Mañana iremos al mercado.", "Bihar azokara joango gara."),
    ("El niño juega en el parque.", "Umea parkean jolasten ari da."),
    ("No entiendo esta frase.", "Ez dut esaldi hau ulertzen."),
]


class _StubTranslationDataset(Dataset):
    def __init__(self, pairs):
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        src, tgt = self._pairs[idx]
        return {"source": src, "target": tgt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=4)
    args = ap.parse_args()

    supertagging = BasqueUDDataset(
        data_dir=Path(args.data_dir), split="train", fmt="supertag", limit=32
    )
    translation = _StubTranslationDataset(SPANISH_BASQUE_STUB)
    print(f"[info] supertagging={len(supertagging)} translation_stub={len(translation)}")

    joint = JointMTLDataset(translation=translation, supertagging=supertagging, translate_weight=0.5)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = build_joint_collator(tokenizer, max_length=96)
    loader = DataLoader(
        joint, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    counts = {TASK_TRANSLATE: 0, TASK_SUPERTAG: 0}
    for i, batch in enumerate(loader):
        for t in batch["task"]:
            counts[t] += 1
        if i < args.batches:
            print(
                f"\n[batch {i}] tasks={batch['task']} "
                f"input_ids={tuple(batch['input_ids'].shape)} "
                f"labels={tuple(batch['labels'].shape)} "
                f"forced_bos={batch['forced_bos_token_id'].tolist()}"
            )
            first_labels = batch["labels"][0].clone()
            first_labels[first_labels == -100] = tokenizer.pad_token_id
            print(
                "  src[0]:",
                tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)[:140],
            )
            print(
                "  tgt[0]:",
                tokenizer.decode(first_labels, skip_special_tokens=True)[:140],
            )
        if i >= args.batches:
            break
    print(f"\n[info] task mix in sampled batches: {counts}")


if __name__ == "__main__":
    main()
