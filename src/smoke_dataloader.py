"""Smoke test for the joint MTL data loader.

Downloads UD_Basque-BDT + Tatoeba Es-Eu, constructs a JointMTLDataset, wraps it
in a DataLoader with the NLLB tokenizer collator, and prints a few batches to
confirm end-to-end shape. The UD dataset is loaded with fmt="supertag" (UPOS +
FEATS per token), which is the auxiliary task used for MTL fine-tuning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import (
    BasqueUDDataset,
    JointMTLDataset,
    TASK_SUPERTAG,
    TASK_TRANSLATE,
    TatoebaEsEuDataset,
    build_joint_collator,
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tl-limit", type=int, default=256, help="Tatoeba rows to load for smoke")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=3)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print("[info] loading Basque UD (train split)…")
    supertagging = BasqueUDDataset(data_dir=data_dir, split="train", fmt="supertag", limit=128)
    print(f"[info] supertagging examples: {len(supertagging)}")

    print("[info] loading Tatoeba spa-eus (train split)…")
    translation = TatoebaEsEuDataset(data_dir=data_dir, split="train", limit=args.tl_limit)
    print(f"[info] translation examples: {len(translation)}")

    joint = JointMTLDataset(translation=translation, supertagging=supertagging, translate_weight=0.7)
    print(f"[info] joint dataset size: {len(joint)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = build_joint_collator(tokenizer, max_length=128)
    loader = DataLoader(
        joint, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    task_counts = {TASK_TRANSLATE: 0, TASK_SUPERTAG: 0}
    for i, batch in enumerate(loader):
        for t in batch["task"]:
            task_counts[t] += 1
        if i < args.batches:
            print(
                f"\n[batch {i}] tasks={batch['task']} "
                f"input_ids={tuple(batch['input_ids'].shape)} "
                f"labels={tuple(batch['labels'].shape)}"
            )
            print(
                "  first source:",
                tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)[:140],
            )
            # Replace -100 for decoding preview.
            first_labels = batch["labels"][0].clone()
            first_labels[first_labels == -100] = tokenizer.pad_token_id
            print(
                "  first target:",
                tokenizer.decode(first_labels, skip_special_tokens=True)[:140],
            )
        if i >= args.batches:
            break

    print(f"\n[info] task mix seen: {task_counts}")


if __name__ == "__main__":
    main()
