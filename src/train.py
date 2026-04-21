"""Fine-tune NLLB-200-distilled-600M on joint Spanish→Basque translation + Basque supertagging.

Full parameter fine-tuning with configurable per-task loss weighting and early stopping
on FLORES-200 dev BLEU. Saves a checkpoint after every epoch and auto-resumes from the
latest checkpoint on restart, so interrupted SLURM jobs can be resubmitted unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent))

from data import (
    BasqueUDDataset,
    JointMTLDataset,
    TatoebaEsEuDataset,
    build_joint_collator,
    TASK_TRANSLATE,
    TASK_SUPERTAG,
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "spa_Latn"
TGT_LANG = "eus_Latn"
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_ROOT = "flores200_dataset"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# FLORES-200 evaluation
# ---------------------------------------------------------------------------

def ensure_flores(data_dir: Path) -> Path:
    root = data_dir / FLORES_ROOT
    if (root / "dev" / f"{SRC_LANG}.dev").exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    tarball = data_dir / "flores200_dataset.tar.gz"
    if not tarball.exists():
        print("[info] downloading FLORES-200")
        urllib.request.urlretrieve(FLORES_URL, tarball)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(data_dir, filter="data")
    return root


def eval_flores(
    model,
    tokenizer,
    data_dir: Path,
    device,
    split: str = "dev",
    batch_size: int = 8,
    num_beams: int = 4,
    max_new_tokens: int = 256,
) -> dict:
    """Translate FLORES-200 split and return BLEU, chrF, hypotheses, and sacrebleu signatures."""
    root = ensure_flores(data_dir)
    src_lines = (root / split / f"{SRC_LANG}.{split}").read_text("utf-8").splitlines()
    tgt_lines = (root / split / f"{TGT_LANG}.{split}").read_text("utf-8").splitlines()
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    hypotheses: list[str] = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(src_lines), batch_size), desc=f"eval/{split}", leave=False):
            tokenizer.src_lang = SRC_LANG
            enc = tokenizer(
                src_lines[i : i + batch_size],
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(device)
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            hypotheses.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    model.train()

    bleu_metric = sacrebleu.BLEU()
    chrf_metric = sacrebleu.CHRF()
    return {
        "bleu": bleu_metric.corpus_score(hypotheses, [tgt_lines]).score,
        "chrf": chrf_metric.corpus_score(hypotheses, [tgt_lines]).score,
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
        "hypotheses": hypotheses,
        "references": tgt_lines,
        "sources": src_lines,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    state: dict,
) -> None:
    epoch_dir = ckpt_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(epoch_dir)
    torch.save(optimizer.state_dict(), epoch_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), epoch_dir / "scheduler.pt")
    # Write state last so it only exists once everything else is flushed.
    (ckpt_dir / "train_state.json").write_text(json.dumps(state, indent=2))
    print(f"[ckpt] saved epoch {epoch} → {epoch_dir}")


def load_latest_checkpoint(
    ckpt_dir: Path,
    model,
    optimizer,
    scheduler,
) -> dict | None:
    state_file = ckpt_dir / "train_state.json"
    if not state_file.exists():
        return None
    state = json.loads(state_file.read_text())
    epoch_dir = ckpt_dir / f"epoch_{state['epoch']}"
    if not epoch_dir.exists():
        print(f"[warn] train_state.json points to missing {epoch_dir}; starting fresh")
        return None
    print(f"[ckpt] resuming from epoch {state['epoch']}  best_bleu={state['best_bleu']:.2f}")
    loaded = AutoModelForSeq2SeqLM.from_pretrained(epoch_dir)
    model.load_state_dict(loaded.state_dict())
    del loaded
    # Use the model's actual device so optimizer state tensors land on CUDA,
    # not CPU — mixing devices would crash on the first optimizer.step().
    param_device = next(model.parameters()).device
    optimizer.load_state_dict(
        torch.load(epoch_dir / "optimizer.pt", map_location=param_device, weights_only=False)
    )
    scheduler.load_state_dict(
        torch.load(epoch_dir / "scheduler.pt", weights_only=False)
    )
    return state


def copy_to_best(ckpt_dir: Path, epoch: int) -> None:
    best_dir = ckpt_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(ckpt_dir / f"epoch_{epoch}", best_dir)
    print(f"[ckpt] best → epoch {epoch}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device,
    supertag_loss_weight: float,
) -> float:
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        tasks = batch.pop("task")
        batch.pop("forced_bos_token_id")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Per-token loss with -100 positions automatically zeroed by ignore_index.
        B, T, V = outputs.logits.shape
        per_token = loss_fct(
            outputs.logits.reshape(B * T, V),
            labels.reshape(B * T),
        ).reshape(B, T)

        # Average each example over its non-padding tokens.
        non_pad = (labels != -100).float()
        per_example = (per_token * non_pad).sum(-1) / non_pad.sum(-1).clamp(min=1)

        # Apply configurable per-task loss weight before averaging across the batch.
        weights = torch.tensor(
            [1.0 if t == TASK_TRANSLATE else supertag_loss_weight for t in tasks],
            dtype=torch.float32,
            device=device,
        )
        loss = (per_example * weights).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune NLLB-200 with joint translation + supertagging MTL."
    )
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="Directory in scratch to store per-epoch checkpoints")
    ap.add_argument("--output-dir", default="outputs/finetuned",
                    help="Where to write final devtest metrics and hypothesis files")
    ap.add_argument("--tatoeba-limit", type=int, default=100_000)
    ap.add_argument("--translate-weight", type=float, default=0.8,
                    help="Fraction of translation examples in the joint dataset mix")
    ap.add_argument("--supertag-loss-weight", type=float, default=1.0,
                    help="Loss weight for supertagging relative to translation (1.0 = equal 1:1)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--eval-num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop after this many epochs with no FLORES dev BLEU improvement")
    args = ap.parse_args()

    device = pick_device()
    print(f"[info] device={device}")

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets ---
    print("[info] loading datasets...")
    translation = TatoebaEsEuDataset(
        data_dir=data_dir, split="train", limit=args.tatoeba_limit
    )
    supertagging = BasqueUDDataset(data_dir=data_dir, split="train", fmt="supertag")
    print(f"[info] translation={len(translation)}  supertagging={len(supertagging)}")

    joint = JointMTLDataset(
        translation=translation,
        supertagging=supertagging,
        translate_weight=args.translate_weight,
    )

    # --- Model and tokenizer ---
    print(f"[info] loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.generation_config.max_length = None  # avoid max_new_tokens/max_length conflict warning
    collate = build_joint_collator(tokenizer, max_length=128)
    loader = DataLoader(
        joint,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.max_epochs
    print(f"[info] {steps_per_epoch} steps/epoch  {total_steps} total planned steps")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Resume from checkpoint if one exists ---
    resume = load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler)
    if resume is not None:
        start_epoch = resume["epoch"] + 1
        best_bleu = resume["best_bleu"]
        patience_count = resume["patience_count"]
        print(
            f"[info] resuming: start_epoch={start_epoch}"
            f"  best_bleu={best_bleu:.2f}"
            f"  patience={patience_count}/{args.patience}"
        )
    else:
        start_epoch = 0
        best_bleu = -1.0
        patience_count = 0

    # --- Training loop ---
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\n── epoch {epoch} / {args.max_epochs - 1} ──")

        train_loss = train_one_epoch(
            model, loader, optimizer, scheduler, device,
            supertag_loss_weight=args.supertag_loss_weight,
        )
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}")

        print(f"[epoch {epoch}] evaluating on FLORES-200 dev...")
        result = eval_flores(
            model, tokenizer, data_dir, device,
            split="dev",
            batch_size=args.eval_batch_size,
            num_beams=args.eval_num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        dev_bleu = result["bleu"]
        print(f"[epoch {epoch}] FLORES dev  BLEU={dev_bleu:.2f}  chrF={result['chrf']:.2f}")

        # Determine improvement before saving so the flag drives copy_to_best.
        # save_checkpoint must come first: copy_to_best copies from the epoch_N/
        # directory that save_checkpoint creates.
        is_new_best = dev_bleu > best_bleu
        if is_new_best:
            best_bleu = dev_bleu
            patience_count = 0
        else:
            patience_count += 1
            print(
                f"[epoch {epoch}] no improvement"
                f"  patience={patience_count}/{args.patience}"
            )

        save_checkpoint(ckpt_dir, epoch, model, optimizer, scheduler, {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_bleu": dev_bleu,
            "best_bleu": best_bleu,
            "patience_count": patience_count,
        })

        if is_new_best:
            copy_to_best(ckpt_dir, epoch)

        if patience_count >= args.patience:
            print(f"[info] early stopping after epoch {epoch}")
            break

    # --- Final devtest eval from best checkpoint ---
    best_dir = ckpt_dir / "best"
    if best_dir.exists():
        print("\n[info] loading best checkpoint for final devtest eval...")
        best_model = AutoModelForSeq2SeqLM.from_pretrained(best_dir).to(device)
        best_model.generation_config.max_length = None
    else:
        print("[warn] no best/ checkpoint found; using current model weights")
        best_model = model

    print("[info] running FLORES-200 devtest eval...")
    final = eval_flores(
        best_model, tokenizer, data_dir, device,
        split="devtest",
        batch_size=args.eval_batch_size,
        num_beams=args.eval_num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[result] FLORES devtest  BLEU={final['bleu']:.2f}  chrF={final['chrf']:.2f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hyp.devtest.eus_Latn.txt").write_text(
        "\n".join(final["hypotheses"]) + "\n", encoding="utf-8"
    )
    (out_dir / "ref.devtest.eus_Latn.txt").write_text(
        "\n".join(final["references"]) + "\n", encoding="utf-8"
    )
    (out_dir / "src.devtest.spa_Latn.txt").write_text(
        "\n".join(final["sources"]) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.devtest.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "split": "devtest",
                "n": len(final["hypotheses"]),
                "num_beams": args.eval_num_beams,
                "batch_size": args.eval_batch_size,
                "tatoeba_limit": args.tatoeba_limit,
                "translate_weight": args.translate_weight,
                "supertag_loss_weight": args.supertag_loss_weight,
                "lr": args.lr,
                "best_dev_bleu": best_bleu,
                "bleu": final["bleu"],
                "bleu_signature": final["bleu_signature"],
                "chrf": final["chrf"],
                "chrf_signature": final["chrf_signature"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[info] outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
