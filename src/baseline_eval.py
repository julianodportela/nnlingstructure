"""Baseline evaluation: translate FLORES-200 Spanish -> Basque with NLLB-200.

Saves hypotheses + references + BLEU/chrF to outputs/.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "spa_Latn"
TGT_LANG = "eus_Latn"

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_ROOT_DIR = "flores200_dataset"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_flores(data_dir: Path) -> Path:
    """Download and extract the official FLORES-200 tarball if not already present."""
    root = data_dir / FLORES_ROOT_DIR
    if (root / "dev" / f"{SRC_LANG}.dev").exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    tarball = data_dir / "flores200_dataset.tar.gz"
    if not tarball.exists():
        print(f"[info] downloading FLORES-200 from {FLORES_URL}")
        urllib.request.urlretrieve(FLORES_URL, tarball)
    print(f"[info] extracting {tarball}")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(data_dir, filter="data")
    return root


def load_flores_pair(data_dir: Path, split: str, limit: int | None):
    root = ensure_flores(data_dir)
    src_file = root / split / f"{SRC_LANG}.{split}"
    tgt_file = root / split / f"{TGT_LANG}.{split}"
    src_lines = src_file.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_file.read_text(encoding="utf-8").splitlines()
    assert len(src_lines) == len(tgt_lines), "FLORES splits must be line-aligned"
    pairs = list(zip(src_lines, tgt_lines))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def translate_batch(model, tokenizer, sentences, device, max_new_tokens: int, num_beams: int):
    tokenizer.src_lang = SRC_LANG
    enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    out = model.generate(
        **enc,
        forced_bos_token_id=forced_bos,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--limit", type=int, default=None, help="Use only first N rows (for smoke tests)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--output-dir", default="outputs/baseline")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    device = pick_device()
    print(f"[info] device={device}")

    pairs = load_flores_pair(Path(args.data_dir), args.split, args.limit)
    print(f"[info] loaded {len(pairs)} FLORES-200 {args.split} pairs")

    print(f"[info] loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()

    hypotheses: list[str] = []
    references: list[str] = [tgt for _, tgt in pairs]
    sources: list[str] = [src for src, _ in pairs]

    with torch.inference_mode():
        for i in tqdm(range(0, len(sources), args.batch_size), desc="translate"):
            batch = sources[i : i + args.batch_size]
            hyps = translate_batch(
                model, tokenizer, batch, device,
                max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
            )
            hypotheses.extend(hyps)

    bleu_metric = sacrebleu.BLEU()
    chrf_metric = sacrebleu.CHRF()
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    chrf = chrf_metric.corpus_score(hypotheses, [references])
    print(f"[result] BLEU = {bleu.score:.2f}")
    print(f"[result] chrF = {chrf.score:.2f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}" + (f"_n{args.limit}" if args.limit else "")

    (out_dir / f"hyp.{tag}.eus_Latn.txt").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    (out_dir / f"ref.{tag}.eus_Latn.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
    (out_dir / f"src.{tag}.spa_Latn.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    (out_dir / f"metrics.{tag}.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "split": args.split,
                "n": len(hypotheses),
                "num_beams": args.num_beams,
                "batch_size": args.batch_size,
                "bleu": bleu.score,
                "bleu_signature": str(bleu_metric.get_signature()),
                "chrf": chrf.score,
                "chrf_signature": str(chrf_metric.get_signature()),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[info] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
