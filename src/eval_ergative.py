"""Evaluate ergative-absolutive alignment in Spanish→Basque translations.

Loads a model checkpoint (or baseline NLLB-200), translates the Spanish sources
from the UD/FLORES-200 derived test set, and reports case-marking accuracy
alongside standard BLEU/chrF scores. Run build_ergative_testset.py first to
generate eval/ergative_test.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "spa_Latn"
TGT_LANG = "eus_Latn"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize(text: str) -> set[str]:
    return set(re.split(r"[\s,.:;!?\"'()\[\]«»—–]+", text.lower())) - {""}


def translate_batch(
    model,
    tokenizer,
    sources: list[str],
    device,
    num_beams: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer.src_lang = SRC_LANG
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)
    enc = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            forced_bos_token_id=forced_bos,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def evaluate(
    model,
    tokenizer,
    test_cases: list[dict],
    device,
    num_beams: int = 4,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> tuple[list[dict], dict]:
    sources = [c["source_es"] for c in test_cases]
    hypotheses: list[str] = []
    for i in tqdm(range(0, len(sources), batch_size), desc="translating"):
        hypotheses.extend(
            translate_batch(model, tokenizer, sources[i : i + batch_size], device, num_beams, max_new_tokens)
        )

    results = []
    for case, hyp in zip(test_cases, hypotheses):
        hyp_tokens = tokenize(hyp)
        erg_forms = case["erg_forms_in_ref"]
        abs_forms = case["abs_forms_in_ref"]

        erg_hit = any(f in hyp_tokens for f in erg_forms) if erg_forms else None
        abs_hit = any(f in hyp_tokens for f in abs_forms) if abs_forms else None

        # Error: the counterpart (wrong-case) form appears where the correct one was expected.
        erg_error_forms = [cf for f in erg_forms for cf in case["erg_counterparts"].get(f, [])]
        abs_error_forms = [cf for f in abs_forms for cf in case["abs_counterparts"].get(f, [])]
        erg_error = bool(erg_error_forms) and any(f in hyp_tokens for f in erg_error_forms)
        abs_error = bool(abs_error_forms) and any(f in hyp_tokens for f in abs_error_forms)

        results.append({
            "id": case["id"],
            "source_es": case["source_es"],
            "reference_eu": case["reference_eu"],
            "hypothesis": hyp,
            "erg_forms_expected": erg_forms,
            "abs_forms_expected": abs_forms,
            "erg_hit": erg_hit,
            "abs_hit": abs_hit,
            "erg_error": erg_error,
            "abs_error": abs_error,
        })

    refs = [[c["reference_eu"] for c in test_cases]]
    bleu_metric = sacrebleu.BLEU()
    chrf_metric = sacrebleu.CHRF()
    bleu = bleu_metric.corpus_score(hypotheses, refs).score
    chrf = chrf_metric.corpus_score(hypotheses, refs).score

    erg_results = [r for r in results if r["erg_hit"] is not None]
    abs_results = [r for r in results if r["abs_hit"] is not None]

    def rate(items, key):
        return sum(r[key] for r in items) / len(items) if items else None

    metrics = {
        "n_total": len(results),
        "n_erg_sentences": len(erg_results),
        "n_abs_sentences": len(abs_results),
        "bleu": bleu,
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf": chrf,
        "chrf_signature": str(chrf_metric.get_signature()),
        "ergative_hit_rate": rate(erg_results, "erg_hit"),
        "ergative_error_rate": rate(erg_results, "erg_error"),
        "absolutive_hit_rate": rate(abs_results, "abs_hit"),
        "absolutive_error_rate": rate(abs_results, "abs_error"),
    }

    return results, metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate ergative-absolutive alignment in Spanish→Basque translations."
    )
    ap.add_argument(
        "--model-path",
        default=None,
        help="Fine-tuned checkpoint directory; defaults to baseline NLLB-200",
    )
    ap.add_argument("--test-file", default="eval/ergative_test.json")
    ap.add_argument("--output-dir", default="outputs/ergative_eval")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    device = pick_device()
    print(f"[info] device={device}")

    model_src = args.model_path or MODEL_NAME
    print(f"[info] loading model from {model_src}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_src).to(device)
    model.generation_config.max_length = None
    model.eval()

    test_file = Path(args.test_file)
    test_cases = json.loads(test_file.read_text("utf-8"))
    print(f"[info] loaded {len(test_cases)} test cases from {test_file}")

    results, metrics = evaluate(
        model, tokenizer, test_cases, device,
        args.num_beams, args.batch_size, args.max_new_tokens,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n── Ergative-Absolutive Alignment Evaluation ──")
    print(f"  n={metrics['n_total']}  BLEU={metrics['bleu']:.2f}  chrF={metrics['chrf']:.2f}")
    print(f"  Ergative sentences ({metrics['n_erg_sentences']}):")
    if metrics["ergative_hit_rate"] is not None:
        print(f"    hit rate:   {metrics['ergative_hit_rate']:.1%}")
        print(f"    error rate: {metrics['ergative_error_rate']:.1%}")
    print(f"  Absolutive sentences ({metrics['n_abs_sentences']}):")
    if metrics["absolutive_hit_rate"] is not None:
        print(f"    hit rate:   {metrics['absolutive_hit_rate']:.1%}")
        print(f"    error rate: {metrics['absolutive_error_rate']:.1%}")
    print(f"\n[info] detailed results → {out_dir}/")


if __name__ == "__main__":
    main()
