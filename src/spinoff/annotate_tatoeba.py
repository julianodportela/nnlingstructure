"""Annotate Tatoeba Es-Eu Basque sentences with stanza's eu pipeline.

Writes one JSON record per line to --output:
  {"src_es": "...", "tgt_eu": "...", "annotation": "word/UPOS|FEATS/deprel ..."}

Resumes from an existing partial output file if interrupted.
Run once before src/train_spinoff.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.tatoeba import TatoebaEsEuDataset
from src.data.ud_treebank import ParseFormat


def _parse_feats(feats_str: str | None) -> dict[str, str]:
    if not feats_str or feats_str == "_":
        return {}
    result: dict[str, str] = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _linearize(words, fmt: ParseFormat) -> str:
    """Convert stanza sentence words to a supertag target string."""
    parts: list[str] = []
    for w in words:
        form = w.text
        upos = w.upos or "_"
        deprel = w.deprel or "_"
        if fmt in ("supertag", "supertag+deprel"):
            feats = _parse_feats(w.feats)
            feats_str = "|".join(f"{k}={v}" for k, v in feats.items()) if feats else ""
            tag = f"{upos}|{feats_str}" if feats_str else upos
            if fmt == "supertag+deprel":
                parts.append(f"{form}/{tag}/{deprel}")
            else:
                parts.append(f"{form}/{tag}")
        elif fmt == "pos":
            parts.append(f"{form}/{upos}")
        elif fmt == "deprel":
            parts.append(f"{form}/{deprel}")
        else:  # pos+deprel
            parts.append(f"{form}/{upos}/{deprel}")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=100_000,
                    help="Max Tatoeba pairs to annotate")
    ap.add_argument("--fmt", default="supertag+deprel",
                    choices=["supertag", "supertag+deprel", "pos", "deprel", "pos+deprel"])
    ap.add_argument("--stanza-dir", default=None,
                    help="Directory for stanza model cache (default: ~/stanza_resources)")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="Documents per stanza batch (tune for GPU memory)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import stanza

    if args.stanza_dir:
        os.environ["STANZA_RESOURCES_DIR"] = args.stanza_dir

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Count already-done lines for resume logic.
    n_done = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            n_done = sum(1 for _ in f)
        print(f"[info] resuming: {n_done} lines already written")

    print("[info] loading Tatoeba pairs...")
    ds = TatoebaEsEuDataset(data_dir=Path(args.data_dir), split=args.split, limit=args.limit)
    all_pairs = [ds[i] for i in range(len(ds))]
    print(f"[info] {len(all_pairs)} pairs total; {len(all_pairs) - n_done} remaining")

    if n_done >= len(all_pairs):
        print("[info] already fully annotated")
        return

    remaining = all_pairs[n_done:]

    # Stanza processors: depparse is expensive; only include if needed.
    needs_dep = "deprel" in args.fmt
    processors = "tokenize,pos,lemma" + (",depparse" if needs_dep else "")
    print(f"[info] loading stanza eu pipeline  processors={processors!r}")

    stanza_kwargs: dict = dict(
        lang="eu",
        processors=processors,
        tokenize_no_ssplit=True,
        verbose=False,
    )
    if args.stanza_dir:
        stanza_kwargs["dir"] = args.stanza_dir

    # Download model if not present. stanza.download uses model_dir (not dir).
    # The env var STANZA_RESOURCES_DIR set above also controls the path.
    stanza.download("eu", processors=processors,
                    **({"model_dir": args.stanza_dir} if args.stanza_dir else {}))
    nlp = stanza.Pipeline(**stanza_kwargs)

    failed = 0
    print(f"[info] annotating {len(remaining)} sentences  fmt={args.fmt!r}")
    with out_path.open("a", encoding="utf-8") as out_f:
        chunk_size = args.chunk_size
        for i in tqdm(range(0, len(remaining), chunk_size), desc="annotate"):
            chunk = remaining[i : i + chunk_size]
            # Stanza batch API: pass a list of Document stubs.
            in_docs = [stanza.Document([], text=p["target"]) for p in chunk]
            out_docs = nlp(in_docs)

            for pair, doc in zip(chunk, out_docs):
                if not doc.sentences:
                    failed += 1
                    continue
                annotation = _linearize(doc.sentences[0].words, args.fmt)
                if not annotation:
                    failed += 1
                    continue
                out_f.write(
                    json.dumps(
                        {"src_es": pair["source"], "tgt_eu": pair["target"],
                         "annotation": annotation},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            out_f.flush()

    total_written = n_done + len(remaining) - failed
    print(f"[info] done  written={total_written}  failed/empty={failed}  → {out_path}")


if __name__ == "__main__":
    main()
