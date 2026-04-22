"""Build an ergative-absolutive alignment test set from UD_Basque-BDT + FLORES-200.

Extracts unambiguous ergative and absolutive word forms from UD_Basque-BDT (gold
Case annotations), then filters FLORES-200 devtest sentence pairs where the Basque
reference contains at least one such attested form. Writes eval/ergative_test.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import conllu

sys.path.insert(0, str(Path(__file__).parent))
from data.ud_treebank import download_ud_basque_bdt, UD_FILES

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_ROOT = "flores200_dataset"
TARGET_UPOS = {"NOUN", "PROPN", "PRON"}


def ensure_flores(data_dir: Path) -> Path:
    root = data_dir / FLORES_ROOT
    if (root / "devtest" / "spa_Latn.devtest").exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    tarball = data_dir / "flores200_dataset.tar.gz"
    if not tarball.exists():
        print("[info] downloading FLORES-200")
        urllib.request.urlretrieve(FLORES_URL, tarball)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(data_dir, filter="data")
    return root


def build_ud_lookup(
    data_dir: Path,
) -> tuple[set[str], set[str], dict[str, list[str]], dict[str, list[str]]]:
    """Return (erg_only, abs_only, erg_counterparts, abs_counterparts).

    erg_only: word forms that appear *exclusively* with Case=Erg across all UD splits.
    abs_only: word forms that appear *exclusively* with Case=Abs across all UD splits.
    erg_counterparts[form]: absolutive forms sharing the same lemma (i.e. the wrong
        form a model would produce if it drops the ergative marking).
    abs_counterparts[form]: ergative forms sharing the same lemma.
    """
    ud_dir = download_ud_basque_bdt(data_dir)

    form_cases: dict[str, set[str]] = defaultdict(set)
    lemma_case_forms: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for fname in UD_FILES.values():
        sentences = conllu.parse((ud_dir / fname).read_text("utf-8"))
        for sent in sentences:
            for tok in sent:
                if isinstance(tok["id"], tuple):
                    continue
                if tok.get("upos") not in TARGET_UPOS:
                    continue
                feats = tok.get("feats") or {}
                case = feats.get("Case")
                if not case:
                    continue
                form = tok["form"].lower()
                lemma = (tok.get("lemma") or tok["form"]).lower()
                form_cases[form].add(case)
                lemma_case_forms[lemma][case].add(form)

    erg_only = {f for f, cases in form_cases.items() if cases == {"Erg"}}
    abs_only = {f for f, cases in form_cases.items() if cases == {"Abs"}}

    erg_counterparts: dict[str, list[str]] = {}
    abs_counterparts: dict[str, list[str]] = {}
    for lemma, case_forms in lemma_case_forms.items():
        erg_forms = case_forms.get("Erg", set()) & erg_only
        abs_forms = case_forms.get("Abs", set()) & abs_only
        for ef in erg_forms:
            erg_counterparts[ef] = sorted(abs_forms)
        for af in abs_forms:
            abs_counterparts[af] = sorted(erg_forms)

    return erg_only, abs_only, erg_counterparts, abs_counterparts


def tokenize(text: str) -> set[str]:
    return set(re.split(r"[\s,.:;!?\"'()\[\]«»—–]+", text.lower())) - {""}


def build_test_set(
    flores_root: Path,
    erg_only: set[str],
    abs_only: set[str],
    erg_counterparts: dict[str, list[str]],
    abs_counterparts: dict[str, list[str]],
    split: str = "devtest",
) -> list[dict]:
    src_lines = (flores_root / split / f"spa_Latn.{split}").read_text("utf-8").splitlines()
    ref_lines = (flores_root / split / f"eus_Latn.{split}").read_text("utf-8").splitlines()

    test_cases = []
    for i, (src, ref) in enumerate(zip(src_lines, ref_lines)):
        tokens = tokenize(ref)
        erg_hits = sorted(tokens & erg_only)
        abs_hits = sorted(tokens & abs_only)
        if not erg_hits and not abs_hits:
            continue
        test_cases.append({
            "id": f"flores_{split}_{i:04d}",
            "source_es": src,
            "reference_eu": ref,
            "erg_forms_in_ref": erg_hits,
            "abs_forms_in_ref": abs_hits,
            "erg_counterparts": {f: erg_counterparts.get(f, []) for f in erg_hits},
            "abs_counterparts": {f: abs_counterparts.get(f, []) for f in abs_hits},
        })

    return test_cases


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build UD/FLORES-200 ergative-absolutive alignment test set."
    )
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", default="eval/ergative_test.json")
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    print("[info] building UD case lookup (all splits)...")
    erg_only, abs_only, erg_counterparts, abs_counterparts = build_ud_lookup(data_dir)
    print(f"[info] unambiguous forms: erg_only={len(erg_only)}  abs_only={len(abs_only)}")

    print("[info] ensuring FLORES-200 is present...")
    flores_root = ensure_flores(data_dir)

    print(f"[info] filtering FLORES-200 {args.split}...")
    test_cases = build_test_set(
        flores_root, erg_only, abs_only, erg_counterparts, abs_counterparts, args.split
    )

    n_erg = sum(1 for c in test_cases if c["erg_forms_in_ref"])
    n_abs = sum(1 for c in test_cases if c["abs_forms_in_ref"])
    print(
        f"[info] {len(test_cases)} sentences pass filter "
        f"({n_erg} with erg forms, {n_abs} with abs forms)"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(test_cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[info] wrote {out}")


if __name__ == "__main__":
    main()
