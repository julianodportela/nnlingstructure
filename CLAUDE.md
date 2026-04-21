# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LING 3850 final project investigating whether injecting syntactic inductive biases into multilingual NMT improves translation quality for morphologically-rich, agglutinative Basque (Euskara). The core experiment: jointly train `facebook/nllb-200-distilled-600M` on Spanish→Basque translation AND a Basque **supertagging** auxiliary task, then compare BLEU/chrF against the unmodified NLLB baseline.

**Team:** Juliano Dantas Portela, Gbemiga Salu, Joe Zimmerman

### Project phases

| # | Task | Status |
|---|------|--------|
| 1 | Baseline FLORES-200 eval + joint PyTorch data pipeline | Done |
| 2 | MTL architecture + training loop + Bouchet fine-tune run | Pending |
| 3 | Final metrics + linguistically-informed Basque analysis | Pending |
| 4 | Final report | Pending |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.11+. Target compute: CUDA (Linux/Bouchet) or MPS (macOS); CPU works but is too slow for training. Disk: ~4 GB for NLLB weights, ~1.2 GB for Tatoeba archive.

## Commands

**Baseline eval (FLORES-200, Spanish→Basque):**
```bash
# Quick smoke test
.venv/bin/python src/baseline_eval.py --split dev --limit 8 --batch-size 4 --num-beams 1 --max-new-tokens 128

# Full canonical benchmark (1,012 sentences)
.venv/bin/python src/baseline_eval.py --split devtest --batch-size 8 --num-beams 4 --max-new-tokens 256
```
Outputs BLEU + chrF scores + hypothesis files to `outputs/baseline/`.

**Data pipeline validation:**
```bash
# Dry-run (no downloads, in-memory stubs — run this first)
.venv/bin/python src/smoke_dataloader_dry.py --batches 4

# Full smoke test (downloads ~1.1 GB Tatoeba archive on first run)
.venv/bin/python src/smoke_dataloader.py --tl-limit 128 --batches 3
```

## Architecture

### Data layer (`src/data/`)

Three dataset classes compose into a joint MTL loader:

**`TatoebaEsEuDataset`** (`tatoeba.py`)  
Streams Tatoeba Challenge v2023-09-26 `eus-spa` pairs via gzip to avoid loading 1.1 GB into memory. The archive stores pairs as Basque→Spanish (alphabetical ISO); the loader swaps direction to yield Spanish→Basque.

**`BasqueUDDataset`** (`ud_treebank.py`)  
Downloads UD_Basque-BDT from UniversalDependencies GitHub and parses CONLLU format. Currently linearizes dependency parses as seq2seq targets in configurable `"pos"`, `"deprel"`, or `"pos+deprel"` format (e.g., `Etxera/NOUN/obl`). **This will be updated to emit supertags** (see Step 2 below).

**`JointMTLDataset`** (`joint.py`)  
Reproducibly interleaves translation and parsing examples using a pre-computed deterministic task schedule (controlled by `translate_weight`, default 0.8). Returns `JointExample` with `task`, `src_lang`, `tgt_lang` fields.

**`build_joint_collator(tokenizer)`** (`joint.py`)  
NLLB-aware collator. NLLB requires setting `src_lang` on the tokenizer before encoding; the collator groups examples by source language, tokenizes per-group, then pads and concatenates. Emits standard HF seq2seq batches: `input_ids`, `attention_mask`, `labels` (padded with -100), `task` list, and `forced_bos_token_id` tensor.

### Evaluation layer (`baseline_eval.py`)

Loads NLLB-200, generates translations on FLORES-200 `devtest`, scores with sacrebleu. The saved outputs serve as the Step 3 comparison baseline.

### Datasets

| Dataset | Split | Purpose |
|---------|-------|---------|
| FLORES-200 | `devtest` | Primary eval (1,012 es→eu sentences) |
| Tatoeba Challenge v2023-09-26 | `eus-spa/train` + `dev` | Translation fine-tuning (~5.3M pairs) |
| UD_Basque-BDT | `train` / `dev` / `test` | Supertagging auxiliary task (~5,396 train sentences) |

All datasets download lazily into `data/` (git-ignored) on first use.

## Step 2: What needs to be built

### Training script (`src/train.py`)

- Load model: `AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")`
- Construct `JointMTLDataset` from `TatoebaEsEuDataset` + `BasqueUDDataset`
- Wrap with `DataLoader(collate_fn=build_joint_collator(tokenizer))`
- Joint cross-entropy loss on both tasks; `labels` are already -100-padded
- **Strip non-model keys before the forward pass** — the collator returns `task` and `forced_bos_token_id` which `model.forward()` does not accept. Use the provided helper: `outputs = model(**model_inputs(batch))`
- Use `batch["task"]` (popped before forward) for per-task loss weighting if routing losses separately
- **Overfitting / catastrophic forgetting:** fine-tuning on a small Basque-only corpus risks degrading multilingual capabilities. Mitigations to consider: (a) low learning rate with early stopping on FLORES-200 BLEU, (b) interleave a small sample of other NLLB language pairs during training to regularize, (c) phased curriculum (Costa-jussà et al. 2022 §4) starting with broad multilingual data then Basque-specific.
- Checkpoint after each epoch; eval against FLORES-200 `dev` split for early stopping.

### Linguistically-informed evaluation (Step 3 requirement)

Beyond aggregate BLEU/chrF, the professor requires a **Basque-specific test set** targeting known difficulty features:
- **Ergative-absolutive alignment** — does the model correctly mark transitive subjects (ergative `-k`) vs. intransitive subjects/objects (absolutive)?
- **Agglutinative case morphology** — correct use of locative (`-n`), dative (`-i`), instrumental (`-z`), etc.
- **SOV word order and verb agreement** — Basque verbs agree with subject, object, and indirect object; hallucination of agreement affixes is a key failure mode.
- Design ~50–100 targeted sentence pairs from FLORES-200 or Tatoeba dev that exercise each feature, then score those subsets separately.

### Training data note

Tatoeba and UD_Basque-BDT may overlap with NLLB-200's pre-training data. If so, fine-tuning on the same sentences primarily tests overfitting risk rather than new knowledge acquisition. Consider (a) using only the Tatoeba `dev` split (held-out sentences) for evaluation, (b) focusing on supertagging as the novel signal, since it adds explicit morphosyntactic supervision not present in standard NMT training.

## Key references

- Costa-jussà et al. 2022 — *No Language Left Behind* (NLLB model + phased curriculum)
- Niehues & Cho 2017 — *Exploiting Linguistic Resources for NMT Using Multi-task Learning*
- Marvin & Linzen 2018 — *Targeted Syntactic Evaluation of LMs* (joint LM + supertagging)
- Artetxe et al. 2020 — *Translation Artifacts in Cross-lingual Transfer Learning*
