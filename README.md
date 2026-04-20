# nnlingstructure

**Injecting Syntactic Inductive Biases into Massively Multilingual NMT for Agglutinative Languages.**

Final project for LING 3850. Team: Juliano Dantas Portela, Gbemiga Salu, Joe Zimmerman.

We fine-tune Meta's `nllb-200-distilled-600M` on Basque (Euskara) using multi-task
learning — jointly training on Spanish→Basque translation and Basque syntactic
parsing — and measure whether the syntactic signal improves Basque translation
over the NLLB baseline on FLORES-200.

See [docs/project.md](docs/project.md) for the full project spec, task
breakdown, and references.

## Project status

| step | description | owner | status |
| --- | --- | --- | --- |
| 1 | Baseline evaluation & data pipeline | Joe, Gbemiga | shipped in this repo |
| 2 | Multi-task architecture & fine-tuning | Juliano, Joe | not started |
| 3 | Final evaluation & linguistic analysis | Juliano, Gbemiga | not started |
| 4 | Final report | everyone | not started |

Step 1 produces:
- A reproducible FLORES-200 Spanish→Basque eval harness for NLLB-200 that records BLEU + chrF with sacrebleu signatures.
- A joint PyTorch data pipeline that interleaves Tatoeba Es-Eu parallel pairs with Basque UD Treebank parse pairs, emitting NLLB-ready batches with per-example `src_lang` / `forced_bos_token_id`.

## Requirements

- Python 3.11+ (tested on 3.13)
- macOS (MPS) or Linux (CUDA) — CPU works but is slow
- ~4 GB disk for the NLLB-200 distilled weights + ~1.2 GB for the Tatoeba archive

## Setup

```bash
cd final/code
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Everything below assumes the venv is active (or prefixed with `.venv/bin/`).

## Usage

### 1. Baseline translation eval (FLORES-200 Spanish → Basque)

Smoke test on a small subset (~15 seconds on MPS after model download):

```bash
.venv/bin/python src/baseline_eval.py --split dev --limit 8 \
    --batch-size 4 --num-beams 1 --max-new-tokens 128
```

Full FLORES-200 devtest (1012 sentences, beam=4, expect long wall time locally
— intended to run on Bouchet for the final numbers):

```bash
.venv/bin/python src/baseline_eval.py --split devtest \
    --batch-size 8 --num-beams 4 --max-new-tokens 256
```

Outputs land in `outputs/baseline/`:
- `hyp.<tag>.eus_Latn.txt` — model hypotheses
- `ref.<tag>.eus_Latn.txt` — FLORES-200 Basque references
- `src.<tag>.spa_Latn.txt` — Spanish sources
- `metrics.<tag>.json` — BLEU, chrF, sacrebleu signatures, run config

### 2. Joint MTL data pipeline smoke tests

Dry-run (no 1.1 GB Tatoeba download — uses a tiny in-memory Spanish-Basque stub
+ real Basque UD Treebank):

```bash
.venv/bin/python src/smoke_dataloader_dry.py --batches 4
```

Full smoke against real Tatoeba + UD (downloads the 1.1 GB `eus-spa.tar` on
first run):

```bash
.venv/bin/python src/smoke_dataloader.py --tl-limit 128 --batches 3
```

Both print sample batches with task tags, tokenised shapes, decoded
source/target previews, and a task-mix histogram.

## Layout

```
final/code/
├── README.md                  ← you are here
├── requirements.txt
├── docs/
│   └── project.md             ← detailed project spec + plan
├── src/
│   ├── baseline_eval.py       ← FLORES-200 eval harness for step 1
│   ├── smoke_dataloader.py    ← full pipeline smoke (real Tatoeba)
│   ├── smoke_dataloader_dry.py← fast pipeline smoke (stub translation)
│   └── data/
│       ├── __init__.py
│       ├── ud_treebank.py     ← Basque UD BDT → seq2seq parse pairs
│       ├── tatoeba.py         ← Tatoeba Challenge Es-Eu loader (streams + swaps direction)
│       └── joint.py           ← JointMTLDataset + NLLB-aware collator
└── outputs/ (git-ignored)     ← eval hypotheses + metrics
└── data/    (git-ignored)     ← downloaded FLORES / Tatoeba / UD
```

## Datasets

| dataset | split we use | purpose | source |
| --- | --- | --- | --- |
| FLORES-200 | `devtest` | eval (primary) | https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz |
| Tatoeba Translation Challenge v2023-09-26 | `eus-spa` pair | Es→Eu translation fine-tuning | https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eus-spa.tar |
| UD_Basque-BDT | `train`/`dev`/`test` | Basque parsing task | https://github.com/UniversalDependencies/UD_Basque-BDT |

All three are downloaded lazily on first use into `data/`. Note that the
Tatoeba pair archive is 1.13 GB.

## Notes for step 2 (MTL fine-tuning)

`src/data/joint.py` already exposes everything the training loop should need:

- `JointMTLDataset(translation, parsing, translate_weight=...)` — reproducible
  interleaving, `__getitem__` returns a `JointExample` with `task`, `src_lang`,
  `tgt_lang`.
- `build_joint_collator(tokenizer, max_length=...)` — returns batches with
  `input_ids`, `attention_mask`, `labels` (padded with `-100`), `task` list,
  and per-example `forced_bos_token_id`.
- `infinite_iter(loader)` — endless iterator for step-based training loops.

The parse-task linearisation format (pos / deprel / pos+deprel) is switchable
via `BasqueUDDataset(fmt=...)`.
