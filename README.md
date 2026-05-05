# Injecting Syntactic Inductive Biases into Multilingual NMT for Basque

**LING 3850 Final Project** — Juliano Dantas Portela, Gbemiga Salu, Joe Zimmerman

We investigate whether jointly training Meta's `nllb-200-distilled-600M` on
Spanish→Basque translation and a Basque **supertagging** auxiliary task improves
translation quality over the unmodified NLLB-200 baseline. The supertagging
signal — UPOS tag + morphological features + dependency relation per token —
is drawn from UD_Basque-BDT and targets Basque's morphologically-rich
agglutinative case system (ergative-absolutive alignment, locative, dative,
instrumental suffixes, SOV word order).

---

## Results

All models evaluated on FLORES-200 `devtest` (1,012 Spanish→Basque sentences).
The ergative-absolutive subset covers 950 of those sentences whose Basque references
contain at least one UD-attested unambiguously case-marked form (112 with ergative
forms, 946 with absolutive forms; overlap permitted).

`tw` = translate weight (fraction of translation examples in the joint dataset mix).
Best fine-tuned value per column in **bold**.

### FLORES-200 devtest (n=1,012)

| Model | BLEU | chrF | Δ BLEU |
|---|---|---|---|
| NLLB-200 baseline (unmodified) | 10.67 | 49.78 | — |
| Fine-tuned, translation-only | 11.61 | 51.81 | +0.94 |
| Fine-tuned, MTL tw=0.65 (main run) | 11.64 | 51.80 | +0.97 |
| Fine-tuned, MTL tw=0.80 | 11.80 | **51.94** | +1.13 |
| Fine-tuned, MTL tw=0.50 | **11.85** | **51.94** | +1.18 |

### Ergative-absolutive case-marking accuracy (n=950 subset)

| Model | ERG hit | ERG err | ABS hit | ABS err |
|---|---|---|---|---|
| NLLB-200 baseline | 52.7% | 6.25% | 83.3% | 1.37% |
| Translation-only | 51.8% | **4.46%** | 84.2% | 1.27% |
| MTL tw=0.65 | **55.4%** | 5.36% | 84.2% | **0.95%** |
| MTL tw=0.80 | 52.7% | **4.46%** | **84.4%** | 1.16% |
| MTL tw=0.50 | 54.5% | 6.25% | 83.7% | 1.48% |

**ERG hit** = fraction of sentences where the expected ergative form appears in the hypothesis.
**ERG err** = fraction where the wrong-case counterpart appears instead.

The aggregate FLORES-200 metrics show fine-tuning lifts BLEU/chrF by ≈ +1 BLEU and +2 chrF
regardless of whether the supertagging objective is included, but the ergative subset reveals
a dissociation: translation-only fine-tuning *degrades* ergative recall (52.7% → 51.8%), while
MTL tw=0.65 is the only configuration to improve it (52.7% → 55.4%, +2.7 pp). This indicates
that the supertagging auxiliary task contributes a targeted morphosyntactic effect that
aggregate metrics fail to register.

---

## Setup

Python 3.11+. CUDA GPU required for training and full evaluation.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Disk: ~4 GB for NLLB-200 weights, ~1.2 GB for Tatoeba archive, ~200 MB for UD treebank.
All datasets download lazily into `data/` on first use.

---

## Reproducing the experiments

All training and evaluation runs on Bouchet via SLURM (`education_gpu` partition).
The cluster enforces one GPU job at a time per user, so jobs must be chained sequentially.

### 1. Baseline evaluation

```bash
# Smoke test (8 sentences, fast)
.venv/bin/python src/baseline_eval.py --split dev --limit 8 \
    --batch-size 4 --num-beams 1 --max-new-tokens 128

# Full benchmark (1,012 sentences) — or via SLURM:
sbatch jobs/baseline_eval.sh
```

Outputs: `outputs/baseline/hyp.devtest.eus_Latn.txt`, `metrics.devtest.json`.

### 2. Data pipeline smoke tests

```bash
# Dry-run — no downloads, in-memory stubs
.venv/bin/python src/smoke_dataloader_dry.py --batches 4

# Full smoke — downloads Tatoeba archive (~1.1 GB) on first run
.venv/bin/python src/smoke_dataloader.py --tl-limit 128 --batches 3
```

### 3. Main MTL training

```bash
sbatch jobs/train.sh
```

Trains NLLB-200-distilled-600M for up to 20 epochs with joint cross-entropy
loss on:
- **Translation**: Tatoeba Es→Eu, 100k pairs
- **Supertagging**: UD_Basque-BDT train split (~5,396 sentences), target format `supertag+deprel`

The main run (`jobs/train.sh`) uses `--translate-weight 0.65` (a 65/35 translation/supertag mix).
Per-epoch checkpoints are saved to `$SCRATCH/nllb_checkpoints/run_<timestamp>/`.
The best checkpoint (highest FLORES-200 dev BLEU) is copied to `best/`.
Final devtest metrics are written to `outputs/finetuned/metrics.devtest.json`.

Key flags for `src/train.py`:

| flag | default | description |
|---|---|---|
| `--tatoeba-limit` | `100000` | Number of Tatoeba training pairs |
| `--translate-weight` | `0.8` | Fraction of translation examples in the joint mix (overridden to `0.65` in the main `jobs/train.sh`) |
| `--supertag-fmt` | `supertag` | Target format; `supertag+deprel` adds dependency relation (used by the main run, recommended) |
| `--supertag-loss-weight` | `1.0` | Loss weight for supertagging relative to translation |
| `--lr` | `2e-5` | Learning rate (cosine schedule with 500-step warmup) |
| `--patience` | `3` | Early-stopping patience on FLORES-200 dev BLEU; `999` to disable |
| `--max-epochs` | `20` | Maximum training epochs |
| `--translation-only` | off | Fine-tune on translation data only (ablation baseline) |

### 4. Full evaluation

```bash
sbatch jobs/eval_full.sh
```

Runs four steps and prints a side-by-side comparison summary:
1. FLORES-200 devtest — baseline
2. FLORES-200 devtest — fine-tuned best checkpoint
3. Ergative-absolutive alignment — baseline
4. Ergative-absolutive alignment — fine-tuned

Outputs land in `outputs/full_eval/`.

To build the ergative test set manually (happens automatically inside `eval_full.sh`):

```bash
python src/build_ergative_testset.py \
    --data-dir $SCRATCH/nnling_data \
    --output   eval/ergative_test.json \
    --split    devtest
```

### 5. Ablation studies

Two ablations were run against the main MTL configuration:

| Ablation | Script | Description |
|---|---|---|
| Translation-only | `jobs/train_translation_only.sh` | No supertagging; establishes how much gain comes from fine-tuning alone vs. the syntactic signal |
| Weight sweep tw=0.5 | `jobs/train_weight_sweep.sh` | Equal translation/supertagging mix instead of the default 65/35 |

To run all ablations automatically (CPU watcher chains GPU jobs sequentially):

```bash
sbatch jobs/pipeline_ablations.sh
```

To evaluate a specific checkpoint:

```bash
sbatch jobs/eval_ablation.sh <checkpoint_path> <output_tag>
# e.g.:
sbatch jobs/eval_ablation.sh $SCRATCH/nllb_checkpoints/trans_only_run/best translation_only
```

---

## Architecture

A single NLLB-200 encoder–decoder (one set of weights) is fine-tuned on two
sequence-to-sequence tasks. The task is implicit in the source language:
Spanish input triggers translation, Basque input triggers supertagging.

```
   Translation example                Supertagging example
   src_lang=spa_Latn                  src_lang=eus_Latn
   ─────────────────────              ─────────────────────
   "El hombre comió                   "Gizonak sagarra
    la manzana."                       jan zuen."
        │                                    │
        ▼                                    ▼
   ┌─────────────────────────────────────────────────┐
   │   NLLB-200-distilled-600M encoder–decoder       │
   │   (shared weights, single forward pass)         │
   └─────────────────────────────────────────────────┘
        │                                    │
        ▼                                    ▼
   "Gizonak sagarra                   "Gizonak/NOUN|Case=Erg|
    jan zuen."                         Number=Sing/nsubj
                                       sagarra/NOUN|Case=Abs|
                                       Number=Sing/obj …"
```

**Joint dataset**: `JointMTLDataset` interleaves translation pairs (Tatoeba)
and supertagging pairs (UD_Basque-BDT) at a configurable ratio using a
pre-computed deterministic schedule for reproducibility.

**Collation**: The NLLB tokenizer requires `src_lang` to be set before
encoding. `build_joint_collator` groups examples by source language
(Spanish for translation, Basque for supertagging), tokenizes per group,
then pads and concatenates.

**Loss**: Per-token cross-entropy with HuggingFace's `-100` label convention
for padding. Each example's loss is averaged over non-padding tokens, then
a per-task weight is applied before averaging across the batch.

**Supertagging target format** (`supertag+deprel`):
```
Gizonak/NOUN|Case=Erg|Number=Sing/nsubj sagarra/NOUN|Case=Abs|Number=Sing/obj jan/VERB|Aspect=Perf/root du/AUX|Number[subj]=Sing/aux
```

---

## Evaluation

### FLORES-200 (primary)

1,012 Spanish→Basque sentence pairs from the `devtest` split.
Scored with [sacrebleu](https://github.com/mjpost/sacrebleu): BLEU (tokenized, mixed case)
and chrF (character n-gram F-score).

Sacrebleu signatures for reproducibility:
- BLEU: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

### Ergative-absolutive alignment (linguistically-targeted)

`eval/ergative_test.json` is built by `src/build_ergative_testset.py` by:
1. Extracting word forms that appear *exclusively* with `Case=Erg` (or `Case=Abs`)
   across all UD_Basque-BDT splits (~1,518 unambiguous ergative forms,
   ~3,976 unambiguous absolutive forms)
2. Filtering FLORES-200 devtest to the 950 sentences whose Basque references contain
   at least one such unambiguous form (112 with ergative forms, 946 with absolutive)
3. Building counterpart maps: for each ergative form, the absolutive form sharing
   the same lemma, and vice versa

At evaluation time, for each test sentence:
- **Hit**: expected case-marked form appears in the hypothesis
- **Error**: wrong-case counterpart appears in the hypothesis

This isolates case-marking behavior from overall translation quality.

---

## Spinoff experiment

`src/spinoff/` and `src/train_spinoff.py` contain a separate experiment
(not part of the main project submission) that investigates scaling the
syntactic signal by replacing the small UD treebank (~5k sentences) with
stanza-annotated Tatoeba pairs (~100k sentences) as the supertagging source.

See commit history for `[spinoff]`-prefixed commits. To run it:

```bash
# Step 1: annotate (GPU, ~30 min) — or CPU variant: jobs/annotate_tatoeba_cpu.sh
sbatch jobs/annotate_tatoeba.sh

# Step 2+3: train then eval, fully automated
sbatch jobs/pipeline_spinoff.sh
```

---

## Repository layout

```
.
├── requirements.txt
├── eval/
│   └── ergative_test.json          ← generated by build_ergative_testset.py
├── jobs/
│   ├── train.sh                    ← SLURM: main MTL fine-tuning (20 epochs)
│   ├── train_translation_only.sh   ← SLURM: translation-only ablation
│   ├── train_weight_sweep.sh       ← SLURM: tw=0.5 ablation
│   ├── baseline_eval.sh            ← SLURM: FLORES-200 baseline eval
│   ├── eval_full.sh                ← SLURM: full eval (BLEU + ergative, both models)
│   ├── eval_ergative.sh            ← SLURM: ergative eval only
│   ├── eval_ablation.sh            ← SLURM: eval a specific checkpoint
│   ├── eval_checkpoint.sh          ← SLURM: FLORES-200 eval on any checkpoint
│   ├── pipeline_ablations.sh       ← SLURM: CPU watcher — chains all ablation jobs
│   ├── test_cuda.sh                ← SLURM: sanity-check CUDA availability
│   ├── [spinoff] annotate_tatoeba.sh / annotate_tatoeba_cpu.sh
│   ├── [spinoff] train_spinoff.sh / eval_spinoff.sh / pipeline_spinoff.sh
│   └── logs/                       ← SLURM stdout/stderr (git-ignored)
├── src/
│   ├── train.py                    ← MTL fine-tuning loop
│   ├── baseline_eval.py            ← FLORES-200 eval harness
│   ├── build_ergative_testset.py   ← constructs eval/ergative_test.json
│   ├── eval_ergative.py            ← ergative-absolutive alignment eval
│   ├── smoke_dataloader.py         ← full data pipeline smoke test
│   ├── smoke_dataloader_dry.py     ← dry-run smoke test (no downloads)
│   ├── data/
│   │   ├── tatoeba.py              ← Tatoeba Es-Eu streaming loader
│   │   ├── ud_treebank.py          ← UD_Basque-BDT loader + linearization
│   │   ├── joint.py                ← JointMTLDataset + NLLB-aware collator
│   │   └── tatoeba_annotated.py    ← [spinoff] stanza-annotated Tatoeba loader
│   ├── train_spinoff.py            ← [spinoff] training script
│   └── spinoff/
│       └── annotate_tatoeba.py     ← [spinoff] stanza annotation preprocessor
└── outputs/                        ← committed eval results (see below)
```

### outputs/ structure

The `outputs/` directory is committed to the repo so all reported numbers are
reproducible without re-running the experiments. Devtest subdirectories contain
hypothesis, reference, and source files alongside `metrics.devtest.json`;
ergative subdirectories contain `metrics.json` (aggregate scores) and
`results.json` (per-sentence hits/errors).

```
outputs/
├── baseline/                       ← unmodified NLLB-200 on FLORES devtest
├── finetuned/                      ← main MTL run (tw=0.65, supertag+deprel)
├── full_eval/                      ← baseline + finetuned + ergative eval (final)
├── full_eval_run1/                 ← earlier run with tw=0.8 (kept for provenance)
└── ablations/
    ├── translation_only/           ← translation-only fine-tune
    ├── translation_only_ergative/
    ├── weight_sweep_050/           ← MTL tw=0.5
    ├── weight_sweep_050_ergative/
    ├── mtl_run1_tw08/              ← MTL tw=0.8 (run 1, full devtest)
    └── mtl_run1_tw08_ergative/
```

The spinoff experiment (stanza-annotated Tatoeba MTL) has training/eval scripts
in [src/spinoff/](src/spinoff/) and [jobs/](jobs/) but was not run for the final report;
no `outputs/spinoff*/` directories are committed.

---

## Datasets

| Dataset | Split used | Purpose | Size |
|---|---|---|---|
| [FLORES-200](https://github.com/facebookresearch/flores) | `devtest` | Primary translation eval | 1,012 es→eu sentences |
| [Tatoeba Challenge v2023-09-26](https://github.com/Helsinki-NLP/Tatoeba-Challenge) | `eus-spa/train` | Translation fine-tuning | ~5.3M pairs (capped at 100k) |
| [UD_Basque-BDT](https://github.com/UniversalDependencies/UD_Basque-BDT) | `train` / `dev` | Supertagging auxiliary task + ergative test set | ~5,396 train sentences |

All three download automatically into `data/` (git-ignored) on first use.

---

## References

- Costa-jussà et al. 2022 — [*No Language Left Behind*](https://arxiv.org/abs/2207.04672) (NLLB-200 model and phased multilingual curriculum)
- Niehues & Cho 2017 — [*Exploiting Linguistic Resources for NMT Using Multi-task Learning*](https://aclanthology.org/W17-4708/) (MTL with linguistic auxiliary tasks)
- Sennrich & Haddow 2016 — [*Linguistic Input Features Improve NMT*](https://aclanthology.org/W16-2209/) (morphological factor injection into NMT)
- Marvin & Linzen 2018 — [*Targeted Syntactic Evaluation of Language Models*](https://aclanthology.org/D18-1151/) (targeted morphosyntactic test sets)
- Artetxe et al. 2020 — [*Translation Artifacts in Cross-lingual Transfer*](https://aclanthology.org/2020.emnlp-main.618/) (artifacts and evaluation biases in cross-lingual NMT)
- Tiedemann 2020 — [*The Tatoeba Translation Challenge*](https://aclanthology.org/2020.wmt-1.139/) (Tatoeba dataset)
