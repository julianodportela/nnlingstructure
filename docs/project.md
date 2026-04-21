# Project Specification

**Title:** Injecting Syntactic Inductive Biases into Massively Multilingual NMT for Agglutinative Languages.

**Course:** LING 3850 — Final Project.

**Team:** Juliano Dantas Portela, Gbemiga Salu, Joe Zimmerman.

## 1. Research question

Does jointly training a pre-trained massively-multilingual NMT model on
translation **and** Basque supertagging improve its ability to translate
a morphologically dense, non-Indo-European target language?

Concretely: fine-tune Meta's `facebook/nllb-200-distilled-600M` for
Spanish→Basque via multi-task learning (MTL) on (a) Spanish-Basque parallel
text and (b) Basque supertagging; compare to a plain NLLB baseline.

## 2. Hypothesis

Exposing the shared encoder-decoder to explicit morphosyntactic structure
(supertags) during fine-tuning reduces structural hallucinations (spurious
affixes, mangled ergative-absolutive alignment) and improves translation of
morphologically complex Basque output.

## 3. Approach

- Backbone: `facebook/nllb-200-distilled-600M` (frozen tokenizer / vocab, fine-tuned weights).
- Two tasks, single shared encoder-decoder, alternating batches:
  1. **Translation:** Spanish (`spa_Latn`) → Basque (`eus_Latn`) on Tatoeba Challenge.
  2. **Supertagging:** Basque (`eus_Latn`) → per-token supertag sequence on UD_Basque-BDT.
     A supertag combines UPOS + morphological features, e.g. `joango/VERB|Aspect=Prosp|VerbForm=Part`.
     This is "nearly parsing" (Bangalore & Joshi 1999) without requiring a full parser — the approach
     used by Marvin & Linzen (2018) to improve grammatical knowledge via joint LM + supertag training.
- Combined loss: standard cross-entropy on both tasks, weighted by task mix
  (see `JointMTLDataset.translate_weight`).
- Evaluation: FLORES-200 Spanish→Basque `devtest`, BLEU + chrF vs baseline NLLB
  on the same decoding config. Additionally, a targeted Basque-specific test set
  evaluating ergative-absolutive marking, agglutinative case, and verb agreement.

## 4. Success criteria

**Quantitative:** Improvement in BLEU and chrF on FLORES-200 `devtest` over the
baseline NLLB numbers produced by [`src/baseline_eval.py`](../src/baseline_eval.py).

**Qualitative:** Inspection of model output for (a) ergative vs absolutive
marking, (b) affix hallucination rate, (c) handling of agglutinative noun/verb
morphology. This is the analysis slice owned by step 3.

## 5. Data

| purpose | dataset | split | notes |
| --- | --- | --- | --- |
| translation train | Tatoeba Translation Challenge v2023-09-26 | `eus-spa/train` | pair dir is alphabetical; loader swaps to Es→Eu |
| translation dev | Tatoeba Translation Challenge v2023-09-26 | `eus-spa/dev` | ~1000 sentences |
| supertagging train | UD_Basque-BDT | `train` | ~5396 sentences; UPOS+FEATS supertags |
| supertagging dev | UD_Basque-BDT | `dev` | held out for supertag-task monitoring |
| translation eval | FLORES-200 | `devtest` | 1012 sentences, canonical comparison |

Loaders live in `src/data/` and download into `data/` on first use.

## 6. Task breakdown and timeline

Lifted from the approved proposal (see `docs/proposal.pdf` if added to the
repo, or the original submission).

| # | task | owner | est. | status |
| --- | --- | --- | --- | --- |
| 1 | Baseline eval + joint PyTorch data pipeline | Joe, Gbemiga | 1 week | **done** — this repo |
| 2 | MTL architecture + joint-loss training loop + fine-tune run on Bouchet | Juliano, Joe | 1.5 weeks | pending |
| 3 | Final metrics + linguistic analysis (ergative/absolutive, Basque morphology) | Juliano, Gbemiga | 1 week | pending |
| 4 | Final report | everyone | 1 week | pending |

## 7. What step 1 delivers (this repo, checked-in code)

- `src/baseline_eval.py` — FLORES-200 Es→Eu eval for NLLB-200, BLEU + chrF
  with sacrebleu signatures for reproducibility.
- `src/data/ud_treebank.py` — Basque UD-BDT CONLLU loader that emits
  `{source: sentence, target: linearised_supertags}` pairs. Target format is
  switchable: `"supertag"` (default, UPOS+FEATS), `"pos"`, `"deprel"`, or `"pos+deprel"`.
- `src/data/tatoeba.py` — Tatoeba Challenge `eus-spa` loader with streaming
  gzip reads (avoids pulling the whole train file into memory when sampling)
  and direction-swap to Spanish→Basque.
- `src/data/joint.py` — `JointMTLDataset` (reproducible interleaving of the
  translation and supertagging tasks with tunable `translate_weight`) and
  `build_joint_collator` (emits NLLB-ready batches with per-example `src_lang`
  and `forced_bos_token_id`).
- `src/smoke_dataloader.py` / `src/smoke_dataloader_dry.py` — end-to-end
  validators for the pipeline shape (the dry version needs no 1.1 GB
  download).

## 8. What step 2 needs to add

- A training script (`src/train.py` or similar) that:
  - Constructs the joint dataset from `BasqueUDDataset(fmt="supertag")` + `TatoebaEsEuDataset`.
  - Wraps it in a `DataLoader` with `build_joint_collator(tokenizer)`.
  - Loads `AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")`.
  - Runs a joint-loss training loop: for each batch, compute seq2seq cross-entropy
    and optionally route per-task loss weighting via `batch["task"]`. Labels already
    come masked with `-100` on padding so the default HF loss ignores them.
  - Guards against catastrophic forgetting: low LR, early stopping on FLORES-200 BLEU,
    and/or interleaving a small sample of other NLLB language pairs during training.
  - Checkpoints periodically; optionally phased curriculum à la Costa-jussà et al. 2022.
- A matching eval step that reruns `baseline_eval.py`'s harness against the
  fine-tuned checkpoint for apples-to-apples comparison.

## 9. References

1. Costa-jussà et al. 2022 — *No Language Left Behind: Scaling Human-Centered Machine Translation.* The NLLB paper — covers the model we fine-tune, Mixture-of-Experts, and the phased curriculum we may adopt.
2. Niehues & Cho 2017 — *Exploiting Linguistic Resources for Neural Machine Translation Using Multi-task Learning.* The MTL recipe this project builds on.
3. Marvin & Linzen 2018 — *Targeted Syntactic Evaluation of Language Models.* Validates joint LM + supertagging training as an auxiliary objective that improves grammatical knowledge — the direct precedent for our supertagging task.
4. Bangalore & Joshi 1999 — *Supertagging: An Approach to Almost Parsing.* Original formulation of supertagging as "nearly parsing".
5. Artetxe et al. 2020 — *Translation Artifacts in Cross-lingual Transfer Learning.* Relevant for interpreting eval-time behaviour when back-translated data is part of the training mix.
