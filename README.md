# The Validity Gap in LLM-Agent Consensus — Experiment Package

Code, source data, derived results, and audit materials for the paper
**"GRAIN: Granularity and the Validity Gap in LLM-Agent Consensus."**

This is the artifact behind every number in the manuscript: the sample collectors, the scoring and
certification code, the archived model replies they run on, the human-annotation materials, and the
JSON result files the paper's tables and figures are generated from. All analyses that do not call a
model API can be rerun offline from what is stored here.

Paper source files (LaTeX, `.bbl`), figure-only plotting scripts, build caches, and build logs are
**intentionally excluded** — see [What is not included](#what-is-not-included).

---

## Table of contents

1. [The problem in one page](#the-problem-in-one-page)
2. [Theory at a glance](#theory-at-a-glance)
3. [Repository layout](#repository-layout)
4. [Experiment index](#experiment-index)
5. [Headline results](#headline-results)
6. [Data policy](#data-policy)
7. [Reproduction](#reproduction)
8. [Provenance and integrity checks](#provenance-and-integrity-checks)
9. [What is not included](#what-is-not-included)
10. [Legacy and superseded files](#legacy-and-superseded-files)
11. [Scope and caveats](#scope-and-caveats)

---

## The problem in one page

Classical Byzantine validity says: *if every honest participant proposes `v`, then `v` is decided.*
The clause is stated over an execution in which each honest proposal is a **fixed value**. A
language-model agent's proposal is instead a **random draw** from a distribution, and whether two
draws count as "the same proposal" is a design choice — the **grain** of agreement, written `g_x`.

So the antecedent of validity stops being a fact about the execution and becomes a **collision event**

```
A(h) = Σ_c p(c)^h          # probability that h honest draws all land in the same class
```

with `A(h) ≤ p_max^(h-1)`, where `p_max = max_c p(c)`. Refining the partition can never raise `A(h)`.
Where `A(h)` is small the validity clause is satisfied **vacuously**: a protocol may decide anything
and still be "correct." This is not a feasibility failure — agreement is unaffected — it is an
**informativeness** failure. The paper calls this the *validity gap* and measures it in three places:

| Gap | Question | Where it is measured |
|---|---|---|
| **Activation** (Q1) | How does the grain change how often validity is triggered? | `baseline/`, `E2`, `E3`, `E4`, `further_vendors/` |
| **Observability / certification** (Q2) | What can an observed histogram certify, and what fault budget survives? | `E5_certification/`, `X2_external_checker/` |
| **Instrument** (Q3) | Which relation do candidate agreement instruments actually compute? | `E1_human_calibration/`, `R1b_operation_partition/` |

Two partitions are measured directly: the **answer grain** (normalised final answers) and the
**lexical grain** (answer + a short normalised decisive-step summary, the KEY). The **operation
grain** — the human relation "same decisive operation" — is the intended target that instruments are
scored against.

## Theory at a glance

Every formal statement in the manuscript is machine-checked by
`E5_certification/code/check_paper_theory.py`, which writes
`E5_certification/results/paper_theory_checks.json` (all entries `pass`):

- **Proposition 1 (collision identity).** `Pr[U_h] = A(h) = Σ_c p(c)^h ≤ p_max^(h-1)`, with equality
  for `h > 1` iff `p` is uniform on its support. Equivalently `A(h) = exp{-(h-1) H_h(p)}`, the Rényi
  entropy form. Conditional independence does real work: if all agents copy one shared draw then
  `A(h) = 1`.
- **Proposition 2 (vacuity is not infeasibility).** A constant rule satisfies classical validity with
  probability `1 - A(h) + p(d)^h`; agreement remains achievable at any `A(h)`.
- **Proposition 3 (refinement and mixtures).** Refinement cannot raise `A(h)`, and the top-two margin
  is *not* monotone under refinement.
- **Theorem 1 (exact margin certificate).** `certify(N,f) = c  ⟺  N_c > f + max_{d≠c} N_d` returns the
  unique element of the compatible set `K_f(N)` and abstains otherwise; with up to `e` further
  relabellings the guard becomes `N_c > f + 2e + max_{d≠c} N_d`.
- **Proposition 4 (honest margin).** `c*` survives every append of at most `f` labels **iff**
  `H_{c*} - max_{d≠c*} H_d > 2f` — one `f` consumed by the attack, one required by the certificate.
  This is necessary for *every* rule sound for honest-plurality validity.
- **Theorem 7 (dispersion-limited resilience).** A sound observer certifies the honest mode with
  probability → 1 iff the fault fraction is below `Δ/(2+Δ)`, where `Δ` is the honest top-two margin.
  The classical `1/3` is the deterministic case `Δ = 1`, and `β* ≤ 1/3` always.
- **Shared-context proposition.** A context shared by the agents makes `A(h) = E_Z Σ_c p_Z(c)^h`, so
  the threshold itself becomes random.

## Repository layout

```
.
├── README.md                     # this file
└── experiments/
    ├── README.md                 # experiment index, data policy, provenance, v15 changelog
    ├── env.sh                    # sets PYTHONPATH for cross-folder imports
    ├── shared/                   # online-collection utilities (API-calling)
    │   ├── collection_manifest.json
    │   └── code/                 # common.py, measure_pmax.py, benign_debate.py, judge prompt, ...
    ├── baseline/                 # main activation/prediction/interaction/instrument analyses
    │   ├── data/                 # tasks.jsonl, samples.jsonl, samples2.jsonl, instrument2.json
    │   ├── code/                 # scoring.py, recompute_e1.py, certify.py, sampling.py, ...
    │   └── results/              # e1_analysis.json, prediction.json, theory_checks.json, ...
    ├── E1_human_calibration/     # E1 human + instrument calibration
    │   ├── annotation/           # blind pairs, three annotator sheets, returned sheets, admin key
    │   ├── code/                 # scoring, normalisation, supplementary checks, embedding sweep
    │   └── results/              # annotation_scoring.json, e1_calibration_instruments.json, ...
    ├── E2_temperature/           # E2 temperature sensitivity (T = 0.3 / 0.7 / 1.0)
    ├── E3_mixed_group/           # E3 mixed-model groups (3/3/4 slots)
    ├── E4_output_form_ablation/  # E4 output-form ablation (canonicalisation)
    ├── E5_certification/         # E5 certification + estimation sensitivity
    │   ├── code/                 # split-half, Dirichlet, Hoeffding, resilience threshold, ...
    │   ├── legacy/               # pre-revision threshold analysis (not used)
    │   └── results/              # submission_certification_check.json, resilience_threshold.json, ...
    ├── E6_attacks/               # E6 forced and prompted attackers
    │   ├── data/nano_prompted/   # raw prompted-attacker transcripts
    │   ├── code/                 # attack runners and analysis
    │   └── results/              # forced/prompted analyses + transport-retry audit
    ├── X2_external_checker/      # external-checker case study (MBPP)
    ├── R1b_operation_partition/  # operation-grain partition annotation (in progress)
    │   ├── annotation/           # sheets, ANNOTATION_GUIDE.md, hidden key
    │   ├── code/                 # build_sheets.py, score_partitions.py (--selftest)
    │   └── results/              # r1b_selftest.json
    └── further_vendors/          # further-vendor sensitivity (DeepSeek v3.2, GLM 4.7, Qwen3 14B/8B)
```

## Experiment index

| Directory | Experiment | What it contains |
|---|---|---|
| `baseline/` | Main activation, prediction, interaction, instrument analyses | Three-model source replies, task catalogue, corrected scoring, headline analysis, prediction, debate checks, instrument heterogeneity |
| `E1_human_calibration/` | E1 human calibration and instrument calibration | Blind pairs, original and normalised annotations, scoring code, unit tests, audit results, embedding and LLM-judge results |
| `E2_temperature/` | E2 temperature sensitivity | Temperature 0.3 and 0.7 samples, the T = 1.0 comparison, k = 16 control, analysis code |
| `E3_mixed_group/` | E3 mixed-model groups | Mixed 3/3/4 collection, raw responses, prediction, analysis |
| `E4_output_form_ablation/` | E4 output-form ablation | Canonicalisation / format-ablation collection code, raw replies, analyses |
| `E5_certification/` | E5 certification and estimation sensitivity | Main certification calculation, split-half and Dirichlet checks, verification code |
| `E6_attacks/` | E6 forced and prompted attackers | Nano and Opus attack code, complete transcripts, transport-retry audit, analyses |
| `X2_external_checker/` | External-checker case study | Verifier collection code, archived candidate records, analysis |
| `R1b_operation_partition/` | Operation-grain partition annotation | Sheets, guide, hidden key, scorer with self-test; **annotation pending** |
| `further_vendors/` | Further-vendor sensitivity | Raw and recomputed results for DeepSeek v3.2, GLM 4.7, Qwen3 14B, and the near-deterministic Qwen3 8B run |
| `shared/` | Shared online-collection utilities | Prompt parsing, debate utilities, judge prompt, collection manifest, reusable collection code |

## Headline results

Every value below is read from a stored result file; the "source" column names it. Numbers quoted in
the manuscript's abstract and prose are marked *(manuscript)*.

### Q1 — Activation is a collision event

| Result | Value | Source |
|---|---|---|
| Single-agent samples predict live-group unanimity, homogeneous group | `gpt-4.1-nano`, 50 tasks, **no fitted parameter**: max abs. gap **0.0099**, mean abs. gap **0.0072** | `baseline/results/prediction.json` |
| Same prediction, mixed group (3/3/4 slots) | predicted **0.7665** vs observed **0.7773** (N = 247, residual **0.0108**) | `E3_mixed_group/results/e3_mixed_analysis.json` |
| Temperature sensitivity, accuracy | T = 0.3 → **0.8877**, T = 0.7 → **0.8846**, T = 1.0 → **0.8724** (123 tasks) | `E2_temperature/results/e2_temperature_analysis.json` |
| k = 16 control (T = 1.0) | answer grain: A(3) = **0.8539**, coverage at n=10,f=3 = **0.7980**, at n=13,f=3 = **0.8482** | `E2_temperature/results/e2_temperature_k16_control.json` |
| Output-form ablation, `claude-opus-5`, k = 8, 50 tasks | free reasoning acc **0.95** / A(3) **0.9829**; reasoning+schema **0.93** / **0.9579**; schema only **0.9325** / **0.9514** | `E4_output_form_ablation/results/x1_canonicalisation.recomputed.json` |
| Cost of requesting the KEY | **−2.0** accuracy points (95% CI [0.0, 5.3]) | *(manuscript)* |
| Instrument pair counts, 3 models | **174,906** pairs, falling-factorial U-statistic, seed 20260911 | `baseline/results/instruments_heterogeneity.json` |

### Q2 — What an observer can certify

| Result | Value | Source |
|---|---|---|
| Answer-grain coverage at n=10, f=3 | plugin **0.8425**, split-half **0.8393**, Dirichlet **0.7602** | `E5_certification/results/e5_certification_sensitivity.json` |
| Same-grain coverage at f = 0 / 1 / 2 | **0.9702 / 0.9481 / 0.8969** (plugin) | idem |
| Exact-string (lexical) grain coverage at f = 3 | plugin **0.00389** ≈ 0.004 — mostly an instrument artefact | idem |
| Conservative `β*` (5% Dirichlet-posterior quantile of Δ), answer grain | 369 cells, mean **0.2667**; **87.3%** of cells > 0.2, **73.7%** > 0.3 | `E5_certification/results/conservative_beta_star.json` |
| Same, lexical grain | mean **0.0057**; only **0.27%** of cells > 0.2 | idem |
| Dependence ρ from round-0 debates | **0.0825** (50 tasks) | `E5_certification/results/supplementary_certification_numbers.json` |
| External checker (MBPP) | 300 rounds, 30 tasks, pooled α = **0.8**, soundness holds; 2 tasks never solved, 18 always solved | `X2_external_checker/results/x2_verifier_analysis.json` |
| Theory status | all 11 formal statements `pass` | `E5_certification/results/paper_theory_checks.json` |

### Q3 — Which relation do the instruments compute?

| Result | Value | Source |
|---|---|---|
| Human reference sample | **300** pairs drawn from a **518,837**-pair frame; 122 tasks, 113 same-model pairs | `E1_human_calibration/results/annotation_scoring.json` |
| Annotator agreement | answer axis raw **0.98** / κ **0.947**; decisive-operation axis raw **0.693** / κ **0.532** | idem |
| Lexical rule false-rejection rate vs the human relation | **0.9669** (95% CI [0.9628, 0.9701]) — i.e. it splits ~97% of human-equivalent pairs | idem |
| Exact-KEY rule false-rejection rate | **0.9679** (95% CI [0.9640, 0.9710]) | idem |
| Embedding instrument at cosine 0.9 | false accept **0.0093**, false reject **0.7225**; equal-error threshold **0.735** (FA 0.189 / FR 0.197) | `E1_human_calibration/results/embedding_threshold_sweep.json` |
| Operation-grain coverage bound, f = 3 | **[0.06, 0.79]**; same-model same-operation rate ∈ [**0.686**, **0.885**] | `E1_human_calibration/results/operation_grain_bounds.json` |
| LLM judges | track plain answer equality rather than the decisive-operation relation | *(manuscript)* |

### Further vendors

| Model | p_max (answer) | A(3) | A(10) | Accuracy | n tasks | Source |
|---|---|---|---|---|---|---|
| `qwen3-8b` (near-deterministic) | **0.9626** | 0.9269 | 0.8902 | 0.8005 | 150 | `further_vendors/results/e1_crossvendor_recomputed.json` |
| `deepseek-v3.2` | 0.9542 | 0.9008 | 0.8202 | 0.9094 | 148 | idem |
| `glm-4.7` | 0.9584 | 0.9043 | 0.8279 | 0.9161 | 150 | idem |
| `qwen3-14b` | 0.9473 | 0.8823 | 0.8186 | 0.8386 | 150 | `further_vendors/results/e1_qwen14b_recomputed.json` |

Sampler diversity: `qwen3-8b` produces a **single distinct reply opening** on **115 of 150** tasks
(mean distinct openings 1.25), versus 15.29 / 13.82 / 9.20 for DeepSeek v3.2 / GLM 4.7 / Qwen3 14B —
the near-deterministic regime is an artifact of the sampler, and `A(h)` alone does not reveal it
(`further_vendors/results/sampler_diversity.json`).

### Attackers

| Result | Value | Source |
|---|---|---|
| Round-2 prompted attackers carry out their assigned attack | **3.5–78%**, depending on the model | *(manuscript)* |
| Forced attackers (nano) | target adoption **1.0** in every f/repetition cell | `E6_attacks/results/e6_forced_nano_analysis.json` |
| Prompted attackers (nano) | target adoption **0.0–0.097** across cells — far below the forced arm | `E6_attacks/results/matched_prompted_analysis.json` |
| Cross-model role compliance | forced **1.0**; prompted cells as low as **~0.02–0.04** | `E6_attacks/results/x3_attacks_analysis.json` |
| Transport audit | 16 completed-after-retry attempts, 0 aborted, 292 failed slots | `E6_attacks/results/matched_prompted_analysis.json` |

## Data policy

- `baseline/data/tasks.jsonl` — the common task catalogue (123 common tasks; 369 model–task cells).
- `baseline/data/samples.jsonl`, `samples2.jsonl` — the three stored model reply sets
  (`gpt-4.1-nano`, `gpt-4o-mini`, `gpt-4.1-mini`; `min_valid_per_task = 10`).
- `baseline/data/instrument2.json` — the archived 800-pair instrument decisions.
- `baseline/data/interaction/e3_benign.json` — raw homogeneous benign-interaction transcripts.
- `E6_attacks/data/nano_prompted/e4_byz.json` — raw prompted-attacker transcripts for the nano arm.
- `shared/collection_manifest.json` — collection parameters (seed 20260919, 300 pairs, 204 jointly
  resolved, k = 16, T ∈ {0.3, 0.7}, benchmarks GSM8K 50 / MBPP 47 / MMLU 26) and stated limitations.
- Original source directories were **not modified** when files were copied here.
- **API credential files are not included.**

## Reproduction

### Offline analyses (no API calls, ~5 minutes)

From the repository root:

```bash
source experiments/env.sh        # sets PYTHONPATH for cross-folder imports
```

In the full submission bundle a top-level `reproduce.sh` runs every offline analysis below in order,
and `audit/verify_paper_numbers.py path/to/iclr-15.tex` asserts the manuscript's numbers against the
result files and rejects superseded values. If your checkout does not contain those wrappers, run the
analyses directly:

| Command | Writes |
|---|---|
| `python3 experiments/baseline/code/submission_debate_check.py` | `baseline/results/submission_debate_check.json` |
| `python3 experiments/baseline/code/analyse_key_length.py` | `baseline/results/key_length_sensitivity.json` |
| `python3 experiments/E1_human_calibration/code/run_calibration.py` | E1 calibration outputs |
| `python3 experiments/E1_human_calibration/code/supplementary_checks.py` | 800-pair intervals, sensitivity, sampling frame |
| `python3 experiments/E1_human_calibration/code/operation_grain_bounds.py` | `operation_grain_bounds.json` |
| `python3 experiments/E2_temperature/code/e2_analyse_temperature.py` | `e2_temperature_analysis.json`, k=16 control |
| `python3 experiments/E3_mixed_group/code/e3_predict_mixture.py experiments/E3_mixed_group/results/e3_mixed.json` | `e3_mixed_analysis.json` |
| `python3 experiments/E4_output_form_ablation/code/analyse_canonicalisation.py` | `x1_canonicalisation.recomputed.json` |
| `python3 experiments/E5_certification/code/submission_certification_check.py` | certification checks |
| `python3 experiments/E5_certification/code/e5_split_half_certification.py` | split-half sensitivity |
| `python3 experiments/E5_certification/code/check_paper_theory.py` | `paper_theory_checks.json` |
| `python3 experiments/E5_certification/code/supplementary_certification_numbers.py` | Hoeffding vs exact, Dirichlet, ρ |
| `python3 experiments/E5_certification/code/resilience_threshold.py` | `resilience_threshold.json` |
| `python3 experiments/E5_certification/code/conservative_beta_star.py` | `conservative_beta_star.json` |
| `python3 experiments/E1_human_calibration/code/embedding_threshold_sweep.py` | `embedding_threshold_sweep.json` |
| `python3 experiments/R1b_operation_partition/code/score_partitions.py --selftest` | `r1b_selftest.json` |
| `python3 experiments/E6_attacks/code/e6_analyse_forced.py experiments/E6_attacks/results/e6_forced_nano.json experiments/E6_attacks/data/nano_prompted/e4_byz.json` | forced-attack analysis |
| `python3 experiments/E6_attacks/code/analyse_matched_prompted.py` | `matched_prompted_analysis.json` |
| `python3 experiments/further_vendors/code/sampler_diversity.py` | `sampler_diversity.json` |

### Scripts that call model APIs (credentials required)

These are the collectors. They need API credentials that are **not** shipped here, and they spend
tokens, so they are not part of `reproduce.sh`:

- `experiments/shared/code/` — shared collection utilities and the debate runner
- `experiments/E6_attacks/code/exp_attacks.py` — attack collection
- `experiments/X2_external_checker/code/exp_verifier_e2e.py` — external-checker collection
- `experiments/E4_output_form_ablation/code/exp_canonicalisation.py` — output-form collection
- `experiments/E1_human_calibration/code/e1_calibration_instruments.py` — judge/embedding instruments

### A note on `env.sh`

Archived files whose SHA-256 values are recorded in `baseline/results/revision_analysis.json` are kept
**byte-identical**. That is why cross-folder imports are resolved through `experiments/env.sh` rather
than by editing those scripts' import statements. Always `source experiments/env.sh` first.

## Provenance and integrity checks

- The baseline inputs, legacy analysis artifacts, and dependency modules were copied from `llmbft`;
  their SHA-256 values match the source hashes recorded in `baseline/results/revision_analysis.json`.
- The Opus prompted experiment uses the **complete 144-debate** file from `llmbft/v5`, not the
  incomplete 55-record copy that was previously present.
  `E6_attacks/results/matched_prompted_analysis.json` records the exact hashes of the complete
  transcripts, the transport-failure audit, the task catalogue, the scorer, and the analysis script.
- `baseline/results/key_length_sensitivity.json` and `shared/collection_manifest.json` likewise record
  `sources_sha256` for the scorer, normaliser, and sample files they depend on.

## What is not included

- Model **API credentials** and environment files.
- **Paper source** (LaTeX, `.bbl`, class/style files) — the manuscript is a separate deliverable.
- **Figure-only plotting scripts** and the top-level `figures/` directory
  (`make_fig_instruments.py`, `figure1.tex`) — these live in the submission bundle, not here.
- `reproduce.sh` and `audit/verify_paper_numbers.py` — the submission bundle's top-level wrappers
  around this `experiments/` tree.
- Caches, build logs, and editor/OS junk (see `.gitignore`).

> If you need the wrappers or the figure scripts alongside the analyses, take them from the full
> supplementary bundle; the `experiments/` tree here is byte-identical to the one it drives.

## Legacy and superseded files

Kept only for provenance — **no reported number depends on them**:

- `E5_certification/legacy/threshold_analysis.py` (+ `.json`): pre-revision analysis with `h` fixed at
  10 and an `f` (not `2f`) guard; not used by the manuscript and not runnable from this layout.
- `E4_output_form_ablation/legacy/x1_canon_opus5.partial.json`: stale partial summary (709 calls). The
  manuscript uses `results/x1_canonicalisation.json`, reproduced by `analyse_canonicalisation.py`.
- `baseline/code/certify.py::coverage_hoeffding_bound` and the labels in `baseline/code/check_theory.py`
  predate the honest-margin (`2f`) statement and use old theorem numbers. `check_paper_theory.py`
  checks the current statements instead, including the `(m-1)`-term Hoeffding bound with `2f`.
- Legacy `bootstrap_ci.json` and `correlation_fit.json` were not copied; their analyses are superseded.
- An empty `x1_canon.samples.jsonl` was removed.

## Scope and caveats

The paper's own framing, repeated here so the package is not over-read:

- We do **not** claim that language-model groups cannot reach agreement.
- The certificate **bounds every sound rule**, but only under **one label law and a common view**;
  it does not bound the fault tolerance of arbitrary protocols.
- The **lexical grain does not measure reasoning**. It records a stated summary string, and ~97% of
  human-equivalent pairs fail exact string equality.
- `qwen3-8b`'s very high `A(h)` is a **sampler** artifact (near-deterministic decoding), not evidence
  of a more reliable agent.
- `R1b_operation_partition/` is **annotation pending**: it ships the sheets, guide, hidden key, and a
  self-tested scorer, but no consensus human partition yet.
