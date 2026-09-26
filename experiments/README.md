# Experiment Index

## Layout

| Directory | Experiment | Contents |
|---|---|---|
| `baseline/` | Main activation, prediction, interaction, and instrument analyses | Three-model source replies, task data, corrected scoring, headline analysis, prediction, debate checks, and instrument heterogeneity |
| `E1_human_calibration/` | E1 human calibration and instrument calibration | Blind pairs, original and normalized annotations, scoring code, tests, audit results, embedding and judge results |
| `E2_temperature/` | E2 temperature sensitivity | Temperature 0.3 and 0.7 samples, the temperature 1.0 comparison, and analysis code |
| `E3_mixed_group/` | E3 mixed-model groups | Mixed 3/3/4 collection, raw responses, prediction, and analysis |
| `E4_output_form_ablation/` | E4 output-form ablation | Canonicalisation/format-ablation collection code, raw replies, and analyses |
| `E5_certification/` | E5 certification and estimation sensitivity | Main certification calculation, split-half and Dirichlet checks, and verification code |
| `E6_attacks/` | E6 forced and prompted attackers | Nano and Opus attack code, complete transcripts, transport retry audit, and analyses |
| `X2_external_checker/` | External checker case study | Verifier collection code, archived candidate records, and analysis |
| `further_vendors/` | Further-vendor sensitivity | Raw and recomputed results for DeepSeek v3.2, GLM 4.7, Qwen3 14B, and the near-deterministic Qwen3 8B run |
| `shared/` | Shared online-collection utilities | Prompt parsing, debate utilities, judge prompt, collection manifest, and reusable collection code |

## Data policy

- `baseline/data/tasks.jsonl` is the common task catalogue.
- `baseline/data/samples.jsonl` and `samples2.jsonl` are the three-model stored replies.
- `baseline/data/instrument2.json` contains the archived 800-pair instrument decisions.
- `baseline/data/interaction/e3_benign.json` contains the raw homogeneous benign-interaction transcripts used by the interaction analysis.
- `E6_attacks/data/nano_prompted/e4_byz.json` contains the raw prompted-attacker transcripts for the nano experiment.
- Original source directories were not modified when files were copied here.
- API credential files are not included.

## Provenance and version checks

The baseline inputs, legacy analysis artifacts, and dependency modules were
copied from `llmbft`. Their SHA-256 values match the source hashes recorded in
`baseline/results/revision_analysis.json`.

The Opus prompted experiment uses the complete 144-debate file from
`llmbft/v5`, not the incomplete 55-record copy previously present here.
`E6_attacks/results/matched_prompted_analysis.json` records the exact hashes of
the complete transcripts, transport-failure audit, task catalogue, scorer, and
analysis script.

## Reproduction

From the repository root:

```bash
source experiments/env.sh        # sets PYTHONPATH for cross-folder imports
bash reproduce.sh                # offline analyses only; no API calls
python audit/verify_paper_numbers.py path/to/iclr-15.tex
```

Scripts that call model APIs (collection code under `shared/`, `E6_attacks/code/exp_attacks.py`,
`X2_external_checker/code/exp_verifier_e2e.py`, `E4_output_form_ablation/code/exp_canonicalisation.py`,
`E1_human_calibration/code/e1_calibration_instruments.py`) need credentials that are not included.
Archived files whose SHA-256 values are recorded in `baseline/results/revision_analysis.json`
are kept byte-identical; that is why imports are resolved through `env.sh` rather than by editing
those scripts.

## Added in the v15 package

| File | Purpose |
|---|---|
| `E2_temperature/results/temp_T0.3.samples.jsonl`, `temp_T0.7.samples.jsonl` | Reply-level dumps of the temperature sweep, recovered from `llmbft/results/v11/` |
| `E2_temperature/code/e2_analyse_temperature.py` | Repository-relative paths; reproduces `e2_temperature_analysis.json` byte for byte; adds the k=16 control for T=1.0 (`e2_temperature_k16_control.json`) and the revised figure |
| `E1_human_calibration/code/supplementary_checks.py` | Regenerates the 800-pair archived-label intervals, annotator/pair-type sensitivity, Table a2 rows, conditional instrument rates on all vs. resolved pairs, and re-derives the 518,837-pair sampling frame and the 148/69/43/40 allocation from the stored replies |
| `further_vendors/code/sampler_diversity.py` | Reply-opening and KEY diversity per vendor; replaces an unarchived qwen3-8b probe |
| `E4_output_form_ablation/code/analyse_canonicalisation.py` | Offline re-analysis of the output-form ablation from the archived claude-opus-5 replies |
| `E5_certification/code/check_paper_theory.py` | Brute-force checks of every formal statement in the manuscript, with the manuscript's numbering |
| `figures/make_fig_instruments.py`, `figures/figure1.tex` | Q3 figure from the current-normaliser rates; TikZ source of Figure 1 (numbers cited in its header) |
| `E1_human_calibration/code/operation_grain_bounds.py` | Bounds on n=3f+1 coverage at the human decisive-operation grain from same-model calibration pairs (Appendix G, tab:op-bounds) |
| `E5_certification/code/supplementary_certification_numbers.py` | Hoeffding bound vs exact coverage, Dirichlet prior-mass sensitivity (closed form), dependence rho from round-0 debates |
| `E5_certification/code/resilience_threshold.py` | Theorem 7 (dispersion-limited resilience): beta* per model-task cell, coverage as n grows, Fig. 5 |
| `E5_certification/code/conservative_beta_star.py` | Conservative beta* from the 5% Dirichlet-posterior quantile of Delta |
| `E1_human_calibration/code/embedding_threshold_sweep.py` | Cosine-threshold sweep of the embedding instrument against the human reference |
| `R1b_operation_partition/` | Partition-annotation materials for the operation grain (sheets, guide, hidden key, scorer with self-test); annotation pending |
| `audit/verify_paper_numbers.py` | Asserts manuscript numbers against the result files and rejects superseded values |

## Legacy and removed files

- `E5_certification/legacy/threshold_analysis.py` (+ `.json`): pre-revision analysis with h fixed at 10
  and an `f` (not `2f`) guard; not used by the manuscript and not runnable from this layout.
- `E4_output_form_ablation/legacy/x1_canon_opus5.partial.json`: stale partial summary (709 calls);
  the manuscript uses `results/x1_canonicalisation.json`, reproduced by `analyse_canonicalisation.py`.
- An empty `x1_canon.samples.jsonl` was removed.
- `baseline/code/certify.py::coverage_hoeffding_bound` and the labels in `baseline/code/check_theory.py`
  predate the honest-margin (`2f`) statement and use old theorem numbers. They are kept unchanged for
  provenance and are not used by any reported number; `check_paper_theory.py` checks the current
  statements, including the (m-1)-term Hoeffding bound with `2f`.
- Legacy `bootstrap_ci.json` and `correlation_fit.json` were not copied; their analyses are superseded.
- Returned annotation sheets live in `E1_human_calibration/annotation/results/` (originally a
  Chinese-named sub-folder of `annotation/` meaning "results"); the source fields in the E1 result
  JSONs were updated accordingly.
