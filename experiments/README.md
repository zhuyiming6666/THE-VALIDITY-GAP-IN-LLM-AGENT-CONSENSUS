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

## Reproduction notes

Some archived scripts retain paths from the original `llmbft` working tree.
The files are grouped here by experiment for review and preservation; path
normalization into a new one-command runner remains separate work. Scripts that
compile or edit the manuscript were deliberately excluded.

The temperature experiment retains its per-task count summaries and final
analysis, but the original `temp_T0.3.samples.jsonl` and
`temp_T0.7.samples.jsonl` reply-level dumps have not been located. No other
manuscript experiment is currently known to lack its principal raw or result
artifact.

Legacy `bootstrap_ci.json` and `correlation_fit.json` were not copied. Their
analyses are superseded or no longer reported in the current manuscript.
