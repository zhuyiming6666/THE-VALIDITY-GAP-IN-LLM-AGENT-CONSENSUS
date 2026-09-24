# Further Vendor Sensitivity

This directory contains the four additional-model runs reported in the
manuscript's further-vendors table:

- `deepseek-v3.2`, `glm-4.7`, and `qwen3-8b` are stored in
  `results/e1_crossvendor*`.
- `qwen3-14b` is stored in `results/e1_qwen14b*`.

The two `.samples.jsonl` files contain 9,600 reply-level records in total. The
matching `.json` files contain collection summaries, and the
`*_recomputed.json` files contain the table statistics. Collection and
recomputation use the shared `measure_pmax.py` and baseline `recompute_e1.py`
implementations elsewhere in this experiment package.
