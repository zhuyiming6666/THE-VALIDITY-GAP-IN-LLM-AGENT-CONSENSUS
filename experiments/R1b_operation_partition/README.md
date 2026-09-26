# R1b: partition annotation at the decisive-operation grain

Purpose: replace the wide pairwise bound on operation-grain coverage (f=3: [0.06, 0.79], Appendix G)
by a point estimate. Annotators group, within each task, 12 replies of one model by decisive
operation; the groups form a partition, so r_max, A(h) and the n=3f+1 coverage r_max^h follow directly.

| file | role |
|---|---|
| `code/build_sheets.py` | selects 24 model-task cells (8 per benchmark, quartiles of answer-grain p_max, models in rotation) and 12 replies per cell (seed 20260925); writes the sheets |
| `annotation/ANNOTATION_GUIDE.md` | instructions (same SAME/DIFF rules as the pairwise calibration) |
| `annotation/tasks.tsv` | question text for T01-T24 |
| `annotation/annotator_{A,B,C}.tsv` | identical blank sheets, 288 rows; fill `group`, `vague`, `note` |
| `annotation/admin/key.json` | hidden mapping to model, task and source line; do not distribute |
| `code/score_partitions.py` | consensus partition (>=2 of 3 annotators), per-grain r_max, A(3), coverage h=3,5,7, ARI; `--selftest` checks the pipeline against the answer and lexical partitions |

Expected effort: 1.5-2 hours per annotator. The 24 cells reproduce the answer-grain f=3 coverage of the full
common set (0.843), so they are representative for the mean.

Distribution packages (guide + tasks + one sheet per annotator, no `admin/`) are in `grain-iclr15/docs/R1b/R1b_annotation_package_{A,B,C}.zip`.
