# R1b Annotation Guide: Grouping by "Decisive Operation" within a Task

## 1. Purpose
The paper needs one quantity: the **grouping** (partition) of the "decisive operations" that a single
model produces when it answers the same question several times. The earlier 300-pair calibration was a
**pairwise** judgement; it cannot form a partition and can only bound the f=3 coverage by the wide
interval [0.06, 0.79]. In this round the annotators group the 12 replies to each task directly, which
yields point estimates of r_max, A(h) and the coverage.

## 2. Materials
- `tasks.tsv`: the question text of the 24 tasks (T01–T24).
- `annotator_X.tsv`: one row per reply (288 rows in total), with the columns
  `task, item, answer, key, group, vague, note`. Fill in the last three columns only.
- Model names are not shown, and neither are reference answers.

## 3. How to group (independently within each task)
1. Read the question first, then read through all 12 KEYs of that task.
2. Give every reply a `group` name, for example G1, G2, .... **Replies with the same decisive operation
   get the same group name**; replies with different operations get different group names. Group names
   are meaningful only within their own task: G1 in one task has nothing to do with G1 in another.
3. The SAME/DIFF criteria are exactly those of the first round:
   - The difference lies only in wording, language, notation, or in an expansion that does not change
     the operation → same group.
   - The operation, its operand, or its direction clearly changes → different group. Two adjacent steps
     of one and the same solution must not be merged merely because they are adjacent.
   - Two replies compute different results but state the same operation → they may still share a group
     (the `answer` column records the difference in results separately).
4. If the KEY is empty, merely restates the question, gives only a result, or the operation cannot be
   identified: set `vague` to Y and leave `group` empty.
5. If the level of abstraction differs and you cannot tell whether it is the same operation: pick the
   group you consider most likely and write `unsure` in `note`. Do not create a new group for this.
6. Grouping must be transitive: if A is grouped with B and B with C, then A must be grouped with C. If
   that feels wrong, re-examine those three replies.

## 4. Procedure
1. The three annotators work independently and do not discuss with each other; no session longer than
   1.5 hours. Expect 3–5 minutes per task, about 1.5–2 hours in total.
2. Modify only the `group`, `vague` and `note` columns; do not change the other columns or the row order.
3. Hand the sheet to the coordinator when you are done. The aggregation rules are fixed in advance:
   - two replies are linked in the consensus partition if at least 2 of the 3 annotators put them in the
     same group;
   - the consensus partition is the set of connected components of the link relation;
   - vague replies each form their own group in the main analysis (conservative: this can only lower
     r_max), and are dropped in the sensitivity analysis.

## 5. How the output is used
`code/score_partitions.py` computes, for every cell, the following quantities at three grains — answer,
human operation and lexical: the modal mass r_max, the U-statistic A(3), and the coverage r_max^h at
n=3f+1 (h=3,5,7). These are compared against the R1a bounds. Agreement between annotators is reported
with the adjusted Rand index (ARI) and the pairwise agreement rate.
