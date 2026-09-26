# Source from the repository root before running any script:
#     source experiments/env.sh
# Scripts import shared modules across experiment folders (scoring, collision,
# sampling, certify, annotation_normalization, ...).  The archived files are kept
# byte-identical to their recorded SHA-256 values, so the import path is set here
# instead of editing each script.
_GRAIN_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$_GRAIN_EXP/baseline/code:$_GRAIN_EXP/shared/code:$_GRAIN_EXP/E1_human_calibration/code:$_GRAIN_EXP/E5_certification/code${PYTHONPATH:+:$PYTHONPATH}"
