#!/bin/bash
# Mark in blue everything in the rebuttal paper files that differs from the submitted version.
#   bash experiments/paper_mark_all_blue.sh <submitted folder> <rebuttal folder> [output folder, default = rebuttal folder]
# Submitted version: main_EG.tex, supplemental_EG.tex and the tables they input (tables_v4/).
# Safe to run again after edits: existing blue marks are removed first.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SUB=${1:?submitted folder}; REB=${2:?rebuttal folder}; OUT=${3:-$REB}
mkdir -p "$OUT/tables_rebuttal"
m() { python3 "$HERE/paper_mark_changes_blue.py" "$1" "$REB/$2" -o "$OUT/$2"; }
m "$SUB/main_EG.tex"                                  main_EG_rebuttal_11.10.tex
m "$SUB/main_EG.tex"                                  main_EG_rebuttal_full_11.10.tex
m "$SUB/supplemental_EG.tex"                          supplemental_EG_rebuttal_11.10.tex
m "$SUB/supplemental_EG.tex"                          supplemental_EG_rebuttal_full_11.10.tex
m "$SUB/tables_v4/quantitative_comparison_table.tex"      tables_rebuttal/quantitative_comparison_table.tex
m "$SUB/tables_v4/quantitative_comparison_full_table.tex" tables_rebuttal/quantitative_comparison_full_table.tex
m "$SUB/tables_v4/ablations_table.tex"                    tables_rebuttal/ablations_table_full.tex
m -                                                       tables_rebuttal/quantitative_comparison_ddim25_table.tex
