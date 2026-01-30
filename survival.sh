# python -m src.disease.survival \
#   --pd_labels_full_csv outputs/pd_labels.csv \
#   --score_csv outputs/pd_axis_v0/pd_axis_scores.csv \
#   --score_col pd_axis_proj \
#   --pull_date 2025-11-01 \
#   --outdir outputs/pd_survival_v0 \
#   --ymin 0.985

python -m src.disease.make_labels "$@"