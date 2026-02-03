# python -m src.accel.normative.fit_normative \
#   --emb_dir outputs/embeddings_v0 \
#   --covars_csv /home/aakarsh/pd_learn/data/files/acc_qced.csv \
#   --outdir outputs/normative_v0 \
#   --id_col "Participant ID" \
#   --age_col "Age" \
#   --sex_col "Sex" \
#   --accmean_col "Overall acceleration average" \
#   --alpha 10 \
#   --n_knots 8 \
#   --degree 3

python -m src.normative.fit_normative \
  --emb_dir outputs/embeddings_v1 \
  --covars_csv /home/aakarsh/pd_learn/data/files/acc_qced.csv \
  --wear_dates_csv outputs/wear_dates.csv \
  --outdir outputs/normative \
  --id_col "Participant ID" \
  --age_col "Age" \
  --sex_col "Sex" \
  --height_col "height" \
  --townsend_col "townsend" \
  --smoking_col "smoking" \
  --accmean_col "Overall acceleration average" \
  --ref_ids_csv outputs/reference_sets/controls_5yr.csv \
  --geom_var 0.99 \
  --cv_alphas "0.1,1,10,100,1000" \
  --n_knots 8 \
  --degree 3
   # --alpha 10