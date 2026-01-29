python -m src.accel.normative.compute_residuals \
  --emb_dir outputs/embeddings_v0 \
  --covars_csv /home/aakarsh/pd_learn/data/files/acc_qced.csv \
  --model_path outputs/normative_v0/normative_model.joblib \
  --outdir outputs/residuals_v0 \
  --id_col "Participant ID"