python -m src.accel.probes \
  --emb_dir outputs/embeddings_v0 \
  --covars_csv /home/aakarsh/pd_learn/data/files/acc_qced.csv \
  --id_col "Participant ID" \
  --age_col "Age" \
  --sex_col "Sex" \
  --accmean_col "Overall acceleration average"