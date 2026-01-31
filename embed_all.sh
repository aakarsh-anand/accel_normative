python -m src.accel.embed_all \
  --config configs/ssl_mae_byol.yaml \
  --ckpt checkpoints/ssl/last.ckpt \
  --outdir outputs/embeddings_v1 \
  --windows_per_subject 16 \
  --batch_size 32 \
  --num_workers 8 \
  --save_window_embeddings