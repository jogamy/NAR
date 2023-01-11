CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --train_mode constrainer \
  --model_path /root/NAR/examples/KMA/model/sejong/v4/last.ckpt \
  --task KMA --dataset sejong \
  --max_epochs 100 --batch_size 128 \
  --num_workers 8 --lr 5e-4 \
  --devices 2 \
  --warmup_ratio 0.05 \
  --max_len 200 \
  --d_model 512 --feedforward 2048 \
  --dropout 0.3 \
  --lp_structure eojeol --lp_max_length 111 \
  --default_root_dir v4
  
  