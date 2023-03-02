# # snips constrainer
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode constrainer \
#   --model_path /root/NAR/examples/SLU/snips/model/v6/last.ckpt \
#   --task SLU --dataset snips \
#   --max_epochs 100 --batch_size 32 \
#   --num_workers 8 --lr 1e-3 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 50 \
#   --enc_n_layers 6 --dec_n_layers 1 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --default_root_dir v6

# # snips PLM
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode model \
#   --task SLU --dataset snips \
#   --max_epochs 100 --batch_size 32 \
#   --num_workers 8 --lr 1e-3 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 50 \
#   --enc_plm bert-base-uncased \
#   --enc_n_layers 6 --dec_n_layers 1 \
#   --d_model 768 --feedforward 2048 \
#   --dropout 0.3 \
#   --default_root_dir plm

# # ATIS
# # const
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode constrainer \
#   --model_path /root/NAR/examples/SLU/atis/model/v6/last.ckpt \
#   --task SLU --dataset atis \
#   --max_epochs 100 --batch_size 32 \
#   --num_workers 8 --lr 1e-3 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 50 \
#   --enc_n_layers 6 --dec_n_layers 1 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --default_root_dir v6
  
# # ATIS
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode model \
#   --task SLU --dataset atis \
#   --max_epochs 100 --batch_size 32 \
#   --num_workers 8 --lr 1e-3 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 50 \
#   --enc_n_layers 6 --dec_n_layers 1 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --default_root_dir v6


# NER
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_mode model \
  --task NER --dataset conll2003 \
  --max_epochs 200 --batch_size 64 \
  --num_workers 8 --lr 5e-4 \
  --devices 1 \
  --warmup_ratio 0.05 \
  --max_len 200 \
  --enc_n_layers 3 \
  --n_heads 4 \
  --d_model 128 \
  --dropout 0.3 \
  --default_root_dir b64_e3_d128

# # NER PLM
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode model \
#   --task NER --dataset conll2003 \
#   --max_epochs 400 --batch_size 32 \
#   --num_workers 8 --lr 5e-4 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 200 \
#   --enc_n_layers 1 \
#   --d_model 768 --feedforward 2048 \
#   --dropout 0.3 \
#   --enc_plm bert-base-uncased \
#   --default_root_dir plm

# # len token constrianer
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode constrainer \
#   --model_path /root/NAR/examples/KMA/sejong/model/len_token/last.ckpt \
#   --task KMA --dataset sejong \
#   --max_epochs 100 --batch_size 2 \
#   --num_workers 8 --lr 5e-4 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 200 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --lp_structure len_token --lp_max_length 200 \
#   --default_root_dir len_token

# # kma len token
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode model \
#   --task KMA --dataset sejong \
#   --max_epochs 100 --batch_size 32 \
#   --num_workers 8 --lr 5e-4 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 200 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --lp_structure len_token --lp_max_length 200 \
#   --default_root_dir len_token

# # eojeol constrainer
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode constrainer \
#   --model_path /root/NAR/examples/KMA/sejong/model/eojeol/last.ckpt \
#   --task KMA --dataset sejong \
#   --max_epochs 100 --batch_size 2 \
#   --num_workers 8 --lr 5e-4 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 200 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --train_logic eojeol \
#   --lp_structure eojeol --lp_max_length 111 \
#   --default_root_dir eojeol

# # eojeol
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --train_mode model \
#   --task KMA --dataset sejong \
#   --max_epochs 100 --batch_size 32 \
#   --num_workers 8 --lr 5e-4 \
#   --devices 1 \
#   --warmup_ratio 0.05 \
#   --max_len 200 \
#   --d_model 512 --feedforward 2048 \
#   --dropout 0.3 \
#   --train_logic eojeol \
#   --lp_structure eojeol --lp_max_length 111 \
#   --default_root_dir eojeol