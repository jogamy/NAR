# Requirements

  - ordered-set==4.1.0
  - pytorch-lightning==1.8.6
  - torch==1.13.0
  - torchmetrics==0.11.0
  - torchtext==0.14.0
  - torchvision==0.14.0
  - einops==0.6.0
  - tokenizers==0.13.2
  - datasets==2.8.0
  - transformers==4.25.1

  pip install -r requirements.txt
    
-------------------------------------
# Structure
```
NAR
|-- README.md
|-- examples
|   |-- KMA
|   |   |-- datamodule.py
|   |   |-- sejong
|   |-- MixSLU(Work in progress)
|   |-- NER
|   |   |-- datamodule.py
|   |   |-- CONLL2003
|   |-- __init__.py
|   `-- mytokenizer.py
|-- infer.py(Work in progress)
|-- requirements.txt
|-- train.py
`-- x_transformers
    |-- __init__.py
    |-- autoregressive_wrapper.py
    |-- constrainer.py
    |-- continuous_autoregressive_wrapper.py
    |-- lengthpredictor.py
    |-- non_autoregressive_wrapper.py
    `-- x_transformers.py
```

examples: tasks

x_transformers: model tamplates

x_transforemrs: https://github.com/lucidrains/x-transformers

-------------------------------------
# Training

Argument details

    --train_mode    model/constrainer/model and constrainer
    --task          KMA/SLU/NER
    --dataset       sejong/MixATIS/CONLL2003
    --train_logic   random/uniform/eojeol
    --lp_structure  len_token/eojeol

## Korean morphological anaylsis
```
cd NAR
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --train_mode model \
  --task KMA --dataset sejong \
  --max_epochs 100 --batch_size 128 \
  --num_workers 8 --lr 5e-4 \
  --devices 2 \
  --warmup_ratio 0.05 \
  --max_len 200 \
  --d_model 512 --feedforward 2048 \
  --dropout 0.3 \
  --train_logic eojeol \
  --lp_structure eojeol --lp_max_length 111 \
  --default_root_dir v4
```

Constrainer 학습
```
cd NAR
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
```

## NER
'''
cd NAR
CUDA_VISIBLE_DEVICES=1 python train.py \
  --train_mode model \
  --task NER --dataset CONLL2003 \
  --max_epochs 100 --batch_size 128 \
  --num_workers 8 --lr 5e-4 \
  --devices 1 \
  --warmup_ratio 0.05 \
  --max_len 200 \
  --d_model 768 --feedforward 2048 \
  --dropout 0.3 \
  --plm_path bert-base-uncased \
  --default_root_dir v4
'''

----------------------------------
# Inference(WIP)
```
cd NAR
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --hparams /root/NAR/examples/KMA/model/sejong/v4/tb_logs/lightning_logs/version_0/hparams.yaml \
    --model_path /root/NAR/examples/KMA/model/sejong/v4/last.ckpt \
    --outputfile infer.txt --beam_size 5 
```