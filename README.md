# Structure

```
NAR
├── README.md
├── examples
│   ├── KMA
│   │   ├── sejong
│   │   │   ├── datamodule.py
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   └── valid.txt
│   │   └── utils
│   │       ├── eval.py
│   │       └── util.py
│   ├── MIXSLU
│   │   ├── MixATIS
│   │   │   └── datamodule.py
│   │   ├── MixSNIPS
│   │   │   └── datamodule.py
│   │   └── utils
│   │       ├── eval.py
│   │       └── util.py
│   ├── NER
│   │   ├── conll2003
│   │   │   └── datamodule.py
│   │   └── utils
│   │       ├── eval.py
│   │       └── util.py
│   ├── SLU
│   │   ├── atis
│   │   │   ├── datamodule.py
│   │   │   ├── dev.txt
│   │   │   ├── test.txt
│   │   │   └── train.txt
│   │   ├── snips
│   │   │   ├── datamodule.py
│   │   │   ├── dev.txt
│   │   │   ├── test.txt
│   │   │   └── train.txt
│   │   └── utils
│   │       ├── eval.py
│   │       └── util.py
│   ├── __init__.py
│   ├── datamodule.py
│   └── utils
│       ├── mytokenizer.py
│       └── util.py
├── nn
│   ├── __init__.py
│   ├── constrainer.py
│   ├── custom_nar_wrapper.py
│   ├── lengthpredictor.py
│   ├── model_templates.py
│   └── nar_wrapper.py
├── eval.py
├── infer.py
├── train.py

```
---------
# Arguments

### Arguments for task

|argument|available|
| ------ | ------- |
|train_mode|🔲 model  🔲 constrainer |
|task | 🔲 SLU 🔲 NER  🔲 KMA 🔲 MIXSLU |
|dataset| 🔲 snips  🔲 atis  🔲 mixsnips 🔲 mixatis 🔲 conll2003 🔲 sejong |
|lp_structure | 🔲 cmlm 🔲 ctc 🔲 eojeol 🔲 fertility 🔲 labeling|
|train_logic |🔲 uniform 🔲 full 🔲 eojeol |

### Arguments for neural network
|argument|available|
| ------ | ------- |
| enc_n_layers | - |
| dec_n_layers | - |
| d_model | - |
| max_len | - |
| drop_out | - |

### Arguments for training
|argument|available|
| ------ | ------- |
| epochs | - |
| batch_size | - |
| lr | - |
| num_workers | - |
| devices | - |


-------------------

# Training

examples for NER

```
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
```

|argument|available|
| ------ | ------- |
|train_mode|🔳 model  🔲 constrainer |
|task | 🔲 SLU 🔳 NER  🔲 KMA 🔲 MIXSLU |
|dataset| 🔲 snips  🔲 atis  🔲 mixsnips 🔲 mixatis 🔳 conll2003 🔲 sejong |
|lp_structure | 🔲 cmlm 🔲 ctc 🔲 eojeol 🔲 fertility 🔳 labeling|
|train_logic |🔲 uniform 🔲 full 🔲 eojeol |


----------------

# Inference


```
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --hparams /root/NAR/examples/NER/conll2003/model/v1/tb_logs/lightning_logs/version_0/hparams.yaml \
    --model_path /root/NAR/examples/NER/conll2003/model/v1/last.ckpt
```