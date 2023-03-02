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
├── infer.sh
├── train.py
└── train.sh

```

-------------------

# Training

examples for NER

```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_mode constrainer \
  --model_path /root/NAR/examples/SLU/snips/model/v6/last.ckpt \
  --task SLU --dataset snips \
  --max_epochs 100 --batch_size 32 \
  --num_workers 8 --lr 1e-3 \
  --devices 1 \
  --warmup_ratio 0.05 \
  --max_len 50 \
  --enc_n_layers 6 --dec_n_layers 1 \
  --d_model 512 --feedforward 2048 \
  --dropout 0.3 \
  --default_root_dir v6
```

--train_mode : [ ] [ ]


----------------