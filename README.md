# Structure

```
NAR
├── README.md
├── examples
│   ├── KMA
│   │   ├── sejong
│   │   │   ├── datamodule.py
│   │   │   ├── model
│   │   │   │   ├── eojeol
│   │   │   │   └── len_token
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
│   │   │   ├── datamodule.py
│   │   │   └── model
│   │   │       ├── b64_e3_d128
│   │   │       ├── b64_e3_d256
│   │   │       ├── b64_e4_d128
│   │   │       ├── b64_e4_d256
│   │   │       ├── v1
│   │   │       └── v2
│   │   └── utils
│   │       ├── eval.py
│   │       └── util.py
│   ├── SLU
│   │   ├── atis
│   │   │   ├── datamodule.py
│   │   │   ├── constrainer
│   │   │   │   ├── v1
│   │   │   │   └── v6
│   │   │   ├── model
│   │   │   │   ├── v1
│   │   │   │   └── v6
│   │   │   ├── dev.txt
│   │   │   ├── test.txt
│   │   │   └── train.txt
│   │   ├── snips
│   │   │   ├── datamodule.py
│   │   │   ├── constrainer
│   │   │   │   └── v1
│   │   │   ├── model
│   │   │   │   ├── plm
│   │   │   │   ├── v1
│   │   │   │   └── v6
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