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