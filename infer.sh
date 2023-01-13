CUDA_VISIBLE_DEVICES=1 python infer.py \
    --hparams /root/NAR/examples/KMA/constrainer/sejong/v4/tb_logs/lightning_logs/version_0/hparams.yaml \
    --model_path /root/NAR/examples/KMA/constrainer/sejong/v4/last.ckpt \
    --outputfile infer.txt --beam_size 5 