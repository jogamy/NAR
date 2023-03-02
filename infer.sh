# # snips constrainer
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --hparams /root/NAR/examples/SLU/snips/constrainer/v1/tb_logs/lightning_logs/version_0/hparams.yaml \
#     --model_path /root/NAR/examples/SLU/snips/constrainer/v1/last.ckpt \

# # snips 
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --hparams /root/NAR/examples/SLU/snips/model/plm/tb_logs/lightning_logs/version_0/hparams.yaml \
#     --model_path /root/NAR/examples/SLU/snips/model/plm/last.ckpt \

# # ATIS constrainer # v6
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --hparams /root/NAR/examples/SLU/atis/constrainer/v6/tb_logs/lightning_logs/version_0/hparams.yaml \
#     --model_path /root/NAR/examples/SLU/atis/constrainer/v6/last.ckpt \

# # ATIS
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --hparams /root/NAR/examples/SLU/atis/model/v6/tb_logs/lightning_logs/version_0/hparams.yaml \
#     --model_path /root/NAR/examples/SLU/atis/model/v6/last.ckpt \


# # KMA len_token
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --hparams /root/NAR/examples/KMA/sejong/model/len_token/tb_logs/lightning_logs/version_0/hparams.yaml \
#     --model_path /root/NAR/examples/KMA/sejong/model/len_token/last.ckpt \
#     --beam_size 5 


# # KMA eojeol
# CUDA_VISIBLE_DEVICES=0 python infer.py \
#     --hparams /root/NAR/examples/KMA/sejong/model/eojeol/tb_logs/lightning_logs/version_0/hparams.yaml \
#     --model_path /root/NAR/examples/KMA/sejong/model/eojeol/last.ckpt

# NER
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --hparams /root/NAR/examples/NER/conll2003/model/v1/tb_logs/lightning_logs/version_0/hparams.yaml \
    --model_path /root/NAR/examples/NER/conll2003/model/v1/last.ckpt