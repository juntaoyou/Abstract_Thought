CUDA_VISIBLE_DEVICES=7 python ./pre.py --generate 
CUDA_VISIBLE_DEVICES=7 python ./pre.py --generate --add_principle --principle helpful
CUDA_VISIBLE_DEVICES=7 python ./pre.py --generate --add_principle --principle correct
CUDA_VISIBLE_DEVICES=7 python ./pre.py --generate --add_principle --principle coherent

# CUDA_VISIBLE_DEVICES=7 python ./pre.py --generate --add_principle --principle coherence