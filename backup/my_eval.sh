CUDA_VISIBLE_DEVICES=4 python ./my_evaluate.py \
    --model_dir /NAS/yjt/Abstract_Thought/Qwen3-1.7B_lr_5e-6_base_citation_identification/checkpoint-103 \
    --batch_size 36 \
    --task_name "citation_identification" \
    --max_tokens 48
    # --max_new_tokens 2048