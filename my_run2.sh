learning_rate=5e-5

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 4 ./my_train.py \
    --output_dir "Qwen3-1.7B_shared_lr_${learning_rate}" \
    --task_name "product_review_temporal" \
    --base_model "/NAS/yjt/models/Qwen3-1.7B" \
    --neuron_path "./neuron_deactivation/Qwen3-1.7B_detect_shared_neurons_0.01.json" \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --learning_rate ${learning_rate} \
    --cutoff_len 1024 \
    --group_by_length False
