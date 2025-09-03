TASK_NAME=news_headline_generation

deepspeed --num_gpus=2 ./my_train.py --deepspeed ./deepspeed/ds_z3_config.json\
    --task_name "${TASK_NAME}" \
    --base_model "/NAS/yjt/models/Qwen3-1.7B" \
    --neuron_path "./neuron_deactivation/Qwen3-1.7B_detect_news_headline_generation_shared_neurons_0.01.json" \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --learning_rate 5e-5 \
    --cutoff_len 1024 \
    --output_dir "./Qwen3-1.7B_shared_lr_5e-5_${TASK_NAME}" \
    --group_by_length False
