TASK_NAME=citation_identification
export CUDA_VISIBLE_DEVICES=7
# deepspeed --num_gpus=4 ./baseline_train.py --deepspeed ./deepspeed/ds_z1_config.json\
python ./baseline_train.py \
    --task_name "${TASK_NAME}" \
    --base_model "/NAS/yjt/models/Qwen3-1.7B" \
    --batch_size 64 \
    --micro_batch_size 4 \
    --num_epochs 5 \
    --learning_rate 5e-6 \
    --cutoff_len 1024 \
    --output_dir "./Qwen3-1.7B_lr_5e-6_base_${TASK_NAME}" \
    --group_by_length False
