CUDA_VISIBLE_DEVICES=0 python ./my_evaluate.py \
    --model_dir /NAS/yjt/Abstract_Thought/Qwen3-1.7B_lr_5e-6_base_news_headline_generation/checkpoint-781\
    --task_name "news_headline_generation" \
    --test_path "/NAS/yjt/Abstract_Thought/dataset/LaMP/news_headline_generation/dataset_test" \
    --batch_size 60