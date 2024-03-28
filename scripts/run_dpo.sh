cd dpo_tuning;
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/DINO_gold_dataset_with_prompt_boxes_new_ranked_by_new_strategy.csv --output_dir "./results/DINO/gemma_7b_v2it" --model_name_or_path "google/gemma-7b-it"

# accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/DINO_gold_dataset_with_prompt_boxes_new_ranked_by_new_strategy.csv --output_dir "./results/DINO/gemma_2b_v2it" --model_name_or_path "google/gemma-2b-it"

# accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/DINO_gold_dataset_with_prompt_boxes_new_ranked_by_new_strategy.csv --output_dir "./results/DINO/gemma_2b_v2" --model_name_or_path "google/gemma-2b"

# accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/DINO_gold_dataset_with_prompt_boxes_new_ranked_by_new_strategy.csv --output_dir "./results/DINO/gpt2_xl" --model_name_or_path "openai-community/gpt2-xl"

# accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/DINO_gold_dataset_with_prompt_boxes_new_ranked_by_new_strategy.csv --output_dir "./results/DINO/gpt2_medium" --model_name_or_path "openai-community/gpt2-medium"

accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/dino/small/harmony_dino.csv --output_dir "./results/DINO/llama_ablation_harmony" --model_name_or_path "NousResearch/Llama-2-7b-chat-hf"

accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/dino/small/iou_dino.csv --output_dir "./results/DINO/llama_ablation_iou" --model_name_or_path "NousResearch/Llama-2-7b-chat-hf"

accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/dino/small/score_dino.csv --output_dir "./results/DINO/llama_ablation_score" --model_name_or_path "NousResearch/Llama-2-7b-chat-hf"

accelerate launch --num_processes $1 --gpu_ids $2 run.py --path_to_source ../datasets/dino/small/thresh_05_dino.csv --output_dir "./results/DINO/llama_ablation_thresh_05" --model_name_or_path "NousResearch/Llama-2-7b-chat-hf"