      cd ~/cunning_manipulator/;
      python3 dpo_tuning/validation.py --path_to_source ./datasets/DINO_gold_dataset_with_prompt_boxes_new_ranked_by_new_strategy.csv --path_to_imgs ./gold/images/RGB_raw/ --detector_model_name onepeace --path_to_checkpoint './dpo_tuning/results/DINO/checkpoint-800' --language_model_type tuned --run_name new_data_DINO_train_800

