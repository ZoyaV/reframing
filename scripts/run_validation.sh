      cd ~/cunning_manipulator/;
      python3 dpo_experiment/validation.py --path_to_source ./dpo_experiment/new_DINO_gold_dataset.csv --path_to_imgs /datasets/gold/images/RGB_raw/ --detector_model_name DINO --path_to_checkpoint ./dpo_experiment/results/checkpoint-4900/ --language_model_type tuned --run_name new_data_DINO_train_4900

