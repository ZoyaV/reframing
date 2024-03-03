cd ~/cunning_manipulator/
python3 data_processing/DPO_dataset_generation.py --path_to_source ./dpo_experiment/new_DINO_gold_dataset.csv --path_to_imgs /datasets/gold/images/RGB_raw/ --model_name DINO --path_to_output ./dpo_dataset.csv
