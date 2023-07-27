import yaml
import wandb
from tqdm import tqdm
from trl import PPOConfig
import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser(description='Set up configs from YAML and command line arguments.')
parser.add_argument('--config_path', default='config.yaml', type=str, help='Path to the YAML config file.')
parser.add_argument('--entity', default=None, type=str, help='Entity for wandb.')
parser.add_argument('--project', default=None, type=str, help='Project for wandb.')
parser.add_argument('--txt_in_len', default=None, type=int, help='TXT_IN_LEN parameter.')
parser.add_argument('--txt_out_len', default=None, type=int, help='TXT_OUT_LEN parameter.')
parser.add_argument('--seed', default=None, type=int, help='SEED parameter.')
parser.add_argument('--model_name', default=None, type=str, help='MODEL_NAME parameter.')
parser.add_argument('--pretrained_model', default=None, type=str, help='PRETRAINED_MODEL parameter.')
parser.add_argument('--inp', default=None, type=str, help='INPUT parameter.')
parser.add_argument('--out', default=None, type=str, help='OUTPUT parameter.')
parser.add_argument('--reward_model', default='hf', type=str, help='reward_model parameter.') #detector or hf

# Parse the arguments
args = parser.parse_args()

# Load the YAML file
with open(args.config_path, 'r') as stream:
    try:
        configs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# If any command line argument is set, it overrides the corresponding YAML config
if args.entity:
    configs['wandb']['entity'] = args.entity
if args.project:
    configs['wandb']['project'] = args.project
if args.txt_in_len:
    configs['model']['txt_in_len'] = args.txt_in_len
if args.txt_out_len:
    configs['model']['txt_out_len'] = args.txt_out_len
if args.seed:
    configs['model']['seed'] = args.seed
if args.model_name:
    configs['model']['model_name'] = args.model_name
if args.pretrained_model:
    configs['model']['pretrained_model'] = args.pretrained_model
if args.inp:
    configs['data']['inp'] = args.inp
if args.out:
    configs['data']['out'] = args.out

# Init wandb
wandb.init(entity=configs['wandb']['entity'], project=configs['wandb']['project'])

# Enable progress bar for pandas
tqdm.pandas()

# Set model configs
INPUT = configs['data']['inp']
OUTPUT = configs['data']['out']
TXT_IN_LEN = configs['model']['txt_in_len']
TXT_OUT_LEN = configs['model']['txt_out_len']
SEED = configs['model']['seed']
MODEL_NAME = configs['model']['model_name']
PRETRAINED_MODEL = configs['model']['pretrained_model']
PROMPTS = [ lambda x: f"In other words, '{x}' is = ",
            lambda x: f"A synonym for the word '{x}' is = ",
            lambda x: f"In general, '{x}' is a subclass of = "]
# Set PPO trainer configs
config = PPOConfig(
    model_name=MODEL_NAME,
    steps=configs['ppo_trainer']['config']['steps'],
    learning_rate=configs['ppo_trainer']['config']['learning_rate'],
    remove_unused_columns=configs['ppo_trainer']['config']['remove_unused_columns'],
    log_with=configs['ppo_trainer']['config']['log_with'],
    batch_size=configs['ppo_trainer']['config']['batch_size']
)

REWARD_MODEL = args.reward_model
# Set generation configs
generation_kwargs = configs['text_generation']['parameters']
