from typing import Dict, Optional
from dataclasses import dataclass, field

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    seed: Optional[int] = field(default=105, metadata={"help": "training seed"})
    model_name_or_path: Optional[str] = field(
        default="NousResearch/Llama-2-7b-chat-hf",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    #per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=5000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    #eval_steps: Optional[int] = field(default=1, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=100, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

@dataclass
class ProcessingArguments:
    path_to_source: Optional[str] = field(default='./dataset/modified_output_concatenated_long_with_bboxes.csv',\
                                           metadata={"help": "path to csv with descriptions, img refs and GT bboxes"})
    
    path_to_imgs: Optional[str] = field(default='../ONE-PEACE/gold/images/RGB_raw/',\
                                         metadata={"help": "path to folder with images"})
    
    path_to_output: Optional[str] = field(default="./dataset/dpo_all_pipeline_test.csv", \
                                            metadata={"help": "path to output file"})
    model_name: Optional[str] = field(default="DINO", metadata={"help": "detector model name"})


class ValidationArguments:
    path_to_source: Optional[str] = field(default='./new_DINO_gold_dataset.csv',\
                                           metadata={"help": "path to csv with descriptions, img refs and GT bboxes"})
    
    path_to_imgs: Optional[str] = field(default='../ONE-PEACE/gold/images/RGB_raw/',\
                                         metadata={"help": "path to folder with images"})
    
    path_to_output: Optional[str] = field(default="./dataset/dpo_all_pipeline_test.csv", \
                                            metadata={"help": "path to output file"})
    ValidationArguments: Optional[str] = field(default="DINO", metadata={"help": "detector model name: None/onepeace/DINO"})
    language_model_type: Optional[str] = field(default="none", metadata={"help": "none/pretrained/tuned"})
    language_model_name: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "path to langauge model or it's name"})
    v_range: Optional[str] = field(default="train", metadata={"help": "validation range: train/test"})
    path_to_checkpoint: Optional[str] = field(default="./results/checkpoint-1000", metadata={"help": "path to tuned LLM checkpoint"})
    prompt: Optional[str] = field(default="Paraphrase sentence: ", metadata={"help": "prompt used in dataset"})
    run_name = Optional[str] = field(default="llama2_tuned", metadata={"help": "run name"})