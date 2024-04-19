from trl import DPOTrainer
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, NamedTuple
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from dpo_tuning.utils.detector import get_images
from dpo_tuning.utils.data import pad_to_length

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from dpo_tuning.utils.metrics import box_iou
from torchvision.ops import box_convert
import re


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
class ReframingTrainer(DPOTrainer):
    def __init__(self, 
        detection_model=None, 
        path_to_imgs = None,       
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,):

        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free,
            force_use_ref_model=force_use_ref_model
        )
        
        self.detection_model = detection_model
        if self.detection_model is not None:
            print("Detection model has been set.")
        self.path_to_imgs = path_to_imgs
    def get_eval_images(obj, path):
        name, img_sources, images = get_images(obj, path)
        return name, img_sources, images
    
    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        # policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # dataloader = self.get_eval_dataloader(self.train_dataset)
            ious = []
            scores = []
            ref_ious = []
            ref_scores = []
            values = []
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)
            
            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)
            with open(f'results/input_output_examples.txt', 'a') as f:
                f.write('ДО ПРЕДСКАЗАНИЯ\n')
            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)
            with open(f'results/input_output_examples.txt', 'a') as f:
                f.write('ПОСЛЕ ПРЕДСКАЗАНИЯ\n')
            for item_id, dataset_bbox, prompt, pol, ref in zip(
                                random_batch['item_id'],random_batch['true_bbox'], random_batch["prompt"], policy_output_decoded, ref_output_decoded):
                name, img_sources, images = get_images(item_id, self.path_to_imgs)
                if 'Sure!' in pol or 'Sure,' in pol:
                    pol = pol[len(prompt) :].split(":")[1:]
                else:
                    pol = pol[len(prompt) :]
                if 'Sure!' in ref or 'Sure,' in ref:
                    ref = ref[len(prompt) :].split(":")[1:]
                else:
                    ref = ref[len(prompt) :]
                image_metadata = {'image_np': images, 'image_pil': img_sources, 'correct': pol, 'image_path': self.path_to_imgs+name}
                predicted_bbox, pred_score = self.detection_model.predict(image_metadata)
                image_metadata = {'image_np': images, 'image_pil': img_sources, 'correct': ref, 'image_path': self.path_to_imgs+name}
                ref_predicted_bbox, ref_pred_score = self.detection_model.predict(image_metadata)
                dataset_bbox = torch.Tensor([[float(x) for x in re.split(',', dataset_bbox[1:-1])]])
                real_bbox = box_convert(boxes=dataset_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()[0]
                iou_score = float(box_iou(real_bbox, ref_predicted_bbox))
                ref_iou_score = float(box_iou(real_bbox, predicted_bbox))
                ious.append(iou_score)
                ref_ious.append(iou_score - ref_iou_score)
                scores.append(pred_score)
                ref_scores.append(pred_score - ref_pred_score)
                with open(f'results/input_output_examples.txt', 'a') as f:
                    f.write('ПРОМПТ: '+ str(prompt)+ ' ЗАКОНЧЕН')
                    f.write('\n')
                    f.write('ОТВЕТ ТРЕНИРУЕМОЙ МОДЕЛИ: '+ str(pol)+ ' ЗАКОНЧЕН')
                    f.write('\n')
                    f.write('ОТВЕТ РЕФЕРЕНСНОЙ МОДЕЛИ: '+ str(ref)+ 'ЗАКОНЧЕН')
                    f.write('\n')
                    f.write('--------------------------------------------------------------------------------------------------------------------------------')
                    f.write('\n')
            threshhold=0.7
            for i in range(len(ious)):
                if float(ious[i])>float(threshhold):
                    values.append(scores[i])
                else:
                    values.append(0.0)
            self.log(
                {
                    "metrics/game_log": wandb.Table(
                        columns=["item_id", "Prompt", "Policy", "Ref Model"],
                        rows=[
                            [item_id, prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for item_id, prompt, pol, ref in zip(
                                random_batch['item_id'], random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    ),
                    "metrics/IOU": np.mean(ious),
                    "metrics/score": np.mean(scores),
                    "metrics/IOU-ref_IOU": np.mean(ref_ious),
                    "metrics/score-ref_score": np.mean(ref_scores),
                    "metrics/ranking_score": np.mean(values)
                }
            )
            self.state.log_history.pop()

        # # Base evaluation
        # initial_output = super().evaluation_loop(
        #     dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        # )

        return EvalLoopOutput(predictions=None, label_ids=None, metrics={'proxy':1}, num_samples=1)