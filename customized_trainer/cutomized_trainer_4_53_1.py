# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import glob
import os
from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import peft
import safetensors
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import Dataset

from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer import Trainer, _is_peft_model, TRAINER_STATE_NAME
from transformers.utils import is_datasets_available, logging, find_labels, can_return_loss, is_sagemaker_mp_enabled, \
    ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, is_peft_available, \
    WEIGHTS_INDEX_NAME
from transformers.utils.deprecation import deprecate_kwarg

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
if is_datasets_available():
    import datasets

# if TYPE_CHECKING:
from torch.utils.data import IterableDataset

from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.training_args import TrainingArguments


logger = logging.get_logger(__name__)

CUSTOM_PEFT_MODEL_NAMES = [
    peft.PeftModel,
]

class Seq2SeqTrainer(Trainer):
    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Union[Dataset, "IterableDataset", "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union["PreTrainedTokenizerBase", "BaseImageProcessor", "FeatureExtractionMixin", "ProcessorMixin"]
        ] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], dict]] = None,
        callbacks: Optional[list["TrainerCallback"]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
        # Modified for peft related changes that were missing in the original version
        if _is_peft_model(self.model):
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = self.model.get_base_model()
                default_label_names = find_labels(model_to_inspect.__class__)
                self.can_return_loss = can_return_loss(model_to_inspect.__class__)
        else:
            default_label_names = find_labels(self.model.__class__)
            self.can_return_loss = can_return_loss(self.model.__class__)

        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        # End of modification

    @staticmethod
    def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
        """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig]`):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """

        # GenerationConfig provided, nothing to do
        if isinstance(gen_config_arg, GenerationConfig):
            gen_config = deepcopy(gen_config_arg)
        else:
            # str or Path
            pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
            config_file_name = None

            # Figuring if it is path pointing to a file, pointing to a directory or else a model id or URL
            # This step is required in order to determine config_file_name
            if pretrained_model_name.is_file():
                config_file_name = pretrained_model_name.name
                pretrained_model_name = pretrained_model_name.parent
            # dir path
            elif pretrained_model_name.is_dir():
                pass
            # model id or URL
            else:
                pretrained_model_name = gen_config_arg

            gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)

        # Strict validation to fail early. `GenerationConfig.save_pretrained()`, run at the end of training, throws
        # an exception if there are warnings at validation time.
        try:
            gen_config.validate(strict=True)
        except ValueError as exc:
            raise ValueError(str(exc) + "\n\nFix these issues to train your model.")

        return gen_config

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            with torch.amp.autocast(device_type="cuda"):
                generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
                _generated_tokens = generated_tokens[:, inputs['input_ids'].shape[1]:]
                generated_tokens = _generated_tokens

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).detach().mean()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach().mean()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.processing_class is not None and hasattr(self.processing_class, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.processing_class.pad_token_id
                if self.processing_class.pad_token_id is not None
                else self.processing_class.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _load_best_model(self, order=-1):
        if self.state.best_model_checkpoint is None and self.args.resume_from_checkpoint is not None:
            checkpoint_paths = glob.glob(f'{self.args.resume_from_checkpoint}/checkpoint-*')
            checkpoint_paths.sort(key=lambda x: int(x.split('-')[-1]))
            checkpoint_path = checkpoint_paths[order]
            self.state = TrainerState.load_from_json(os.path.join(checkpoint_path, TRAINER_STATE_NAME))
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        print(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if (
                os.path.exists(best_model_path)
                or os.path.exists(best_safe_model_path)
                or os.path.exists(best_adapter_model_path)
                or os.path.exists(best_safe_adapter_model_path)
        ):
            # if eval is called w/o train, handle model prep here
            if self.is_deepspeed_enabled and self.deepspeed is None:
                _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
            # when we do inference directly without init ds, we load normal ckpt the model first
            if self.is_deepspeed_enabled and 'deepspeed.runtime.engine.DeepSpeedEngine' in str(
                    self.model_wrapped.__class__):
                deepspeed_load_checkpoint(self.model_wrapped, self.state.best_model_checkpoint,
                                          load_module_strict=False)
            else:
                has_been_loaded = True
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                        # If the 'user_content.pt' file exists, load with the new smp api.
                        # Checkpoint must have been saved with the new smp api.
                        smp.resume_from_checkpoint(
                            path=self.state.best_model_checkpoint,
                            tag=WEIGHTS_NAME,
                            partial=False,
                            load_optimizer=False,
                        )
                    else:
                        # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                        # Checkpoint must have been saved with the old smp api.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        state_dict["_smp_is_partial"] = False
                        load_result = model.load_state_dict(state_dict, strict=True)
                elif self.is_fsdp_enabled:
                    self.accelerator.state.fsdp_plugin.load_model(
                        self.accelerator, model, self.state.best_model_checkpoint
                    )
                else:
                    check_is_custom_peft = [isinstance(model, _c) for _c in CUSTOM_PEFT_MODEL_NAMES]
                    check_is_custom_peft = reduce(lambda x, y: x or y, check_is_custom_peft, False)
                    if is_peft_available() and check_is_custom_peft:
                        # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                        if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                            if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                                model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                                # Load_adapter has no return value present, modify it when appropriate.
                                from torch.nn.modules.module import _IncompatibleKeys

                                load_result = _IncompatibleKeys([], [])
                            else:
                                logger.warning(
                                    "The intermediate checkpoints of PEFT may not be saved correctly, "
                                    f"using `TrainerCallback` to save {ADAPTER_WEIGHTS_NAME} in corresponding folders, "
                                    "here are some examples https://github.com/huggingface/peft/issues/96"
                                )
                                has_been_loaded = False
                        else:
                            logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                            has_been_loaded = False
                    else:
                        # We load the model state dict on the CPU to avoid an OOM error.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        # If the model is on the GPU, it still works!
                        # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                        # which takes *args instead of **kwargs
                        load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )