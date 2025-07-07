import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tokenizer(model_name="facebook/opt-1.3b"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def get_models(model_name="facebook/opt-1.3b", enable_checkpoint=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, low_cpu_mem_usage=True
    )

    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer


def get_fft_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        #         device_map='auto',
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    print_trainable_parameters(model)
    return model, tokenizer, None


def get_hira_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16,
                    r_ab=16, target_modules=None, init_a='kaiming',
                    init_b='zero', train_ab='yy',
                    rand_R=False):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        #         device_map='auto',
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    from hira import LoraConfig, get_peft_model
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    _train_ab = [True, True]
    for idx, char in enumerate(train_ab):
        _train_ab[idx] = (char == 'y')
    config = LoraConfig(
        rand_R=rand_R,
        scale_ab=1.0,
        r_ab=r_ab,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_a=init_a,
        init_b=init_b,
        train_a=_train_ab[0],
        train_b=_train_ab[1],
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config



def get_peft_hira_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=16,
                    r=16, target_modules=None):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        #         device_map='auto',
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    from peft.tuners.hira import HiRAConfig
    from peft import get_peft_model
    if target_modules is not None:
        target_modules = target_modules.split(',')
    else:
        target_modules = ["q_proj", "v_proj"]
    config = HiRAConfig(
        r=r,
        target_modules=target_modules,
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


def get_prefix_tuning_models(model_name="facebook/opt-1.3b", enable_checkpoint=False, load_bit=8, virtual_tokens=8):
    load_params = {}
    if load_bit == 16:
        load_params = {'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=(load_bit == 8),
        **load_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    if enable_checkpoint:
        model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    from peft import PrefixTuningConfig, get_peft_model

    config = PrefixTuningConfig(
        num_virtual_tokens=virtual_tokens,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model, tokenizer, config


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return {"trainable": trainable_params, "all": all_param, "trainable%": 100 * trainable_params / all_param}
