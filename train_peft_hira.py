import os
import pickle

import numpy as np
from dotenv import load_dotenv

from customized_trainer.cutomized_trainer_4_53_1 import Seq2SeqTrainer

load_dotenv('.env')
from tqdm import tqdm
from transformers import AutoTokenizer, EarlyStoppingCallback

from dataset.dataset_hg import HGDataset
from dataset.format_inputs import TASK_TYPE, format_causal_input, gen_max_new_token_map, token_length_map, dataset_map, \
    task_map


from datetime import datetime
import jsonlines
import torch
import transformers
from pytictoc import TicToc
from models.get_models import print_trainable_parameters, get_tokenizer, get_prefix_tuning_models, get_hira_models, \
    get_fft_models, get_peft_hira_models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--peft_type', type=str,
                    choices=['prefix', 'hira', 'fft', 'peft_hira'])
parser.add_argument('--enable_grad_ckpt', action='store_true')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--grad_acc', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--epoch', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default="facebook/opt-125m")
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--dataset', type=str, default='boolq',
                    choices=['e2e_nlg', 'dailydialog', 'samsum', 'e2e_cleaned', 'boolq', 'mmlu', 'common_170k', 'gsm8k',
                             'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'convai2', 'meta_math',
                             'common_all'])
parser.add_argument('--dataset_analysis', action='store_true')

parser.add_argument('--dataset_ratio', type=float, default=1.0)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--ds_config', type=str, default=None)
parser.add_argument('--output_folder', type=str, default='results_peft_hira')
parser.add_argument('--load_bit', type=int, default=16)
parser.add_argument('--r_ab', type=int, default=16)
parser.add_argument('--target_modules', type=str, default='q_proj, v_proj')
parser.add_argument('--eval_strategy', type=str, default='epoch', choices=['no', 'steps', 'epoch'])
parser.add_argument('--eval_steps', type=float, default=1.0)
parser.add_argument('--max_new_tokens', type=int, default=None)
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--virtual_tokens', type=int, default=8)
parser.add_argument('--compute_rank', action='store_true')
parser.add_argument('--compute_norm', action='store_true')
parser.add_argument('--load_order', type=int, default=-1)
parser.add_argument('--init_ab', type=str, default='kaiming,zero')
parser.add_argument('--train_ab', type=str, default='yy', help='y means yes, n means no')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--do_sample', default='false', type=str)
parser.add_argument('--rand_R', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--decoding', type=str, default='default', choices=['default', 'greedy'])
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--early_stop_patience', type=int, default=0)
COMPUTE_DS_LENGTH = False
args = parser.parse_args()
if args.compute_rank or args.compute_norm:
    assert args.ckpt is not None
if args.ckpt is not None:
    output_name = 'output_{}_{}'.format(args.load_order, args.dataset)
    if args.dataset == 'mmlu':
        output_name += '_prob'
    if args.max_new_tokens is not None:
        output_name += '_maxT={}'.format(args.max_new_tokens)
    if args.beam_size is not None:
        output_name += '_beam={}'.format(args.beam_size)
    output_path = '{}/{}_eval.jsonl'.format(args.ckpt, output_name)
    if os.path.exists(output_path.format(args.ckpt, args.dataset)) and not (args.compute_rank or args.compute_norm):
        print(f"File exists, skipped.")
        exit(0)
    print(f"Current ckpt args only supports inference!")
    output_jsonl = os.path.join(args.ckpt, 'output.jsonl')
    if os.path.exists(output_jsonl):
        with jsonlines.open(output_jsonl) as reader:
            dict_args = reader.read()
        print(f"dict_args: {dict_args}")
        print("Overriding args")
        args.peft_type = dict_args['peft_type']
        args.model_name = dict_args['model_name']
        args.r_ab = dict_args['r_ab']
        if 'rand_R' in dict_args.keys():
            args.rand_R = dict_args['rand_R']
        args.target_modules = dict_args['target_modules']
    else:
        print(f'cannot find {output_jsonl}')
        exit(0)

dataset_name = args.dataset

MAX_NEW_TOKEN_LENGTH = gen_max_new_token_map[dataset_name]
MAX_TOKEN_LENGTH = token_length_map[dataset_name]

if args.max_new_tokens is not None:
    MAX_NEW_TOKEN_LENGTH = args.max_new_tokens

# keyword_map = {
#     'e2e_nlg': 'TARGET: '
# }

# convert args to dict
args_dict = vars(args)
model_name = args.model_name
peft_type = args.peft_type
train_ab = args.train_ab
# create a directory by time
exp_name = f"{args.output_folder}/{model_name.split('/')[-1]}-{dataset_name}-{peft_type}-lr={format(args.lr, '.2e')}-"
if args.load_bit != 16:
    exp_name = exp_name + f'{args.load_bit}bit-'
if peft_type == 'hira' or peft_type == 'peft_hira':
    init_ab_ = ''.join([i[0] for i in args.init_ab.split(',')])
    exp_name = exp_name + f'r_ab={args.r_ab}-'
    exp_name = exp_name + f'init={init_ab_}-'
    exp_name = exp_name + f'train={train_ab}-'

if args.seed is not None:
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    exp_name = exp_name + f'seed={args.seed}-'
    seed_everything(args.seed)
output_dir_by_time = exp_name + args.exp_name + '-' + datetime.now().strftime(
    "%Y-%m-%d-%H-%M-%S")


if args.ckpt is None:
    os.makedirs(output_dir_by_time, exist_ok=True)

train_dataset = HGDataset(dataset_map[dataset_name], 'train', task_map[dataset_name], training_ratio=args.dataset_ratio)
valid_dataset = HGDataset(dataset_map[dataset_name], 'validation', task_map[dataset_name],
                          training_ratio=args.dataset_ratio)
test_dataset = HGDataset(dataset_map[dataset_name], 'test', task_map[dataset_name], training_ratio=args.dataset_ratio)

if args.dataset_analysis:
    tokenizer_ds = get_tokenizer(model_name=model_name)
    train_dataset.length_analysis(tokenizer_ds)
    valid_dataset.length_analysis(tokenizer_ds)
    test_dataset.length_analysis(tokenizer_ds)
    exit(0)

if peft_type == 'hira':
    init_a, init_b = args.init_ab.split(',')
    model, tokenizer, model_config = get_hira_models(load_bit=args.load_bit,
                                                     model_name=model_name, enable_checkpoint=args.enable_grad_ckpt,
                                                     r_ab=args.r_ab,target_modules=args.target_modules,
                                                     train_ab=args.train_ab,
                                                     rand_R=args.rand_R)
elif peft_type == 'peft_hira':
    init_a, init_b = args.init_ab.split(',')
    model, tokenizer, model_config = get_peft_hira_models(load_bit=args.load_bit,
                                                     model_name=model_name, enable_checkpoint=args.enable_grad_ckpt,target_modules=args.target_modules,)
elif peft_type == 'fft':
    model, tokenizer, model_config = get_fft_models(load_bit=args.load_bit,
                                                    model_name=model_name, enable_checkpoint=args.enable_grad_ckpt)

elif peft_type == 'prefix':
    model, tokenizer, model_config = get_prefix_tuning_models(load_bit=args.load_bit, model_name=model_name,
                                                              enable_checkpoint=args.enable_grad_ckpt,
                                                              virtual_tokens=args.virtual_tokens)
else:
    raise NotImplementedError('Not supported model!')
trainable_params = print_trainable_parameters(model)


def get_parameter_dict(model):
    return dict(model.named_parameters())



tokenizer_left = get_tokenizer(model_name=model_name)
tokenizer_left.padding_side = 'left'

tokenizer_right = get_tokenizer(model_name=model_name)
tokenizer_right.padding_side = 'right'

if tokenizer_left.pad_token_id is None and 'llama-3' in model_name.lower():
    tokenizer_left.pad_token = tokenizer_left.bos_token
    tokenizer_right.pad_token = tokenizer_right.bos_token
    tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = tokenizer.bos_token_id
elif tokenizer_left.pad_token_id is None:
    tokenizer_left.pad_token = tokenizer_left.unk_token
    tokenizer_right.pad_token = tokenizer_right.unk_token
    tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.unk_token_id

# Compute length for this dataset
if COMPUTE_DS_LENGTH:
    all_data = [train_dataset.__getitem__(idx) for idx in range(train_dataset.__len__())]
    all_data = [a['input'] + a['target'] for a in all_data]
    all_data_ids = tokenizer_right(all_data)
    all_data_len = [len(a) for a in all_data_ids['input_ids']]
    all_data_len = torch.tensor(all_data_len, dtype=torch.float)
    print(f"""
        AVG: {all_data_len.mean()}
        MAX: {all_data_len.max()}
        MIN: {all_data_len.min()}
    """)

test_steps = 0


def data_collator_e2e(features, return_tensors="pt"):
    batchfied_features = {}
    keys = features[0].keys()
    for key in keys:
        batchfied_features[key] = [f[key] for f in features]
    split = batchfied_features['split'][0]
    for_inference = (split == 'test')
    template_type = 0
    if dataset_name in ['boolq', 'gsm8k', 'common_170k', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa',
                        'common_all']:
        template_type = 4
    if dataset_name in ['mmlu']:
        template_type = 7
    lm_input, lm_target = format_causal_input(batchfied_features, tokenizer_left, tokenizer_right,
                                              template_type=template_type, max_token_length=MAX_TOKEN_LENGTH,
                                              for_test=for_inference, shift_target=False,
                                              target_length=MAX_NEW_TOKEN_LENGTH)
    # Replace target pad to -100
    lm_target_ce = lm_target.clone()
    lm_target_ce[lm_target_ce == tokenizer_left.pad_token_id] = -100
    if peft_type in ['prefix']:
        lm_input['attention_mask'] = None
    batch = {**lm_input, 'labels': lm_target_ce}
    if for_inference:
        batch = lm_input
    return batch


generation_config = transformers.generation.GenerationConfig(
    max_length=MAX_TOKEN_LENGTH,
    num_beams=1,
)
callbacks = []
if args.early_stop_patience > 0:
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))
eval_bz = max(1, int(args.batch / 4))
if dataset_name == 'common_170k':
    eval_bz = args.batch
trainer_class = Seq2SeqTrainer
trainer_args = transformers.Seq2SeqTrainingArguments(
    deepspeed=args.ds_config,
    local_rank=args.local_rank,
    dataloader_num_workers=args.num_workers,
    resume_from_checkpoint=args.ckpt,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=eval_bz,
    gradient_accumulation_steps=args.grad_acc,
    gradient_checkpointing=args.enable_grad_ckpt,
    warmup_steps=args.warmup,
    weight_decay=args.weight_decay,
    num_train_epochs=args.epoch,
    learning_rate=args.lr,
    bf16=True if torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
    fp16=True if not torch.cuda.is_bf16_supported() and args.load_bit == 16 else False,
    metric_for_best_model='eval_loss',
    logging_steps=1,
    remove_unused_columns=False,
    save_on_each_node=False,
    save_safetensors=peft_type == 'fft',
    output_dir=output_dir_by_time,
    do_eval=True,
    eval_strategy=args.eval_strategy,
    save_strategy=args.eval_strategy,
    save_steps=args.eval_steps,
    logging_strategy='steps',
    save_total_limit=args.save_total_limit,
    report_to=['tensorboard'],
    eval_steps=args.eval_steps,  # this means evaluate every 1 epoch
    eval_accumulation_steps=1,
    generation_config=transformers.generation.GenerationConfig(
        max_length=MAX_TOKEN_LENGTH,
        num_beams=1,
    ),
    load_best_model_at_end=True,
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=callbacks,
    args=trainer_args,
    data_collator=data_collator_e2e
)
if args.ckpt is not None:
    trainer._load_best_model(order=args.load_order)
    train_seconds = -1

    # if not (args.rand_R or args.compute_rank or peft_type == 'fft'):
    #     model.merge_and_unload()
else:
    train_tic = TicToc()
    train_tic.tic()
    trainer.train()
    train_seconds = train_tic.tocvalue()

kwgenargs = {}
if args.do_sample is not None:
    if args.do_sample.lower() in ['yes', 'true']:
        do_sample = True
    else:
        do_sample = False
    kwgenargs['do_sample'] = do_sample
if args.decoding == 'greedy':
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False)
elif dataset_name == 'common_170k':
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False)
elif dataset_name == 'common_all':
    beam = 4
    eval_result = {}
    for _dataset_name in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa']:
        print(f'decoding {dataset_name}')
        test_dataset = HGDataset(dataset_map[_dataset_name], 'test', task_map[_dataset_name],
                                 training_ratio=args.dataset_ratio)
        _output_path = output_path.replace(dataset_name, _dataset_name)
        if os.path.exists(_output_path):
            print(f'{_output_path} exists, skipping')
            continue
        if args.beam_size is not None:
            beam = args.beam_size
        _eval_result = trainer.predict(test_dataset,
                                       max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                       pad_token_id=tokenizer.pad_token_id,
                                       temperature=0.1,
                                       top_p=0.75,
                                       top_k=40,
                                       num_beams=beam,
                                       **kwgenargs,
                                       # do_sample=False, num_beams=10,
                                       # length_penalty=0.9, no_repeat_ngram_size=4
                                       )
        eval_result[_output_path] = _eval_result
#
elif dataset_name in ['mmlu']:
    eval_results = []
    pbar = tqdm(test_dataset)
    id_a, id_b, id_c, id_d = tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D'])
    options = ['A', 'B', 'C', 'D']
    context = []
    text_result = []
    ground_truth = []
    for row in pbar:
        model.eval()
        with torch.no_grad():
            keys = row.keys()
            batchfied_features = {}
            for key in keys:
                batchfied_features[key] = [row[key]]
            lm_input, lm_target = format_causal_input(batchfied_features, tokenizer_left, tokenizer_right,
                                                      template_type=7, max_token_length=MAX_TOKEN_LENGTH,
                                                      for_test=True, shift_target=False,
                                                      target_length=MAX_NEW_TOKEN_LENGTH)
            lm_input = lm_input.to('cuda')
            with torch.autocast('cuda'):
                prob = model(**lm_input).logits
            id_probs = prob[0][-1][[id_a, id_b, id_c, id_d]]
            prob_pred = options[id_probs.argmax().item()]
            answer = row['target']
            eval_results.append(prob_pred == answer)
            acc = np.asarray(eval_results).mean()
            pbar.set_postfix_str(f'Current ACC: {acc * 100}')
            context.append(row['input'])
            text_result.append(prob_pred)
            ground_truth.append(answer)
    print(f'ACC: {acc * 100}')
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write(args_dict)
        writer.write(model_config.to_dict())
        writer.write({"acc": acc * 100})
        writer.write(trainable_params)
        for c, p, g in zip(context, text_result, ground_truth):
            writer.write({
                'context': c,
                'pred': p,
                'gt': g,
            })
    exit(0)
elif dataset_name in ['boolq', 'piqa', 'siqa', 'hellas', 'winog', 'arce', 'arcc', 'obqa', 'gsm8k']:
    beam = 4
    if args.beam_size is not None:
        beam = args.beam_size
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  temperature=0.1,
                                  top_p=0.75,
                                  top_k=40,
                                  num_beams=beam,
                                  **kwgenargs,
                                  # do_sample=False, num_beams=10,
                                  # length_penalty=0.9, no_repeat_ngram_size=4
                                  )
else:
    # actually, this is for e2e/convai2/mmlu
    eval_result = trainer.predict(test_dataset,
                                  max_new_tokens=MAX_NEW_TOKEN_LENGTH,
                                  pad_token_id=tokenizer.pad_token_id,
                                  do_sample=False, num_beams=4,
                                  length_penalty=0.9, no_repeat_ngram_size=4)
if not isinstance(eval_result, dict):
    if args.ckpt is None:
        output_path = '{}/output.jsonl'.format(output_dir_by_time)
    eval_result = {output_path: eval_result}
for _output_path, _eval_result in eval_result.items():
    # convert logits into text
    if args.local_rank in [-1, 0]:
        logits = _eval_result.predictions
        logits[logits == -100] = tokenizer.pad_token_id
        raw_text_result = tokenizer.batch_decode(logits)
        text_result = []
        for tt in raw_text_result:
            tt = tt.replace(tokenizer.pad_token, '')
            keywords = [tokenizer.eos_token, 'Q:', 'R:']
            for keyword in keywords:
                if keyword in tt:
                    tt = tt[:tt.index(keyword)]
            text_result.append(tt)

        context = [test_dataset.__getitem__(i)['input'] for i in range(test_dataset.__len__())]
        ground_truth = [test_dataset.__getitem__(i)['target'] for i in range(test_dataset.__len__())]
        if args.ckpt is not None:
            if os.path.exists(output_dir_by_time):
                os.removedirs(output_dir_by_time)
        else:
            _output_path = '{}/output.jsonl'.format(output_dir_by_time)
        mem_used = torch.cuda.mem_get_info()[1] / 1024 / 1024 - torch.cuda.mem_get_info()[0] / 1024 / 1024

        with jsonlines.open(_output_path, mode='w') as writer:
            writer.write(args_dict)
            if peft_type != 'fft':
                writer.write(str(model_config.to_dict()))
            else:
                writer.write('\n')
            writer.write({"mem_used": mem_used, "train_seconds": train_seconds})
            writer.write(trainable_params)
            for c, p, g in zip(context, text_result, ground_truth):
                writer.write({
                    'context': c,
                    'pred': p,
                    'gt': g
                })
