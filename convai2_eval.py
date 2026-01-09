import csv
import re
from collections import Counter
from glob import glob
from typing import Callable

import jsonlines
import numpy as np
from dotenv import load_dotenv
import evaluate
import os
import json


from tqdm import tqdm
# pred = [p1, p2 ..., pn] gt=[[g1,g2,...,gn]]
def f1_scorer(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    def f1(prediction, ground_truth, normalize_fn):
        prediction_tokens = normalize_fn(prediction).split()
        ground_truth_tokens = normalize_fn(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])
def eval_distinct(corpus):
            unigrams = []
            bigrams = []
            for n, rep in enumerate(corpus):
                rep = rep.strip()
                temp = rep.split(' ')
                unigrams += temp
                for i in range(len(temp) - 1):
                    bigrams.append(temp[i] + ' ' + temp[i + 1])
            distink_1 = len(set(unigrams)) * 1.0 / len(unigrams)
            distink_2 = len(set(bigrams)) * 1.0 / len(bigrams)
            return distink_1, distink_2


load_dotenv('.env')

bleu_scorer = evaluate.load('bleu')
# nist_mt = evaluate.load("nist_mt")
rouge_scorer = evaluate.load('rouge')
distinct_scorer = eval_distinct
meteor_scorer = evaluate.load('meteor')
bert_scorer = evaluate.load('bertscore')



eval_paths = glob('<fill in your results dir here>')

for eval_path in eval_paths:
    jsonl_files = glob(f'{eval_path}/*/*.jsonl')
    csv_header = ['folder_name', 'bleu', 'meteor', 'rougel',
                  'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'bert f1', 'BERT Recall', 'BERT Precision', 'f1', 'dist1','dist2', 'config']
    csv_data = []
    for json_file in tqdm(jsonl_files):
        print(json_file)
        json_data = []
        lines = []
        with open(json_file, 'r') as f:
            for line in f.readlines():
                if '}{' in line:
                    lines += [line[:line.index('}{') + 1], line[line.index('}{')+1:]]
                elif '{{' in line:
                    lines += [line[:line.index('{{')], line[line.index('{{')+1 :]]
                else:
                    lines.append(line)
        lines = [line for line in lines if len(line.strip()) > 0]
        for line in lines:
            # print(line)
            try:
                json_data.append(json.loads(line))
            except Exception as e:
                print(e)
        # with jsonlines.open(json_file, 'r') as file:
        #     for line in file:
        #         print(line)
        #         json_data.append(line)
        configs = json_data[:2]
        contents = [line for line in json_data if line.__class__ is dict and 'context' in line.keys()]
        arranged_contents = []
        for content in contents:
            context = content['context']
            prediction = content['pred']
            reference = content['gt']
            if len(arranged_contents) > 0 and arranged_contents[-1]['context'] == context:
                arranged_contents[-1]['gt'].append(reference)
            else:
                arranged_contents.append({
                    'context': context,
                    'pred': prediction,
                    'gt': [reference],
                })
        def clean_pred(pred):
            for key_indicator in ['TARGET:', 'Q:', 'R:']:
                if key_indicator in pred:
                    pred = pred.split(key_indicator)[0]
            return pred.strip()
        predictions = [clean_pred(ac['pred']) for ac in arranged_contents]


        references = [ac['gt'] for ac in arranged_contents]
        bleu_score = bleu_scorer.compute(predictions=predictions, references=references)
        # nist_score = nist_mt.compute(predictions=predictions, references=references)
        rouge_score = rouge_scorer.compute(predictions=predictions, references=references)
        meteor_score = meteor_scorer.compute(predictions=predictions, references=references)
        bert_score = bert_scorer.compute(predictions=predictions, references=references, lang='en', device='cuda', batch_size=64)
        bert_f1 = np.asarray(bert_score['f1']).mean()*100
        bert_recall = np.asarray(bert_score['recall']).mean()*100
        bert_precision = np.asarray(bert_score['precision']).mean()*100
        bleu = bleu_score['bleu']
        bleu1, bleu2, bleu3, bleu4 = bleu_score['precisions']
        rouge1, rouge2, rougel = [rouge_score['rouge1'], rouge_score['rouge2'], rouge_score['rougeL']]
        f1 = [f1_scorer(p, t) for p, t in zip(predictions, references)]
        f1 = np.asfarray(f1).mean()
        dist1,dist2 = eval_distinct(predictions)
        # nist = nist_score['nist_mt']
        meteor = meteor_score['meteor']
        folder_name = json_file.split(os.sep)[-2]+json_file.split(os.sep)[-1]
        # folder_name = re.sub(r'-\d\d\d\d-\d\d-\d\d.*','',folder_name)
        csv_row = [folder_name, bleu*100.0, meteor*100.0, rougel*100.0,
                   bleu1*100.0, bleu2*100.0, bleu3*100.0, bleu4*100.0, rouge1*100.0, rouge2*100.0,
                   bert_f1, bert_recall, bert_precision, f1*100, dist1*100, dist2*100, configs]
        csv_data.append(csv_row)
    with open(f'{eval_path}/results.csv', 'w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)
