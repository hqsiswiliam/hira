import csv
import re
from glob import glob
import jsonlines
import numpy as np
from dotenv import load_dotenv
import evaluate
import os
import json

from tqdm import tqdm

load_dotenv('.env')

eval_folder_path = '<fill your result dir here>'

import json
import re
from fraction import Fraction
import sys

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'gsm8k':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'\d+', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[-1]


eval_tasks = [
    'gsm8k',
]

eval_task_json_map = {
    'gsm8k': 'data_file/llm_adapt/gsm8k/test.json'
}
all_data = {}
for eval_task in eval_tasks:
    eval_paths = glob(eval_folder_path)
    eval_path = eval_paths[0]
    jsonl_files = glob(f'{eval_path}/*/*{eval_task}*.jsonl')
    csv_header = ['folder_name', 'acc', 'new acc', 'config']
    csv_data = []
    with open(eval_task_json_map[eval_task], 'r') as file:
        answer_json = json.load(file)
    for json_file in tqdm(jsonl_files):
        print(json_file)
        json_data = []
        lines = []
        with open(json_file, 'r') as f:
            for line in f.readlines():
                lines.append(line)
        lines = [line for line in lines if len(line.strip()) > 0]
        for line in lines:
            try:
                json_data.append(json.loads(line))
            except Exception as e:
                print(e)

        configs = json_data[:2]
        contents = [line for line in json_data if line.__class__ is dict and 'context' in line.keys()]
        assert  len(contents) == len(answer_json), 'num of pred must equal to num of gt'
        answers = []
        new_answers = []
        for _answer, content in zip(answer_json, contents):
            pred = extract_answer(f'{eval_task}', content['pred'])
            gt = extract_answer(f'{eval_task}', content['gt'])
            answers.append(pred == gt)
            pred_num = extract_answer_number(content['pred'])
            ans_num = float(_answer['answer'])
            new_answers.append(pred_num == ans_num)
        acc = np.asarray(answers).mean()
        acc_new = np.asarray(new_answers).mean()
        # folder_name = json_file.split(os.sep)[2]
        # folder_name = re.sub(r'-\d\d\d\d-\d\d-\d\d.*','',folder_name)+os.sep+json_file.split(os.sep)[-1]
        folder_name = json_file.replace(eval_task, '')
        csv_row = [folder_name, acc * 100.0, acc_new * 100 ,configs]
        if folder_name in all_data.keys():
            all_data[folder_name][eval_task] = acc_new * 100
        else:
            all_data[folder_name] = {eval_task: acc_new * 100}
        csv_data.append(csv_row)
    with open(f'{eval_path}/results_{eval_task}.csv', 'w', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        writer.writerows(csv_data)

with open(f'{eval_folder_path}/results_all.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['model', *eval_tasks])
    for folder_name in all_data.keys():
        row = [folder_name]
        for eval_task in eval_tasks:
            row.append(all_data[folder_name].get(eval_task, -1))
        writer.writerow(row)
