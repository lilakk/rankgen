import torch
import choix
import pandas as pd
import numpy as np
import pdb
import mauve
import pickle
import os
import random
import json
import re
import nltk
import tqdm
import pdb
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from fuzzysearch import find_near_matches
from scipy import stats

df = pd.read_csv('rankgen_data/human_eval.csv')

models = []

for i, row in df.iterrows():
    if row['model_a'] not in models:
        models.append(row['model_a'])
    if row['model_b'] not in models:
        models.append(row['model_b'])

interesting = []
sense = []
humanlike = []

with open('rankgen_data/webtext.test.jsonl', "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

text_data = [re.sub(' +', ' ', re.sub(r"\n", " ", d['text'])) for d in data]
indices = list(range(0, len(text_data)))
dic = {'index': indices, 'text': text_data}
dff = pd.DataFrame(dic)
dff.to_csv('text.csv')

def clean(s):
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", s.strip())
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"&quot;", "\"", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r'(?<=[.,])(?=[^\s])', r' ', cleaned)
    return cleaned.strip()
    # s = s.replace('<p><strong>', '').replace('</strong></p>', '').replace('</p>', '').replace('<p>', ' ').replace('  ', ' ')

if os.path.exists('gt.pkl'):
    with open('gt.pkl', 'rb') as f:
        gt_data = pickle.load(f)
else:
    gt_data = {}
for i, row in tqdm.tqdm(df.iterrows()):
    model_a = row['model_a']
    model_b = row['model_b']
    prefix = clean(row['ctx'])
    if prefix not in gt_data:
        span = process.extractOne(prefix, text_data, scorer=fuzz.partial_ratio)[0]
        match = find_near_matches(prefix, span[0:300], max_l_dist=20)
        gt_suffix = span[match[0].end:].strip()
        gt_data[prefix] = gt_suffix
    if 'a' in row['q1']:
        interesting.append((models.index(model_a), models.index(model_b)))
    else:
        interesting.append((models.index(model_b), models.index(model_a)))
    if 'a' in row['q2']:
        sense.append((models.index(model_a), models.index(model_b)))
    else:
        sense.append((models.index(model_b), models.index(model_a)))
    if 'a' in row['q3']:
        humanlike.append((models.index(model_a), models.index(model_b)))
    else:
        humanlike.append((models.index(model_b), models.index(model_a)))
with open('gt.pkl', 'wb') as f:
    pickle.dump(gt_data, f)

interesting_params = choix.ilsr_pairwise(len(models), interesting)
sense_params = choix.ilsr_pairwise(len(models), sense)
humanlike_params = choix.ilsr_pairwise(len(models), humanlike)

if os.path.exists('mauves.pkl'):
    with open('mauves.pkl', 'rb') as f:
        mauve_scores = pickle.load(f)
else:
    mauve_scores = []
    for m in models:
        pre_gt = []
        pre_gen = []
        dd = df[(df.model_a == m) | (df.model_b == m)]
        for i, row in dd.iterrows():
            pre = clean(row['ctx'])
            gt_suf = gt_data[pre]
            a = row['model_a']
            b = row['model_b']
            if a == m:
                gen_suf = clean(row['completiona'])
            elif b == m:
                gen_suf = clean(row['completionb'])
            gt_suf = gt_suf[0:len(gen_suf)]
            pre_gt.append(pre + ' ' + gt_suf)
            pre_gen.append(pre + ' ' + gen_suf)
            pdb.set_trace()
        mauve_m = mauve.compute_mauve(p_text=pre_gt, q_text=pre_gen, device_id=0, verbose=False)
        mauve_scores.append(mauve_m)
        print(mauve_m)
    with open('mauves.pkl', 'wb') as f:
        pickle.dump(mauve_scores, f)

mauve_scores = [m.mauve for m in mauve_scores]
print(f'interesting: {stats.spearmanr(interesting_params, mauve_scores)}')
print(f'sensible: {stats.spearmanr(sense_params, mauve_scores)}')
print(f'humanlike: {stats.spearmanr(humanlike_params, mauve_scores)}')
