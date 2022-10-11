import torch
import choix
import pandas as pd
import numpy as np
import pdb
import mauve
import pickle
import os
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

for i, row in df.iterrows():
    model_a = row['model_a']
    model_b = row['model_b']
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

interesting_params = choix.ilsr_pairwise(len(models), interesting)
sense_params = choix.ilsr_pairwise(len(models), sense)
humanlike_params = choix.ilsr_pairwise(len(models), humanlike)

if os.path.exists("mauve.pkl"):
    with open("mauve.pkl", "rb") as f:
        mauve_data = pickle.load(f)
else:
    mauve_data = []
    for m in models:
        dd = df[(df.model_a == m) | (df.model_b == m)]
        m_gen = []
        other_gen = []
        for i, d in dd.iterrows():
            if d['model_a'] == m:
                m_gen.append(d['completiona'])
                other_gen.append(d['completionb'])
            else:
                m_gen.append(d['completionb'])
                other_gen.append(d['completiona'])
        mauve_score = mauve.compute_mauve(p_text=m_gen, q_text=other_gen, device_id=0, max_text_length=1024, verbose=False)
        print(mauve_score.mauve)
        mauve_data.append(mauve_score)
    with open("mauve.pkl", "wb") as f:
        pickle.dump(mauve_data, f)

mauve_scores = [m.mauve for m in mauve_data]
print(mauve_scores)
print(f'interesting: {stats.spearmanr(interesting_params, mauve_scores)}')

pdb.set_trace()