import torch
import choix
import pandas as pd
import numpy as np
import pdb
import mauve
import pickle
import os
import random
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

mauve_data = []
if os.path.exists("mauve_scores.pkl"):
    with open("mauve_scores.pkl", "rb") as f:
        mauve_scores = []
        try:
            while True:
                mauve_scores.append(pickle.load(f))
        except EOFError:
            pass
    pdb.set_trace()
else:
    mauve_scores = []
if len(mauve_scores) < df.shape[0]:
    for i, row in df.iterrows():
        if i < len(mauve_scores):
            'Reading mauve scores from pickle file'
            mauve_a = mauve_scores[i]['mauve_a']
            mauve_b = mauve_scores[i]['mauve_b']
            print(mauve_a.mauve, mauve_b.mauve)
        else:
            "Computing mauve scores"
            mauve_a = mauve.compute_mauve(p_text=row['ctx'], q_text=row['completiona'], device_id=0, verbose=False)
            mauve_b = mauve.compute_mauve(p_text=row['ctx'], q_text=row['completionb'], device_id=0, verbose=False)
            d = {'mauve_a': mauve_a, 'mauve_b': mauve_b}
            print(mauve_a.mauve, mauve_b.mauve)
            mauve_scores.append(d)
            with open("mauve_scores.pkl", "ab+") as f:
                pickle.dump(d, f)
        r = random.randint(0, 1)
        mauve_a = mauve_a.mauve
        mauve_b = mauve_b.mauve
        if mauve_a > mauve_b or (mauve_a == mauve_b and r == 0):
            mauve_data.append((models.index(model_a), models.index(model_b)))
        elif mauve_a < mauve_b or (mauve_a == mauve_b and r == 1):
            mauve_data.append((models.index(model_b), models.index(model_a)))
mauve_params = choix.ilsr_pairwise(len(models), mauve_data)

print(f'interesting: {stats.spearmanr(interesting_params, mauve_params)}')

pdb.set_trace()