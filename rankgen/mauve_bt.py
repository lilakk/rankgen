import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from collections import OrderedDict
import scipy.stats
import pickle

THRESHOLD_TIME = 25

with open('mauve.pkl', 'rb') as f:
    mauve_scores = pickle.load(f)
mauve_scores = [m.mauve for m in mauve_scores]
print(mauve_scores)

mauve_scores_raw = {
      "('gpt2', 'p0.9')": 0.8775254868960343,
      "('gpt2', 'p1.0')": 0.5890426203352459,
      "('gpt2-large', 'p0.95')": 0.9358839080944416,
      "('gpt2-large', 'p1.0')": 0.8449972299530251,
      "('gpt2-medium', 'p0.9')": 0.9146017900202308,
      "('gpt2-medium', 'p1.0')": 0.3727859526320347,
      "('gpt2-xl', 'p0.95')": 0.9396265825822094,
      "('gpt2-xl', 'p1.0')": 0.8815674949514485}

mauve_scores = pd.Series(mauve_scores_raw, name="mauve")
mauve_scores.to_frame()

results_fn = 'rankgen_data/human_eval.csv'
df0 = pd.read_csv(results_fn, index_col=0)

player_names = np.array(list(mauve_scores_raw.keys()) + ["human"])
print(player_names)
player_name_to_idx = OrderedDict(enumerate(player_names))


def process_field_name(field_name):
    if 'q1' in field_name:
        final_name = 'Interesting'
    elif 'q2' in field_name:
        final_name = 'Sensible'
    elif 'q3' in field_name:
        final_name = 'Human-like'
    else:
        raise ValueError(f'Unknown name: {field_name}')
    return final_name


def get_model1_v_model2(results, model1, model2):
    df1 = results[(results['model1'] == model1) & (results['model2'] == model2)]
    df2 = results[(results['model2'] == model1) & (results['model1'] == model2)]
    m1_better = df1['m1 better'].sum() + df2['m2 better'].sum()
    m2_better = df1['m2 better'].sum() + df2['m1 better'].sum()
    return m1_better, m2_better


def get_head2head_and_BT_rank(field_name='q3', threshold_time=25, max_iterations=1000):
    df = df0.copy()[df0['te'] > threshold_time]   # Filter all responses made under `threshold_time`
        
    # Collect head2head numbers from the results dataframe
    # Account for randomization of model_a versus model_b for the human eval
    results = []
    for i, m1 in enumerate(player_names):
        for j, m2 in enumerate(player_names): 
            if i <= j: 
                continue
            df1 = df[(df['model_a'] == m1) & (df['model_b'] == m2)]
            df2 = df[(df['model_b'] == m1) & (df['model_a'] == m2)]
            total = df1.shape[0] + df2.shape[0]
            if total == 0: continue
            m1_better = df1[df1[field_name].isin(['1a', '2a'])].shape[0] + df2[df2[field_name].isin(['1b', '2b'])].shape[0]
            m2_better = df2[df2[field_name].isin(['1a', '2a'])].shape[0] + df1[df1[field_name].isin(['1b', '2b'])].shape[0]
            tie = df1[df1[field_name] == '0'].shape[0] + df2[df2[field_name] == '0'].shape[0]
            res = OrderedDict([('model1', m1), ('model2', m2), ('m1 better', m1_better), ('m2 better', m2_better),
                              ('m1 frac', m1_better/total), ('m2 frac', m2_better/total)
                              ])
            results.append(res)
    results = pd.DataFrame(results)  
    
    # Compute B-T preprocessing: collect the head-to-head
    all_results = np.zeros((player_names.shape[0], player_names.shape[0]), dtype=int)  # head-to-head
    wins_per_model = np.zeros(player_names.shape[0], dtype=int)  # total #wins per model
    
    for i, m1 in player_name_to_idx.items():
        total = 0
        for j, m2 in player_name_to_idx.items():
            if m1 != m2:
                t = get_model1_v_model2(results, m1, m2)[0]  # m1 better than m2
                all_results[i, j] = t
                total += t
        wins_per_model[i] = total
        
    # Compute B-T probs
    ps = np.random.rand(player_names.shape[0])
    ps /= ps.sum()
    qs = np.zeros_like(ps)

    # Run iterations of Zeremelo's algorithm. See e.g. for details: 
    # https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    for iteration in range(max_iterations):
        for i in range(player_names.shape[0]):
            denom = sum([(all_results[i, j] + all_results[j, i]) / (ps[i] + ps[j]) 
                         for j in range(player_names.shape[0]) if i != j])
            qs[i] = wins_per_model[i] / denom 
        ps_new = qs / qs.sum()
        if np.linalg.norm(ps_new - ps, 1) < 1e-16:
            # Algorithm converged
            break
        ps = ps_new
    
    # Convert `ps` into logspace and scale them as described in Appendix E.2 of
    # the [paper](https://arxiv.org/pdf/2102.01454.pdf).
    ps = np.log(ps)
    ps -= ps.mean()
    ps *= 100
    
    # Clean up the output
    final_name = process_field_name(field_name)
    out = pd.Series(dict(zip(player_name_to_idx.values(), ps)), name=f'BT/{final_name}')
    return out.sort_values(ascending=False)


h3 = get_head2head_and_BT_rank(field_name='q3', threshold_time=THRESHOLD_TIME)
correlation = scipy.stats.spearmanr(h3.drop("human").sort_index(), mauve_scores.sort_index())
print("Humanlike correlation =", correlation)

h2 = get_head2head_and_BT_rank(field_name='q2', threshold_time=THRESHOLD_TIME)
correlation = scipy.stats.spearmanr(h2.drop("human").sort_index(), mauve_scores.sort_index())
print("Sensible correlation =", correlation)

h1 = get_head2head_and_BT_rank(field_name='q1', threshold_time=THRESHOLD_TIME)
correlation = scipy.stats.spearmanr(h1.drop("human").sort_index(), mauve_scores.sort_index())
print("Creative correlation =", correlation)
