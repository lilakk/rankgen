import argparse
import json
import torch
import tqdm
import os
import numpy as np
import time
import pdb
from rankgen import RankGenEncoder, RankGenGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="kalpeshk2011/rankgen-t5-xl-pg19", type=str)

parser.add_argument('--cache_dir', default=None, type=str)
args = parser.parse_args()

test_example_file_map = {
    "kalpeshk2011/rankgen-t5-xl-pg19": "/home/ella/rankgen/chapterbreak/chapterbreak_ctx_256.json"
}

rankgen_encoder = RankGenEncoder(args.model_path)

parameters = sum(p.numel() for p in rankgen_encoder.model.parameters())

f = open(test_example_file_map[args.model_path], "r")
data = json.load(f)
pg19 = data['pg19']
ao3 = data['ao3']

avg_scores = []
all_scores = []

# pg19
print('PG19')
for idx, (word_id, d) in tqdm.tqdm(enumerate(pg19.items())):
    for dd in d:
        prefix = dd['ctx']
        cands = [dd['pos']] + dd['negs']
        prefix_vectors = rankgen_encoder.encode([prefix], vectors_type='prefix')['embeddings']
        suffix_vectors = rankgen_encoder.encode(cands, vectors_type='suffix')['embeddings']
        scores = (prefix_vectors * suffix_vectors).sum(dim=1).cpu().tolist()
        avg_scores.append(np.mean([scores[0] > y for y in scores[1:]]))
        all_scores.append(all([scores[0] > y for y in scores[1:]]))
    if (idx + 1) % 50 == 0:
        print(f'{np.mean(avg_scores):.4f} average ({len(avg_scores)} instances). {np.mean(all_scores):.4f} all ({len(all_scores)} instances)')
print(f'{np.mean(avg_scores):.4f} average ({len(avg_scores)} instances). {np.mean(all_scores):.4f} all ({len(all_scores)} instances)')

avg_scores = []
all_scores = []

# ao3
print('AO3')
for idx, (word_id, d) in tqdm.tqdm(enumerate(ao3.items())):
    for dd in d:
        prefix = dd['ctx']
        cands = [dd['pos']] + dd['negs']
        prefix_vectors = rankgen_encoder.encode([prefix], vectors_type='prefix')['embeddings']
        suffix_vectors = rankgen_encoder.encode(cands, vectors_type='suffix')['embeddings']
        scores = (prefix_vectors * suffix_vectors).sum(dim=1).cpu().tolist()
        avg_scores.append(np.mean([scores[0] > y for y in scores[1:]]))
        all_scores.append(all([scores[0] > y for y in scores[1:]]))
    if (idx + 1) % 50 == 0:
        print(f'{np.mean(avg_scores):.4f} average ({len(avg_scores)} instances). {np.mean(all_scores):.4f} all ({len(all_scores)} instances)')
print(f'{np.mean(avg_scores):.4f} average ({len(avg_scores)} instances). {np.mean(all_scores):.4f} all ({len(all_scores)} instances)')