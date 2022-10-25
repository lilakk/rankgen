import argparse
import json
import random
import numpy as np
from torch import trunc
import mauve
import pickle
import os
from utils import truncate

parser = argparse.ArgumentParser()
parser.add_argument('--ranker', default="simcse")
parser.add_argument('--gen_key_type', default="second_idx")
parser.add_argument('--data_length', default=7713, type=int)
parser.add_argument('--max_mauve_length', default=768, type=int)
parser.add_argument('--truncate', default=None, type=int)
parser.add_argument('--refresh', action='store_true')
args = parser.parse_args()

with open(f"ensemble_expt/{args.ranker}_full_rerank.jsonl", 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

data_dict = {x["prefix"]: x for x in data}

mauve_output_key = "random_gen_mauve" if "random" in args.gen_key_type else "max_gen_mauve"

with open("rankgen_data/wiki.jsonl", "r") as f:
    raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
for rid in raw_inp_data:
    assert rid["prefix"] in data_dict
    assert rid["targets"][0] == data_dict[rid["prefix"]]["targets"][0]

all_human = []
all_gen = []
num_tokens = []

output_file = f"ensemble_expt/{args.ranker}_full_rerank.mauve.pkl"
if args.truncate:
    data = data[:args.truncate]
    output_file += f"{output_file}.truncate"

for dd in data:
    all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
    if args.gen_key_type == "second_idx":
        # assert len(dd['targets']) != 21
        all_gen.append(dd['prefix'] + ' ' + dd['targets'][1])
        num_tokens.append(len(dd['targets'][1].split()))
    elif args.gen_key_type == "random":
        random_gen = random.choice(dd['targets'][1:])
        all_gen.append(dd['prefix'] + ' ' + random_gen)
        num_tokens.append(len(random_gen.split()))

print(np.mean(num_tokens))

if os.path.exists(output_file):
    with open(output_file, "rb") as f:
        mauve_data = pickle.load(f)
else:
    mauve_data = {}

if mauve_output_key in mauve_data and not args.refresh:
    print(f"Generation score mauve = {mauve_data[mauve_output_key].mauve}")
else:
    mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=args.max_mauve_length, verbose=False)
    print(f"Generation score mauve = {mauve1.mauve}")
    mauve_data[mauve_output_key] = mauve1

    if args.truncate is None:
        with open(output_file, "wb") as f:
            pickle.dump(mauve_data, f)
