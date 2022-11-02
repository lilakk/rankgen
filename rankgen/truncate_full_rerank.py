import torch
import argparse
import json
import mauve

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="rankgen_data/wiki.jsonl", type=str)
parser.add_argument('--ranker', default="comet")
parser.add_argument('--num_tokens', default=32, type=int)
args = parser.parse_args()

data_file = f'ensemble_expt/{args.ranker}_full_rerank.jsonl'
with open(data_file, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
data_dict = {x["prefix"]: x for x in data}

with open("rankgen_data/wiki.jsonl", "r") as f:
    raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]

for rid in raw_inp_data:
    assert rid["prefix"] in data_dict
    assert rid["targets"][0] == data_dict[rid["prefix"]]["targets"][0]

all_human = []
all_gen = []
num_tokens = []

