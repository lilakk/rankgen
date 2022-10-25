import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import json
import pdb
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rankgen_encoder import RankGenEncoder
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="rankgen_data/wiki.jsonl", type=str)
parser.add_argument('--num_samples', default=10, type=int)
parser.add_argument('--beam_size', default=2, type=int)
parser.add_argument('--num_tokens', default=20, type=int)
parser.add_argument('--max_length', default=115, type=int)
parser.add_argument('--suffix_len', default=128, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--model_size', default='medium', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--ranker', default='comet', type=str)
parser.add_argument('--output_folder', default="suffix_len_expt", type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}", cache_dir=args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}", cache_dir=args.cache_dir)
model.to(device)
model.eval()

raw_data = load_dataset("wikipedia", "20220301.en", cache_dir="/scratch/ella/data")
pdb.set_trace()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
data = data[:1000]

output_file = f"{args.output_folder}/{args.ranker}_suffix_len_{args.suffix_len}.jsonl"

with open(output_file, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]


def postprocess(outputs):
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def truncate(text):
    last_punc = 0
    if "." in text:
        last_punc = max(last_punc, text.rindex("."))
    if "?" in text:
        last_punc = max(last_punc, text.rindex("?"))
    if "!" in text:
        last_punc = max(last_punc, text.rindex("!"))
    if last_punc != 0:
        text = text[:last_punc + 1]
    return text


def scorer_rankgen(rankgen_encoder, prefix, suffixes, prefix_vector=None):
    if prefix_vector is None:
        prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vectors = rankgen_encoder.encode(suffixes, vectors_type="suffix", return_squeeze=False)["embeddings"]
    suffix_vectors = torch.stack(suffix_vectors, dim=0)
    similarities = torch.matmul(prefix_vector, suffix_vectors.t()).squeeze(dim=0)
    return similarities, prefix_vector, suffix_vectors


def scorer_comet(comet_model, prefix, suffixes):
    prefixes = [prefix for _ in range(len(suffixes))]
    dic = []
    for i in range(len(prefixes)):
        dic.append({'src': prefixes[i], 'mt': suffixes[i]})
    seg_scores, sys_score = comet_model.predict(dic, batch_size=8, gpus=1)
    return seg_scores


