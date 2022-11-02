from sys import prefix
from transformers import T5Tokenizer, T5EncoderModel
import pickle
import argparse
import numpy as np
import tqdm
import os
import torch
import random
import json
import nltk
import pdb
from comet import download_model, load_from_checkpoint
from nltk.tokenize import sent_tokenize
from functools import partial
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sim.sim_models import WordAveraging
from sim.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm
from utils import form_partitions
from utils import truncate
from transformers.utils import logging

nltk.download('punkt')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="rankgen_data/wiki.jsonl", type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--beam_size', default=1, type=int)
parser.add_argument('--num_tokens', default=128, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--model_size', default='medium', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

output_file = f"ensemble_expt/sim_full_rerank.jsonl"

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_per_process_memory_fraction(1.0)

random.seed(49)
random.shuffle(data)

random.seed(442)
random.shuffle(data)

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    output_file = f'{output_file}.shard_{args.local_rank}'

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}", cache_dir=args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}", cache_dir=args.cache_dir)
model.to(device)
model.eval()

tok = TreebankWordTokenizer()

sim_model = torch.load('rankgen/sim/sim/sim.pt')
sim_state_dict = sim_model['state_dict']
sim_vocab_words = sim_model['vocab_words']
sim_args = sim_model['args']

sim_model = WordAveraging(sim_args, sim_vocab_words)
sim_model.load_state_dict(sim_state_dict, strict=True)
sp = spm.SentencePieceProcessor()
sp.Load('rankgen/sim/sim/sim.sp.30k.model')
sim_model.eval()


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


def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1


def scorer_sim(prefix, suffixes):
    with torch.no_grad():
        prefix_eg = make_example(prefix, sim_model)
        prefixes = [prefix_eg for _ in range(len(suffixes))]
        suffixes = [make_example(suffix, sim_model) for suffix in suffixes]
        wx1, wl1, wm1 = sim_model.torchify_batch(prefixes)
        wx2, wl2, wm2 = sim_model.torchify_batch(suffixes)
        BATCH_SIZE = 512
        all_scores = []
        for i in range(0, len(wx1), BATCH_SIZE):
            scores = sim_model.scoring_function(wx1[i:i + BATCH_SIZE], wm1[i:i + BATCH_SIZE], wl1[i:i + BATCH_SIZE],
                                            wx2[i:i + BATCH_SIZE], wm2[i:i + BATCH_SIZE], wl2[i:i + BATCH_SIZE])
            all_scores.extend([x.item() for x in scores])
        return all_scores


def beam_search(contexts, scorer=scorer_sim, beam_size=2, temperature=1.0, top_p=0.9, num_tokens=20, num_samples=10, max_length=115):
        final_outputs = []
        final_scores = []
        total_generated_tokens = 0
        for ctx in contexts:
            beams = [{
                "text": "",
                "eos": False
            } for _ in range(beam_size)]
            while True:
                all_outs = []
                max_new_tokens = min(num_tokens, max_length - total_generated_tokens)
                for beam in beams:
                    # if a beam has ended, add it to all_outs
                    if beam["eos"]:
                        all_outs.append(beam)
                        continue
                    # otherwise generate the next n tokens
                    inputs = tokenizer(ctx + beam['text'], truncation=True, padding="longest",
                                            return_tensors="pt", max_length=1024 - max_new_tokens).to(device)
                    num_input_tokens = len(inputs['input_ids'][0])
                    with torch.inference_mode():
                        curr_outs = model.generate(**inputs, do_sample=True, output_scores=True,
                                                                return_dict_in_generate=True,
                                                                max_new_tokens=max_new_tokens, top_k=None, top_p=top_p,
                                                                num_return_sequences=num_samples, temperature=temperature)
                    is_eos = []
                    for curr_out in curr_outs['sequences']:
                        if tokenizer.eos_token_id in curr_out:
                            is_eos.append(True)
                        else:
                            is_eos.append(False)
                    curr_outs_text = postprocess(curr_outs['sequences'][:, num_input_tokens:])
                    for text, eos in zip(curr_outs_text, is_eos):
                        # update all_outs
                        all_outs.append({
                            "text": beam["text"] + text,
                            "eos": eos
                        })
                # Each beam has total_generated_tokens length
                total_generated_tokens += max_new_tokens
                if len(all_outs) > 1:
                    # skip beam scoring if only one output to choose from
                    scores = scorer(prefix=ctx, suffixes=[x["text"] for x in all_outs])
                    top_scores, top_indices = torch.topk(torch.Tensor(scores), k=beam_size)
                    beams = [all_outs[x] for x in top_indices]  # only track the top k beams
                else:
                    top_scores = torch.Tensor([1.0])
                    top_scores.cuda()
                    beams = all_outs

                for beam in beams:
                    if len(tokenizer.tokenize(beam["text"])) >= max_length:
                        beam["eos"] = True

                if all([x["eos"] for x in beams]):
                    final_outputs.append([x["text"] for x in beams])
                    final_scores.append(top_scores)
                    break
        return final_outputs, final_scores


outputs = []

target_seq_len = []
gen_seq_len = []

logging.set_verbosity_error()

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        outputs = f.read().strip().split("\n")

for kk, instance in tqdm.tqdm(enumerate(data), total=len(data)):
    if kk < len(outputs):
        continue
    beam_text, beam_scores = beam_search(contexts=[instance["prefix"]], scorer=scorer_sim,
                                                           beam_size=args.beam_size,
                                                           top_p=args.top_p, num_tokens=args.num_tokens,
                                                           num_samples=args.num_samples)

    beam_text = beam_text[0]
    beam_text = [truncate(" ".join(x.split())) for x in beam_text]
    outputs.append(json.dumps({
        "prefix": instance["prefix"],
        "targets": instance["targets"][0:1] + beam_text,
        "scores": instance["scores"][0:1] + beam_scores[0].cpu().tolist()
    }))
    target_seq_len.append(len(instance["targets"][0].split()))
    gen_seq_len.append(len(beam_text[0].split()))

    if (kk + 1) % 10 == 0:
        print(f"Avg lens ({kk + 1} instances) = {np.mean(gen_seq_len)} generation, {np.mean(target_seq_len)} target")
        print("Saving file...")
        with open(output_file, "w") as f:
            f.write("\n".join(outputs) + "\n")

with open(output_file, "w") as f:
    f.write("\n".join(outputs) + "\n")
