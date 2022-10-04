import torch
import argparse
import os
import sys
import pdb
import re
import random
import pickle
import tqdm
import time
import json
import stanza
import spacy
import nltk
import checklist
import pdb
import numpy as np
from transformers import T5Tokenizer
from rankgen import RankGenGenerator
from rankgen_encoder import RankGenEncoder
from checklist.editor import Editor
from checklist.perturb import Perturb
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import execute_gpt2, cudafy_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--dataset', default='rankgen_data/t5_xl_all_domains_pg19_hard.jsonl', type=str)

args = parser.parse_args()

tokenizer = T5Tokenizer.from_pretrained("t5-large")
rankgen_encoder = RankGenEncoder(model_path=args.rankgen_encoder, cache_dir=args.cache_dir)
rankgen_generator = RankGenGenerator(rankgen_encoder=rankgen_encoder, language_model="gpt2-medium",
                                     cache_dir=args.cache_dir)

rankgen_encoder.eval()

for name, param in rankgen_encoder.named_parameters():
    param.requires_grad = False

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-medium")
model.cuda()
model.eval()

f = open(args.dataset, "r")
data = [json.loads(x) for x in f.read().strip().split("\n")][0:2000]
all_prefixes = [data[i]['prefix'] for i in range(len(data))]
all_suffixes = [data[i]['suffix'] for i in range(len(data))]
all_negatives = [data[i]['negatives'] for i in range(len(data))]

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')


def score(prefix, suffix, ptb_suffix):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
    ptb_suffix_vector = rankgen_encoder.encode(ptb_suffix, vectors_type="suffix")["embeddings"]
    gt_score = torch.matmul(suffix_vector, prefix_vector).squeeze(dim=0)
    ptb_score = torch.matmul(ptb_suffix_vector, prefix_vector).squeeze(dim=0)
    return gt_score, ptb_score


def swap_entities(n=1):
    valid_ne = ['PERSON', 'GPE', 'NORP', 'ORG', 'FAC', 'LOC']
    ne_swapped = []
    suffix_ne = 0
    all_entities_suffix = {}

    for tup in data:
        suffix = tup['prefix']
        doc_suffix = nlp(suffix)
        for ent in doc_suffix.ents:
            if ent.type not in all_entities_suffix:
                all_entities_suffix[ent.type] = [ent.text]
            elif ent.text not in all_entities_suffix[ent.type]:
                all_entities_suffix[ent.type].append(ent.text)

    for tup in data:
        prefix = tup['prefix']
        suffix = tup['suffix']
        negatives = tup['negatives']
        doc_suffix = nlp(suffix)
        suffix_flag = 0
        suffix_candidates = []
        
        # choose candidates for swapping
        for ent in doc_suffix.ents: 
            entity = (ent.text, ent.type, ent.start_char, ent.end_char)
            if ent.type in valid_ne and entity not in suffix_candidates:
                suffix_candidates.append({'entity': entity, 'swapped': False})
        
        n_cands = len(suffix_candidates)
        if n_cands < n:
            continue
        
        new_suffix = suffix
        while suffix_flag < n:
            rand_idx = random.randint(0, n_cands-1)
            if suffix_candidates[rand_idx]['swapped']:
                continue
            cand = suffix_candidates[rand_idx]['entity']
            cand_txt = cand[0]
            label = cand[1]
            begin = cand[2]
            end = cand[3]
            suffix_candidates[rand_idx]['swapped'] = True

            if all([l == cand_txt for l in all_entities_suffix[label]]):
                continue

            prev_flag = suffix_flag
            while suffix_flag == prev_flag:
                rand_idx = random.randint(0, len(all_entities_suffix[label])-1)
                if  all_entities_suffix[label][rand_idx] != cand_txt:
                    new_ne = all_entities_suffix[label][rand_idx]
                    suffix_flag += 1
                    if begin == 0:
                        new_suffix = new_ne + new_suffix[end:]
                    elif end == len(suffix) - 1:
                        new_suffix = new_suffix[:begin] + new_ne
                    else:
                        new_suffix = new_suffix[:begin] + new_ne + new_suffix[end:]
                diff = len(new_ne) - len(cand_txt)
                for e in suffix_candidates:
                    ent = e['entity']
                    if ent[2] > end:
                        e['entity'] = (ent[0], ent[1], ent[2]+diff, ent[3]+diff)
        ne_swapped.append({'prefix':prefix,'suffix':suffix, 'suffix_ptb':new_suffix, 'suffix_ptb_flag':suffix_flag, 'negatives':negatives})
    return ne_swapped


def swap_sents():
    sent_swapped = []
    for tup in data:
        prefix = tup['prefix']
        suffix = tup['suffix']
        negatives = tup['negatives']
        doc_suffix = nlp(suffix)
        suffix_sents = [sent.text for sent in doc_suffix.sentences]
        sent_flag = 0
        if len(suffix_sents) > 1:
            rand_1 = random.randint(0, len(suffix_sents)-1)
            rand_2 = rand_2 = random.randint(0, len(suffix_sents)-1)
            while rand_2 == rand_1:
                rand_2 = random.randint(0, len(suffix_sents)-1)
            s1 = suffix_sents[rand_1]
            s2 = suffix_sents[rand_2]
            suffix_sents[rand_1] = s2
            suffix_sents[rand_2] = s1
            new_suffix = ' '.join(suffix_sents)
            sent_flag = 1
        else:
            new_suffix = ''
        sent_swapped.append({'prefix':prefix,'suffix':suffix, 'suffix_ptb':new_suffix, 'suffix_ptb_flag':sent_flag, 'negatives':negatives})
    return sent_swapped


def insert_sent():
    sent_inserted = []
    for tup in data:
        prefix = tup['prefix']
        suffix = tup['suffix']
        negatives = tup['negatives']
        doc_suffix = nlp(suffix)
        r = random.randint(0, len(negatives)-1)
        negative = negatives[r]
        doc_negative = nlp(negative)
        suffix_sents = [sent.text for sent in doc_suffix.sentences]
        negative_sents = [sent.text for sent in doc_negative.sentences]
        rand_suf = random.randint(0, len(suffix_sents))
        rand_neg = random.randint(0, len(negative_sents)-1)
        suffix_sents.insert(rand_suf, negative_sents[rand_neg])
        new_suffix = ' '.join(suffix_sents)
        insert_flag = 1
        sent_inserted.append({'prefix':prefix,'suffix':suffix, 'suffix_ptb':new_suffix, 'suffix_ptb_flag':insert_flag, 'negatives':negatives})
    return sent_inserted


def antonym_adjective(sent):
    pos = nltk.pos_tag(nltk.word_tokenize(sent))
    flag = 0
    sen =[]
    for i in range(len(pos)):
        w, p = pos[i]
        if p in ['JJ', 'JJR', 'JJS']:
            try:
                syn = Editor().antonyms(sent, w)
            except:
                syn = []
            if len(syn) > 0:
                sen.append(syn[0])
                flag = 1
            else:
                sen.append(w)
        else:
            sen.append(w)
    if flag == 1:
        out = " ".join(x for x in sen)
        return out, flag
    return sent, flag


def negate(n=1):
    negated = []
    for tup in data:
        prefix = tup['prefix']
        suffix = tup['suffix']
        negatives = tup['negatives']
        doc_suffix = nlp(suffix)
        suffix_sents = [sent.text for sent in doc_suffix.sentences]
        if n > len(suffix_sents):
            n = len(suffix_sents)
        indices = []
        indices.append(random.randint(0, len(suffix_sents)-1))
        while len(indices) < n:
            rand = random.randint(0, len(suffix_sents)-1)
            while rand in indices:
                rand = random.randint(0, len(suffix_sents)-1)
            indicies.append(rand)
        neg_flag = 0
        for i in indices:
            sent, flag = antonym_adjective(suffix_sents[i])
            if flag == 0:
                new_suffix = ''
            else:
                neg_flag = 1
                suffix_sents[i] = sent
                new_suffix = ' '.join(suffix_sents)
            negated.append({'prefix':prefix,'suffix':suffix, 'suffix_ptb':new_suffix, 'suffix_ptb_flag':neg_flag, 'negatives':negatives})
    return negated


def compute_gpt2(sequences):
    with torch.no_grad():
        inputs = cudafy_tokens(tokenizer(sequences, return_tensors="pt", padding=True, truncation=True))
        outputs = model(**inputs)
        out_log_probs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        gold_log_probs = torch.gather(out_log_probs[:, :-1, :], 2, inputs['input_ids'][:, 1:].unsqueeze(-1)).squeeze()
        token_mask = inputs['input_ids'][:, 1:] != tokenizer.pad_token_id
        gold_log_probs = gold_log_probs * token_mask
        perplexities = torch.exp(-1 * gold_log_probs.sum(dim=1) / token_mask.sum(dim=1))
        perplexities = perplexities.cpu().tolist()
    return perplexities


def evaluate(task):
    pct = []
    diff = []
    ppl = []
    for i in range(1):
        if task == "swap_entities": task_data = swap_entities(5)
        elif task == "swap_sents": task_data = swap_sents()
        elif task == "insert_sent": task_data = insert_sent()
        elif task == "negate": task_data = negate()
        else:
            print("invalid task")
            return
        pdb.set_trace()
        i_bools = []
        i_diff = []
        i_ppl = []
        n = 0
        for idx, dd in tqdm.tqdm(enumerate(task_data), total=len(task_data)):
            if dd['suffix_ptb_flag'] > 0:
                n += 1
                gt_score, ptb_score = score(dd['prefix'], dd['suffix'], dd['suffix_ptb'])
                i_bools.append(gt_score > ptb_score)
                i_diff.append(gt_score - ptb_score)
                candidates = [dd['suffix'], dd['suffix_ptb']]
                sequences = [dd['prefix'].strip() + " " + x.strip() for x in candidates]
                perplexities = compute_gpt2(sequences)
                i_ppl.append(np.mean([perplexities[0] < y for y in perplexities[1:]]))
        mean_pct = sum(i_bools) / len(i_bools)
        mean_diff = sum(i_diff) / len(i_diff)
        mean_ppl = sum(i_ppl) / len(i_ppl)
        pct.append(mean_pct)
        diff.append(mean_diff)
        ppl.append(mean_ppl)
        # print(f'  % times gt > ptb: {mean_pct}')
        # print(f'  average gt - ptb: {mean_diff}')
        # print(f'  % times gpt2 gt ppl < ptb ppl: {mean_ppl}')
    print(f'  # total instances: {len(task_data)}')
    print(f'  # perturbed instances: {n}')
    print(f'average pct: {sum(pct) / len(pct)}')
    print(f'average diff: {sum(diff) / len(diff)}')
    print(f'average ppl: {sum(ppl) / len(ppl)}')


evaluate("swap_entities")
