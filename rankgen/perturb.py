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
from transformers import T5Tokenizer
from rankgen import RankGenGenerator
from rankgen_encoder import RankGenEncoder
from checklist.editor import Editor
from checklist.perturb import Perturb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--cache_dir', default=None, type=str)

args = parser.parse_args()

tokenizer = T5Tokenizer.from_pretrained("t5-large")
rankgen_encoder = RankGenEncoder(model_path=args.rankgen_encoder, cache_dir=args.cache_dir)
rankgen_generator = RankGenGenerator(rankgen_encoder=rankgen_encoder, language_model="gpt2-medium",
                                     cache_dir=args.cache_dir)

rankgen_encoder.eval()

for name, param in rankgen_encoder.named_parameters():
    param.requires_grad = False

f = open("rankgen_data/test_examples/t5_xl_all.jsonl", "r")
data = [json.loads(x) for x in f.read().strip().split("\n")]
all_prefixes = [data[i]['inputs']['inputs_pretokenized'] for i in range(len(data))]
all_suffixes = [data[i]['inputs']['targets_pretokenized'] for i in range(len(data))]
all_negatives = [data[i]['inputs']['negatives_pretokenized'] for i in range(len(data))]

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')


def score(prefix, suffix, ptb_suffix):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
    ptb_suffix_vector = rankgen_encoder.encode(ptb_suffix, vectors_type="suffix")["embeddings"]
    gt_score = torch.matmul(suffix_vector, prefix_vector).squeeze(dim=0)
    ptb_score = torch.matmul(ptb_suffix_vector, prefix_vector).squeeze(dim=0)
    return gt_score, ptb_score


def swap_entities():
    valid_ne = ['PERSON', 'GPE', 'NORP', 'ORG', 'FAC', 'LOC']
    ne_swapped = []
    suffix_ne = 0
    all_entities_suffix = {}

    for tup in data:
        suffix = tup['inputs']['targets_pretokenized']
        doc_suffix = nlp(suffix)
        for ent in doc_suffix.ents:
            if ent.type not in all_entities_suffix:
                all_entities_suffix[ent.type] = [ent.text]
            elif ent.text not in all_entities_suffix[ent.type]:
                all_entities_suffix[ent.type].append(ent.text)

    for tup in data:
        prefix = tup['inputs']['inputs_pretokenized']
        suffix = tup['inputs']['targets_pretokenized']
        negative = tup['inputs']['negatives_pretokenized']
        doc_suffix = nlp(suffix)
        suffix_candidates = []
        suffix_flag = 0
        
        # choose candidates for swapping
        for ent in doc_suffix.ents: 
            entity = (ent.text, ent.type, ent.start_char, ent.end_char)
            if ent.type in valid_ne and entity not in suffix_candidates:
                suffix_candidates.append(entity)

        # if there are candidates swap ONE for same entity type
        if suffix_candidates != []:
            rand_idx = random.randint(0, len(suffix_candidates)-1)
            cand = suffix_candidates[rand_idx]
            cand_txt = suffix_candidates[rand_idx][0]
            label = suffix_candidates[rand_idx][1]
            begin = suffix_candidates[rand_idx][2]
            end = suffix_candidates[rand_idx][3]

            while suffix_flag != 1:
                rand_idx = random.randint(0, len(all_entities_suffix[label])-1)
                if  all_entities_suffix[label][rand_idx] != cand_txt:
                    new_ne = all_entities_suffix[label][rand_idx]
                    suffix_flag = 1
                    suffix_ne+=1
                    if begin == 0:
                        new_suffix = new_ne + suffix[end:]
                    elif end == len(suffix) - 1:
                        new_suffix = suffix[:begin] + new_ne
                    else:
                        new_suffix = suffix[:begin] + new_ne + suffix[end:]
        else:
            new_suffix = ''
        ne_swapped.append({'prefix':prefix,'suffix':suffix, 'suffix_ne':new_suffix, 'suffix_ne_flag':suffix_flag, 'negative':negative})
    return ne_swapped


def swap_sents():
    sent_swapped = []
    for tup in data:
        prefix = tup['inputs']['inputs_pretokenized']
        suffix = tup['inputs']['targets_pretokenized']
        negative = tup['inputs']['negatives_pretokenized']
        doc_suffix = nlp(suffix)
        suffix_sents = [sent.text for sent in doc_suffix.sentences]
        sent_flag = False
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
            sent_flag = True
        else:
            new_suffix = ''
        sent_swapped.append({'prefix':prefix,'suffix':suffix, 'suffix_sent':new_suffix, 'suffix_sent_flag':sent_flag, 'negative':negative})
    return sent_swapped


def insert_sent():
    sent_inserted = []
    for tup in data:
        prefix = tup['inputs']['inputs_pretokenized']
        suffix = tup['inputs']['targets_pretokenized']
        negative = tup['inputs']['negatives_pretokenized']
        doc_suffix = nlp(suffix)
        doc_negative = nlp(negative)
        suffix_sents = [sent.text for sent in doc_suffix.sentences]
        negative_sents = [sent.text for sent in doc_negative.sentences]
        insert_flag = False
        rand_suf = random.randint(0, len(suffix_sents))
        rand_neg = random.randint(0, len(negative_sents)-1)
        suffix_sents.insert(rand_suf, negative_sents[rand_neg])
        new_suffix = ' '.join(suffix_sents)
        insert_flag = True
        sent_inserted.append({'prefix':prefix,'suffix':suffix, 'suffix_insert':new_suffix, 'suffix_insert_flag':insert_flag, 'negative':negative})
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
        prefix = tup['inputs']['inputs_pretokenized']
        suffix = tup['inputs']['targets_pretokenized']
        negative = tup['inputs']['negatives_pretokenized']
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
        neg_flag = False
        for i in indices:
            sent, flag = antonym_adjective(suffix_sents[i])
            if flag == 0:
                negated.append({'prefix':prefix,'suffix':suffix, 'suffix_neg':'', 'suffix_neg_flag':neg_flag, 'negative':negative})
            else:
                neg_flag = True
                suffix_sents[i] = sent
                new_suffix = ' '.join(suffix_sents)
                negated.append({'prefix':prefix,'suffix':suffix, 'suffix_neg':new_suffix, 'suffix_neg_flag':neg_flag, 'negative':negative})
    return negated


def main():
    print("NEGATION")
    pct = []
    diff = []
    for i in range(5):
        neg_data = negate(n=1)
        neg_bools = []
        neg_diff = []
        j = 0
        for d in neg_data:
            if d['suffix_neg_flag'] > 0:
                j += 1
                gt_score, ptb_score = score(d['prefix'], d['suffix'], d['suffix_neg'])
                neg_bools.append(gt_score > ptb_score)
                neg_diff.append(gt_score - ptb_score)
        p = sum(neg_bools) / j
        d = sum(neg_diff) / j
        pct.append(p)
        diff.append(d)
        print(f'  # total instances: {len(data)}')
        print(f'  # perturbed instances: {j}')
        print(f'  # times gt > ptb: {p}')
        print(f'  average gt - ptb: {d}')
    print(f'avg pct: {sum(pct) / len(pct)}')
    print(f'avg diff: {sum(diff) / len(diff)}')

    # print("ENTITY SWAPPING")
    # pct = []
    # diff = []
    # for i in range(5):
    #     entity_data = swap_entities()
    #     entity_bools = []
    #     entity_diff = []
    #     n = 0
    #     for d in entity_data:
    #         if d['suffix_ne_flag'] > 0:
    #             n += 1
    #             gt_score, ptb_score = score(d['prefix'], d['suffix'], d['suffix_ne'])
    #             entity_bools.append(gt_score > ptb_score)
    #             entity_diff.append(gt_score - ptb_score)
    #     p = sum(entity_bools) / n
    #     d = sum(entity_diff) / n
    #     pct.append(p)
    #     diff.append(d)
    #     print(f'  # total instances: {len(data)}')
    #     print(f'  # perturbed instances: {n}')
    #     print(f'  # times gt > ptb: {p}')
    #     print(f'  average gt - ptb: {d}')
    # print(f'avg pct: {sum(pct) / len(pct)}')
    # print(f'avg diff: {sum(diff) / len(diff)}')

    # print("SENTENCE SWAPPING")
    # pct = []
    # diff = []
    # for i in range(5):
    #     sent_data = swap_sents()
    #     sent_bools = []
    #     sent_diff = []
    #     m = 0
    #     for d in sent_data:
    #         if d['suffix_sent_flag']:
    #             m += 1
    #             gt_score, ptb_score = score(d['prefix'], d['suffix'], d['suffix_sent'])
    #             sent_bools.append(gt_score > ptb_score)
    #             sent_diff.append(gt_score - ptb_score)
    #     p = sum(sent_bools) / m
    #     d = sum(sent_diff) / m
    #     pct.append(p)
    #     diff.append(d)
    #     print(f'  # total instances: {len(data)}')
    #     print(f'  # perturbed instances: {m}')
    #     print(f'  # times gt > ptb: {p}')
    #     print(f'  average gt - ptb: {d}')
    # print(f'avg pct: {sum(pct) / len(pct)}')
    # print(f'avg diff: {sum(diff) / len(diff)}')

    # print("SENTENCE INSERTION")
    # pct = []
    # diff = []
    # for i in range(5):
    #     insertion_data = insert_sent()
    #     insertion_bools = []
    #     insertion_diff = []
    #     k = 0
    #     for d in insertion_data:
    #         if d['suffix_insert_flag']:
    #             k += 1
    #             gt_score, ptb_score = score(d['prefix'], d['suffix'], d['suffix_insert'])
    #             insertion_bools.append(gt_score > ptb_score)
    #             insertion_diff.append(gt_score - ptb_score)
    #     p = sum(insertion_bools) / k
    #     d = sum(insertion_diff) / k
    #     pct.append(p)
    #     diff.append(d)
    #     print(f'  # total instances: {len(data)}')
    #     print(f'  # perturbed instances: {k}')
    #     print(f'  # times gt > ptb: {p}')
    #     print(f'  average gt - ptb: {d}')
    # print(f'avg pct: {sum(pct) / len(pct)}')
    # print(f'avg diff: {sum(diff) / len(diff)}')


main()
