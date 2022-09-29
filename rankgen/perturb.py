import torch
import argparse
import os
import sys
import pdb
import random
import pickle
import tqdm
import time
import json
from transformers import T5Tokenizer
from rankgen import RankGenGenerator
from rankgen_encoder import RankGenEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--loss_fn', default='l2', type=str)
parser.add_argument('--penalty', action='store_true')
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

f = open("rankgen_data/test_examples/t5_xl_all.jsonl", "r")
data = [json.loads(x) for x in f.read().strip().split("\n")]

tokenizer = T5Tokenizer.from_pretrained("t5-large")
rankgen_encoder = RankGenEncoder(model_path=args.rankgen_encoder, cache_dir=args.cache_dir)
rankgen_generator = RankGenGenerator(rankgen_encoder=rankgen_encoder, language_model="gpt2-medium",
                                     cache_dir=args.cache_dir)

rankgen_encoder.eval()
print(f't5_encoder is in training mode: {rankgen_encoder.model.t5_encoder.training}')

for name, param in rankgen_encoder.named_parameters():
    param.requires_grad = False


def id_to_token(index):
    if index < tokenizer.sp_model.get_piece_size():
        token = tokenizer.sp_model.IdToPiece(index)
    else:
        token = f"<extra_id_{tokenizer.vocab_size - 1 - index}>"
    return token.replace("▁", "")


def discretize(embedding, loss_fn):
    all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    if loss_fn == 'cos_sim':
        similarities = torch.nn.functional.cos_sim(all_embeddings[:tokenizer.sp_model.get_piece_size(),:], embedding.unsqueeze(dim=0))
    elif loss_fn == 'l2':
        similarities = -torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - embedding, dim=1, ord=2)
    elif loss_fn == 'l1':
        similarities = -torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - embedding, dim=1, ord=1)
    elif loss_fn == 'dot':
        similarities = torch.matmul(embedding, all_embeddings[:tokenizer.sp_model.get_piece_size(),:].t()).squeeze(dim=0)
    max_index = torch.argmax(similarities).item()  # find most similar word embedding in embedding table
    token = id_to_token(max_index)
    return token


def oracle_infilling(prefix, suffix_1, suffix_2):
    vocab_size = tokenizer.sp_model.get_piece_size()
    if os.path.exists('/home/ella/rankgen/vocab.pkl'):
        with open('/home/ella/rankgen/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = [id_to_token(i) for i in range(vocab_size)]
        with open('/home/ella/rankgen/vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    vocab = ['Beckham', 'Morris', 'Belknap', 'Goebel', 'Becky', 'Kyra', 'Esther', 'Benjamin', 'Kentucky', 'John']
    if os.path.exists('/home/ella/rankgen/vocab_suffix_vectors.pkl'):
        with open('/home/ella/rankgen/vocab_suffix_vectors.pkl', 'rb') as f:
            vocab_vectors = pickle.load(f)
    else:
        vocab_vectors = rankgen_encoder.encode(vocab, return_squeeze=False, vectors_type="suffix")["embeddings"]
        vocab_vectors = torch.stack(vocab_vectors, dim=0)
        with open('/home/ella/rankgen/vocab_suffix_vectors.pkl', 'wb') as f:
            pickle.dump(vocab_vectors, f)
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    start = time.time()
    suffixes = [suffix_1 + v + suffix_2 for v in vocab]
    suffix_vectors = rankgen_encoder.encode(suffixes, return_squeeze=False, vectors_type="suffix")["embeddings"]
    suffix_vectors = torch.stack(suffix_vectors, dim=0)
    similarities = torch.matmul(suffix_vectors, prefix_vector).squeeze(dim=0)
    print(f'average dot product: {torch.mean(similarities)}')
    print(f'median dot product: {torch.median(similarities)}')
    max_indices = torch.argsort(similarities, descending=True)[:10]
    end = time.time()
    print(f'time taken: {end - start}')
    words = []
    for max_index in max_indices:
        word = vocab[max_index.item()]
        print(f'{word}: {similarities[max_index]}')
        words.append(word)
    return words


def initialize_suffix_token():
    return id_to_token(random.randint(0, tokenizer.sp_model.get_piece_size()+1))


def create_learned_embed(word):
    embedding_vector = rankgen_encoder.model.t5_encoder.encoder.embed_tokens
    tokenized = rankgen_encoder.tokenizer(word, return_tensors="pt", padding=True)
    embedding = embedding_vector(tokenized['input_ids'][0].to(rankgen_encoder.device))
    learned_embed = torch.nn.Parameter(embedding[0:1], requires_grad=True) # don't optimize </s> token
    return learned_embed


def main():

    print(data[0].keys())
    print(data[0]['inputs'])

    # oracle_infilling(pre, suf_1, suf_2)
    # oracle(pre)


main()
