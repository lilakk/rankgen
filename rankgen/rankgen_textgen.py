import torch
import argparse
import os
import random
import pickle
import tqdm
import time
from transformers import T5Tokenizer
from rankgen import RankGenGenerator
from rankgen_encoder import RankGenEncoder

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
print(f't5_encoder is in training mode: {rankgen_encoder.model.t5_encoder.training}')

for name, param in rankgen_encoder.named_parameters():
    param.requires_grad = False


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.001):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, prev_loss, curr_loss):
        if (curr_loss - prev_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def dot_product_loss(prefix_vector, suffix_vector):
    similarity = torch.matmul(prefix_vector, suffix_vector.t()).squeeze(dim=0)
    return -similarity


def cosine_similarity_loss(prefix_vector, suffix_vector):
    cosine_sim = torch.nn.functional.cosine_similarity(suffix_vector, prefix_vector, dim=0)
    return 1 - cosine_sim


def id_to_token(index):
    if index < tokenizer.sp_model.get_piece_size():
        token = tokenizer.sp_model.IdToPiece(index)
    else:
        token = f"<extra_id_{tokenizer.vocab_size - 1 - index}>"
    return token.replace("▁", "")


def discretize(embedding):
    """
    Given an optimized embedding, find it's nearest neighbor in the embedding space and convert to discrete tokens.
    """
    all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    # similarities = torch.matmul(embedding, all_embeddings.t()).squeeze(dim=0)
    similarities = torch.nn.functional.cosine_similarity(all_embeddings[:tokenizer.sp_model.get_piece_size(),:], embedding.unsqueeze(dim=0))
    print(similarities)
    max_index = torch.argmax(similarities).item()  # find most similar word embedding in embedding table
    token = id_to_token(max_index)
    return token


def oracle_prefix(prefix, suffix_len=50):
    vocab_size = tokenizer.sp_model.get_piece_size()
    if os.path.exists('/home/ella/rankgen/vocab.pkl'):
        with open('/home/ella/rankgen/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = [id_to_token(i) for i in range(vocab_size)]
        with open('/home/ella/rankgen/vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    if os.path.exists('/home/ella/rankgen/vocab_suffix_vectors.pkl'):
        with open('/home/ella/rankgen/vocab_suffix_vectors.pkl', 'rb') as f:
            vocab_vectors = pickle.load(f)
    else:
        vocab_vectors = rankgen_encoder.encode(vocab, return_squeeze=False, vectors_type="suffix")["embeddings"]
        vocab_vectors = torch.stack(vocab_vectors, dim=0)
        with open('/home/ella/rankgen/vocab_suffix_vectors.pkl', 'wb') as f:
            pickle.dump(vocab_vectors, f)
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    for i in range(0, suffix_len):
        similarities = torch.nn.functional.cosine_similarity(vocab_vectors, prefix_vector)
        max_index = torch.argmax(similarities).item()
        word = vocab[max_index]
        prefix = prefix + ' ' + word
        prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
        print(prefix)
    return


def oracle(prefix, suffix_len=50):
    vocab_size = tokenizer.sp_model.get_piece_size()
    if os.path.exists('/home/ella/rankgen/vocab.pkl'):
        with open('/home/ella/rankgen/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = [id_to_token(i) for i in range(vocab_size)]
        with open('/home/ella/rankgen/vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    if os.path.exists('/home/ella/rankgen/vocab_suffix_vectors.pkl'):
        with open('/home/ella/rankgen/vocab_suffix_vectors.pkl', 'rb') as f:
            vocab_vectors = pickle.load(f)
    else:
        vocab_vectors = rankgen_encoder.encode(vocab, return_squeeze=False, vectors_type="suffix")["embeddings"]
        vocab_vectors = torch.stack(vocab_vectors, dim=0)
        with open('/home/ella/rankgen/vocab_suffix_vectors.pkl', 'wb') as f:
            pickle.dump(vocab_vectors, f)
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    words = ''
    for i in range(0, suffix_len):
        print(f'EPOCH {i}')
        if i == 0:
            similarities = torch.nn.functional.cosine_similarity(vocab_vectors, prefix_vector)
            max_index = torch.argmax(similarities).item()
            word = vocab[max_index]
            words = word
        else:
            suffixes = [words + ' ' + v for v in vocab]
            suffix_vectors = rankgen_encoder.encode(suffixes, return_squeeze=False, vectors_type="suffix")["embeddings"]
            suffix_vectors = torch.stack(suffix_vectors, dim=0)
            similarities = torch.nn.functional.cosine_similarity(suffix_vectors, prefix_vector)
            max_index = torch.argmax(similarities).item()
            word = vocab[max_index]
            words += ' ' + word
        print(words)
    return


def initialize_suffix_token():
    all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    index = random.randint(0, all_embeddings.size()[0] - 1)
    return discretize(all_embeddings[index])


def optimize(prefix, suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    new_suffix_tokenized = rankgen_encoder.tokenizer(suffix, return_tensors="pt", padding=True)
    embedding_vector = rankgen_encoder.model.t5_encoder.encoder.embed_tokens
    new_suffix_embedding = embedding_vector(new_suffix_tokenized['input_ids'][0].to(rankgen_encoder.device))
    learned_vector = torch.nn.Parameter(new_suffix_embedding[0:1], requires_grad=True)  # don't optimize </s> token
    optimizer = torch.optim.Adam([learned_vector], lr=0.05)
    losses = []
    tokens = []
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.01)
    for i in range(epochs):
        optimizer.zero_grad()
        suffix_vector = rankgen_encoder.encode(suffix, learned_vector=learned_vector, vectors_type="suffix")["embeddings"]
        loss = cosine_similarity_loss(prefix_vector, suffix_vector)
        if i % 100 == 0:
            print(f"  EPOCH {i}")
            print(f"    loss: {loss}")
            print(learned_vector)
        loss.backward()
        optimizer.step()
        if len(losses) > 0:
            early_stopping(losses[-1], loss)
        if early_stopping.early_stop:
            print(f"stopping at epoch {i}")
            break
        losses.append(loss)
        rankgen_encoder.zero_grad()
        torch.cuda.empty_cache()
    for j in range(learned_vector.size()[0]):
        tokens.append(discretize(learned_vector[j]))
    return tokens


def main():
    pre = "It echoed similar parallels drawn by the Church of Scientology itself, which until then had received scant notice, \
    and was followed by lobbying efforts of Scientology celebrities in Washington. U.S. Department of State spokesman Nicholas \
    Burns rejected the Nazi comparisons in the open letter as \"outrageous\" and distanced the U.S. government from Nazi comparisons \
    made by the Church of Scientology, saying, \"We have criticized the Germans on this, but we aren't going to support the \
    Scientologists' terror tactics against the German government.\" Chancellor Kohl, commenting on the letter, said that those who \
    signed it \"don't know a thing about Germany and don't want to know.\" German officials argued that \"the whole fuss was cranked \
    up by the Scientologists to achieve what we won't give them: tax-exempt status as a religion. This is intimidation, pure and simple.\""
    oracle(pre)
    # for i in range(1):
    #     suf = initialize_suffix_token()
    #     print(f'new token: {suf}')
    #     suf_optim = optimize(pre, suf, 2000)
    #     print(f'token after optim: {suf_optim}')
    #     for token in suf_optim:
    #        suf += " " + token
    #     print(f'suffix seq: {suf}')


main()
