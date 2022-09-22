import torch
import argparse
import os
import random
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

for name, param in rankgen_encoder.named_parameters():
    if 'weight' in name:
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
    similarities = torch.matmul(embedding, all_embeddings.t()).squeeze(dim=0)
    # print(similarities)
    max_index = torch.argmax(similarities).item()  # find most similar word embedding in embedding table
    token = id_to_token(max_index)
    return token


def oracle(prefix):
    vocab = []
    for i in range(32128):
        vocab.append(id_to_token(i))
    suffix_vectors = [rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"] for suffix in vocab]
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    similarities = [cosine_similarity_loss(prefix_vector, suffix_vector) for suffix_vector in suffix_vectors]
    max_index = max(similarities)
    print(vocab[max_index])
    return


def initialize_suffix_token():
    all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    index = random.randint(0, all_embeddings.size()[0] - 1)
    return discretize(all_embeddings[index])


def optimize(prefix, suffix, new_suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    new_suffix_tokenized = rankgen_encoder.tokenizer(new_suffix, return_tensors="pt", padding=True)
    suffix_index = new_suffix_tokenized['input_ids'][0][0].item()
    for i in range(epochs):
        print(f"EPOCH {i}")
        suffix_vector = rankgen_encoder.encode(suffix + new_suffix, vectors_type="suffix")["embeddings"]
        loss = dot_product_loss(prefix_vector, suffix_vector)
        print(f"loss: {loss}")
        loss.backward(retain_graph=True)
        grad_emb = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight.grad[suffix_index]
        with torch.no_grad():
            emb = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight[suffix_index]
            new_val = emb - grad_emb
            rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight[suffix_index] = new_val
        rankgen_encoder.zero_grad()
    return discretize(rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight.grad[suffix_index])


def optimize_with_new_param(prefix, suffix, new_suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    new_suffix_tokenized = rankgen_encoder.tokenizer(new_suffix, return_tensors="pt", padding=True)
    embedding_vector = rankgen_encoder.model.t5_encoder.encoder.embed_tokens
    new_suffix_embedding = embedding_vector(new_suffix_tokenized['input_ids'][0].to(rankgen_encoder.device))
    learned_vector = torch.nn.Parameter(new_suffix_embedding[:-1], requires_grad=True)  # don't optimize </s> token
    optimizer = torch.optim.Adam([learned_vector], lr=0.01)
    losses = []
    tokens = []
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.001)
    for i in range(epochs):
        print(f"  EPOCH {i}")
        optimizer.zero_grad()
        suffix_vector = \
            rankgen_encoder.encode(suffix + new_suffix, learned_vector=learned_vector, vectors_type="suffix")[
                "embeddings"]
        loss = cosine_similarity_loss(prefix_vector, suffix_vector)
        print(f"    loss: {loss}")
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
    pre = "For two years, schools and researchers have wrestled with pandemic-era learning setbacks."
    oracle(pre)
    suf = ""
    for i in range(1):
        new_suf = initialize_suffix_token()
        print(f'new token: {new_suf}')
        suf_optim = optimize_with_new_param(pre, suf, new_suf, 10000)
        print(f'token after optim: {suf_optim}')
        for token in suf_optim:
            suf += token + " "
        print(f'suffix seq: {suf}')


main()
