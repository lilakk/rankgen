import torch
import argparse
import os
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
    return token


def discretize(embedding):
    """
    Given an optimized embedding, find it's nearest neighbor in the embedding space and convert to discrete tokens.
    """
    all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    similarities = torch.matmul(embedding, all_embeddings.t()).squeeze(dim=0)
    max_index = torch.argmax(similarities).item()  # find most similar word embedding in embedding table
    token = id_to_token(max_index)
    return token


def textgen(prefix, suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_tokenized = rankgen_encoder.tokenizer(suffix, return_tensors="pt", padding=True)
    suffix_index = suffix_tokenized['input_ids'][0][0].item()
    for i in range(epochs):
        print(f"EPOCH {i}")
        suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
        loss = cosine_similarity_loss(prefix_vector, suffix_vector)
        print(f"loss: {loss}")
        loss.backward(retain_graph=True)
        grad_emb = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight.grad[suffix_index]
        with torch.no_grad():
            new_emb = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight[suffix_index]
            new_val = new_emb - grad_emb
            rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight[suffix_index].copy_(new_val)
        rankgen_encoder.zero_grad()
    return


def textgen_new_param(prefix, suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_tokenized = rankgen_encoder.tokenizer(suffix, return_tensors="pt", padding=True)
    suffix_index = suffix_tokenized['input_ids'][0][0].item()
    embedding_vector = rankgen_encoder.model.t5_encoder.encoder.embed_tokens
    suffix_embedding = embedding_vector(suffix_tokenized['input_ids'][0].to(rankgen_encoder.device))
    learned_vector = torch.nn.Parameter(suffix_embedding, requires_grad=True)
    optimizer = torch.optim.SGD([learned_vector], lr=0.1, momentum=0.9)
    for i in range(epochs):
        old_weight = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
        print(f"EPOCH {i}")
        optimizer.zero_grad()
        suffix_vector = rankgen_encoder.encode(suffix, learned_vector=learned_vector, vectors_type="suffix")[
            "embeddings"]
        loss = cosine_similarity_loss(prefix_vector, suffix_vector)
        print(f"loss: {loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
        rankgen_encoder.zero_grad()
        words = []
        for j in range(learned_vector.size()[0]):
            words.append(discretize(learned_vector[j]))
        print(words)
    return words


pre = "For two years, schools and researchers have wrestled with pandemic-era learning setbacks."
suf = "This"

textgen_new_param(pre, suf, 100)
