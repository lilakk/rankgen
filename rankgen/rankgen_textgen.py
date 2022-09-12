import torch
import argparse
import os
from rankgen import RankGenGenerator
from rankgen_encoder import RankGenEncoder


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
args = parser.parse_args()

rankgen_encoder = RankGenEncoder(model_path=args.rankgen_encoder, cache_dir=args.cache_dir)
rankgen_generator = RankGenGenerator(rankgen_encoder=rankgen_encoder, language_model="gpt2-medium", cache_dir=args.cache_dir)


def loss_fn(prefix_vector, suffix_vector):
    similarity = torch.matmul(prefix_vector, suffix_vector.t()).squeeze(dim=0)
    return -similarity

def discretize(embedding):
    """
    Given an optimized embedding, find it's nearest neighbor in the embedding space and convert to discrete tokens.
    """
    all_embeddings = rankgen_encoder.model.t5_encoder.shared._parameters['weight']
    similarities = torch.matmul(embedding, all_embeddings.t()).squeeze(dim=0)
    max_index = torch.argmax(similarities)
    return all_embeddings[max_index]

def textgen(prefix, suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    embedding = rankgen_encoder.model.t5_encoder.shared._parameters['weight']
    optimizer = torch.optim.SGD([embedding], lr=0.001, momentum=0.9)
    for i in range(epochs):
        print(f"EPOCH {i}")
        suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
        optimizer.zero_grad()
        for param in rankgen_encoder.model.parameters():
            print(param)
            param.requires_grad = True
        loss = loss_fn(prefix_vector, suffix_vector)
        print(f"loss: {loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
        print(embedding)
    return loss


pre = "For two years, schools and researchers have wrestled with pandemic-era learning setbacks resulting mostly from a lack of in-person classes."
suf = "You"

textgen(pre, suf, 10)
