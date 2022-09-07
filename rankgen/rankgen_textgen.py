import torch
import argparse
import os
from rankgen import RankGenEncoder, RankGenGenerator


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


def textgen(prefix, suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
    optimizer = torch.optim.SGD([suffix_vector], lr=0.001, momentum=0.9)
    for i in range(epochs):
        print(f"Epoch {i}")
        optimizer.zero_grad()
        suffix_vector.requires_grad = True
        for param in rankgen_encoder.model.parameters():
            param.requires_grad = False
        loss = loss_fn(prefix_vector, suffix_vector)
        print(f"loss: {loss}")
        print(f"suffix vector gradient: {suffix_vector.grad}")
        loss.backward()
        optimizer.step()
    return loss


pre = "pre For two years, schools and researchers have wrestled with pandemic-era learning setbacks resulting mostly from a lack of in-person classes."
suf = "suf You"

textgen(pre, suf, 10)
