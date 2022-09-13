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
    suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
    suffix_len = len(rankgen_encoder.tokenizer(suffix)['input_ids']) # NOTE: this includes the EOS token
    embedding = None
    for param in rankgen_encoder.model.parameters():
        param.requires_grad = False
        if torch.equal(torch.Tensor(list(param.size())), torch.Tensor([suffix_len+1, 2048])):  # +1 to account for [suffi]
            print(param)
            embedding = param
            param.requires_grad = True
    optimizer = torch.optim.SGD([embedding], lr=0.001, momentum=0.9)

    for i in range(epochs):
        print(f"EPOCH {i}")
        optimizer.zero_grad()
        loss = loss_fn(prefix_vector, suffix_vector)
        print(f"loss: {loss}")
        loss.backward(retain_graph=True)
        optimizer.step()
    return loss


pre = "For two years, schools and researchers have wrestled with pandemic-era learning setbacks resulting mostly from a lack of in-person classes."
suf = "You"

textgen(pre, suf, 10)
