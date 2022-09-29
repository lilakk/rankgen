import torch
import argparse
import os
import sys
import pdb
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
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--loss_fn', default='l2', type=str)
parser.add_argument('--penalty', action='store_true')
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

output_file = f'experiments/{args.loss_fn}_scratch_{args.from_scratch}_penalty_{args.penalty}_{args.alpha}_lr_{args.lr}.txt'
sys.stdout = open(output_file, "w")

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


def dot_loss(prefix_vector, suffix_vector):
    similarity = torch.matmul(prefix_vector, suffix_vector.t()).squeeze(dim=0)
    return -similarity


def cos_sim_loss(prefix_vector, suffix_vector, learned_embed, penalty=True, alpha=0.1):
    cosine_sim = torch.nn.functional.cos_sim(suffix_vector, prefix_vector, dim=0)
    if penalty:
        all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
        vocab_dist = torch.mean(torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - learned_embed, dim=1, ord=2))
        return 1 - cosine_sim + alpha * vocab_dist
    else:
        return 1 - cosine_sim


def l2_loss(prefix_vector, suffix_vector, learned_embed, penalty=True, alpha=0.1):
    vector_dist = torch.linalg.norm(suffix_vector-prefix_vector, dim=0, ord=2)
    if penalty:
        all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
        vocab_dist = torch.mean(torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - learned_embed, dim=1, ord=2))
        return vector_dist + alpha * vocab_dist
    else:
        return vector_dist


def l1_loss(prefix_vector, suffix_vector, learned_embed, penalty=True, alpha=0.1):
    vector_dist = torch.linalg.norm(suffix_vector-prefix_vector, dim=0, ord=1)
    if penalty:
        all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
        vocab_dist = torch.mean(torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - learned_embed, dim=1, ord=1))
        return vector_dist + alpha * vocab_dist
    else:
        return vector_dist


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
        start = time.time()
        print(f'EPOCH {i}')
        if i == 0:
            print(vocab_vectors.size())
            print(prefix_vector.size())
            similarities = torch.matmul(vocab_vectors, prefix_vector).squeeze(dim=0)
            max_indices = torch.argsort(similarities, descending=True)[:10]
            for max_index in max_indices:
                word = vocab[max_index.item()]
                print(f'{word}: {similarities[max_index]}')
            words = vocab[max_indices[0].item()]
        else:
            suffixes = [words + ' ' + v for v in vocab]
            suffix_vectors = rankgen_encoder.encode(suffixes, return_squeeze=False, vectors_type="suffix")["embeddings"]
            print(suffix_vectors[100])
            suffix_vectors = torch.stack(suffix_vectors, dim=0)
            similarities = torch.matmul(suffix_vectors, prefix_vector).squeeze(dim=0)
            max_indices = torch.argsort(similarities, descending=True)[:10]
            for max_index in max_indices:
                word = vocab[max_index.item()]
                print(f'{word}: {similarities[max_index]}')
            words += ' ' + vocab[max_indices[0].item()]
        end = time.time()
        print(f'  time taken: {end - start}')
        print(f'  {words}')
    return


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
    # all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    # index = random.randint(0, all_embeddings.size()[0] - 1)
    # return discretize(all_embeddings[index])
    return id_to_token(random.randint(0, tokenizer.sp_model.get_piece_size()+1))
    # r = torch.rand(2048).to(rankgen_encoder.device)


def create_learned_embed(word):
    embedding_vector = rankgen_encoder.model.t5_encoder.encoder.embed_tokens
    tokenized = rankgen_encoder.tokenizer(word, return_tensors="pt", padding=True)
    embedding = embedding_vector(tokenized['input_ids'][0].to(rankgen_encoder.device))
    learned_embed = torch.nn.Parameter(embedding[0:1], requires_grad=True) # don't optimize </s> token
    return learned_embed


def optimize(prefix, suffix, new_suffix, epochs, loss_fn='cos_sim', penalty=True, alpha=0.1):
    '''
    Given a prefix and suffix, generate the next suffix token.
    '''
    print(f'using loss function {loss_fn} with {penalty} penalty, alpha = {alpha}')
    print(f'init suffix token: {new_suffix}')
    all_embeddings = rankgen_encoder.model.t5_encoder.encoder.embed_tokens.weight
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    learned_embed = create_learned_embed(new_suffix)
    optimizer = torch.optim.Adam([learned_embed], lr=args.lr)
    losses = []
    tokens = []
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)
    for i in range(epochs):
        optimizer.zero_grad()
        suffix_vector = rankgen_encoder.encode(suffix, learned_embed=learned_embed, vectors_type="suffix")["embeddings"]
        if loss_fn == 'cos_sim':
            loss = cos_sim_loss(prefix_vector, suffix_vector, learned_embed, penalty=penalty, alpha=alpha)
        elif loss_fn == 'l2':
            loss = l2_loss(prefix_vector, suffix_vector, learned_embed, penalty=penalty, alpha=alpha)
        elif loss_fn == 'l1':
            loss = l1_loss(prefix_vector, suffix_vector, learned_embed, penalty=penalty, alpha=alpha)
        elif loss_fn == 'dot':
            loss = dot_loss(prefix_vector, suffix_vector)
        if i % 500 == 0:
            print(f"EPOCH {i}")
            print(f"  {loss_fn}_loss: {loss}")
            print(f'  suffix vector: {suffix_vector}')
            print(f'  vector l2 distance: {torch.linalg.norm(suffix_vector-prefix_vector, dim=0, ord=2)}')
            print(f'  vector l1 distance: {torch.linalg.norm(suffix_vector-prefix_vector, dim=0, ord=1)}')
            print(f'  l2 dist from vocab: {torch.mean(torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - learned_embed, dim=1, ord=2))}')
            print(f'  l1 dist from vocab: {torch.mean(torch.linalg.norm(all_embeddings[:tokenizer.sp_model.get_piece_size(),:] - learned_embed, dim=1, ord=1))}')
            print(f'  learned embed: {learned_embed}')
        loss.backward()
        optimizer.step()
        if len(losses) > 0:
            early_stopping(losses[-1], loss)
        if early_stopping.early_stop:
            print(f"STOPPING AT EPOCH {i}")
            break
        losses.append(loss)
        rankgen_encoder.zero_grad()
        torch.cuda.empty_cache()
    return discretize(learned_embed[0], loss_fn=loss_fn)


def optimize_clamping(prefix, suffix, epochs=500, k=100):
    '''
    Given a prefix and a suffix, generate the next suffix token using the clamping trick.
    '''
    repeats = epochs // k
    new_suffix = initialize_suffix_token()
    print(f'initial suffix: {new_suffix}')
    for i in range(repeats):
        new_suffix = optimize(prefix, suffix, new_suffix, epochs=k, loss_fn='dot')
        print(f'clamped suffix: {new_suffix}')
    return new_suffix


def main():
    # pre = "It echoed similar parallels drawn by the Church of Scientology itself, which until then had received scant notice, \
    # and was followed by lobbying efforts of Scientology celebrities in Washington. U.S. Department of State spokesman Nicholas \
    # Burns rejected the Nazi comparisons in the open letter as \"outrageous\" and distanced the U.S. government from Nazi comparisons \
    # made by the Church of Scientology, saying, \"We have criticized the Germans on this, but we aren't going to support the \
    # Scientologists' terror tactics against the German government.\" Chancellor Kohl, commenting on the letter, said that those who \
    # signed it \"don't know a thing about Germany and don't want to know.\" German officials argued that \"the whole fuss was cranked \
    # up by the Scientologists to achieve what we won't give them: tax-exempt status as a religion. This is intimidation, pure and simple.\""
    # suf_1 = "Officials explained that precisely because of Germany's Nazi past, Germany took a determined "
    # suf_2 = " against all \"radical cults and sects, including right-wing Nazi groups\", and not just against Scientology."

    pre = "The couple had two sons. As governor, Beckham sought to unite his party and the state by supporting changes to the \
    blatantly-partisan Goebel Election Law, which had been authored by his late running mate while the latter was a member of the \
    General Assembly. He stressed non-controversial issues, such as improvements to roads and the state's educational system. He \
    recommended passage of a law to set uniform school textbook prices, a reform that both he and Goebel had advocated during the \
    gubernatorial campaign. However, his passive leadership ensured that the General Assembly did little to address his agenda. The only \
    major pieces of legislation passed during Beckham's term were a tax increase that added a half million dollars to the state's revenue \
    and a child labor law that forbade children under fourteen to work without their parents' consent. Although the Kentucky Constitution \
    prohibited governors from serving consecutive terms, Beckham announced that he would seek a full term as governor in 1903. His candidacy \
    was challenged in court, but the court ruled Beckham had not served a full first term and so was eligible to run."
    suf_1 = "His record of reconciliation and of supporting non-controversial reforms prevented significant opposition when he won the \
    party's nomination. His record also deprived his Republican opponent, Morris B. Belknap, of any significant campaign issue in the general election. "
    suf_2 = " defeated Belknap and three minor candidates."
    suf = "His record of reconciliation and of supporting non-controversial reforms prevented significant opposition when he won the party's nomination. His record also deprived his Republican opponent, Morris B. Belknap, of any significant campaign issue in the general election. Beckham defeated Belknap and three minor candidates. "    # prefix_vector = rankgen_encoder.encode(pre, vectors_type="prefix")["embeddings"]
    # suffix_vector = rankgen_encoder.encode(suf, vectors_type="suffix")["embeddings"]
    # print(dot_loss(prefix_vector, suffix_vector))

    # oracle_infilling(pre, suf_1, suf_2)
    # oracle(pre)

    # suf = 'His record of reconciliation and of supporting non-controversial reforms prevented significant opposition when he won the party\'s '
    # # suf = 'Today\'s weather is '
    suffix = ''
    for i in range(3):
        new_suffix = initialize_suffix_token()
        if args.from_scratch:
            suf = ''
            suf_optim = optimize(pre, suf, new_suffix, args.epochs, loss_fn=args.loss_fn, penalty=args.penalty, alpha=args.alpha)
        else:
            suf_optim = optimize(pre, suf, new_suffix, args.epochs, loss_fn=args.loss_fn, penalty=args.penalty, alpha=args.alpha)
        print(f'token after optim: {suf_optim}')
        suffix += ' ' + suf_optim
        print(f'suffix seq: {suffix}')


main()
