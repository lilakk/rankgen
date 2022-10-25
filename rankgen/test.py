import pickle
import pdb

with open('ensemble_expt/compare_rankgen_comet_full_rerank.jsonl', 'rb') as f:
    ls = pickle.load(f)

ls = ls[:1000]
spearman = [l['spearman'][0] for l in ls]
kendall = [l['kendall'][0] for l in ls]
same_top_1 = [l['same_top_1'] for l in ls]

print(f'full rerank total length: {len(ls)}')
print('  spearman', sum(spearman)/len(spearman))
print('  kendall', sum(kendall)/len(kendall))
print('  same_top_1', sum(same_top_1)/len(same_top_1))

with open('ensemble_expt/compare_rankgen_comet_beam_search.jsonl', 'rb') as f:
    ls = pickle.load(f)

spearman = [l['spearman'][0] for l in ls]
kendall = [l['kendall'][0] for l in ls]
same_top_1 = [l['same_top_1'] for l in ls]

print(f'beam search total length: {len(ls)}')
print('  spearman', sum(spearman)/len(spearman))
print('  kendall', sum(kendall)/len(kendall))
print('  same_top_1', sum(same_top_1)/len(same_top_1))