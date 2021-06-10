"""
Score the permutations - so for a story, put it in the right order
"""
import sys
sys.path.append('../../')
import h5py
from utils.encode.encoder import get_encoder, MASK
import numpy as np
import itertools
import scipy
from scipy import stats
from tqdm import tqdm
import pandas as pd

def score_permutation(log_probs_out_resh, xa_perm, xb_perm):
    assert len(xa_perm) == len(xb_perm)
    our_probs_eq = np.ones([len(xa_perm), len(xa_perm)])
    our_probs_gtlt = np.ones([len(xa_perm), len(xa_perm)])

    for i, tp_i in enumerate(xa_perm):
        for j, tp_j in enumerate(xb_perm):
            if tp_i == tp_j:
                our_probs_eq[i, j] = log_probs_out_resh[i, j, 0]
            elif tp_i < tp_j:
                our_probs_gtlt[i, j] = log_probs_out_resh[i, j, 1]
            elif tp_i > tp_j:
                our_probs_gtlt[i, j] = log_probs_out_resh[i, j, 2]
    return our_probs_eq, our_probs_gtlt

############
def spearman_acc(story):
    return scipy.stats.spearmanr(story, [0,1,2,3,4])[0]

def absolute_distance(story):
    return np.mean(np.abs(np.array(story) - np.array([0,1,2,3,4])))


def pairwise_acc(story):
    correct = 0
    total = len(story) * (len(story)-1) // 2
    for idx1 in range(len(story)):
        for idx2 in range(idx1+1, len(story)):
            if story[idx1] < story[idx2]:
                correct += 1
    return correct/total

encoder = get_encoder()

SPLIT_NAME = 'val'
h5 = h5py.File(f'logits_{SPLIT_NAME}.h5', 'r')
print(h5, flush=True)


keys = sorted([int(x) for x in h5.keys()])
all_stories = []
for k in tqdm(keys):
    grp_k = h5[str(k)]
    sents_k = np.array(grp_k['sentences'])
    lv_scores = np.array(grp_k['lang_viz_probs'])

    perm_to_prob = {}
    for perm_i in itertools.permutations(list(range(5))):
        lv_match, lv_gtlt = score_permutation(lv_scores, xa_perm=np.arange(5), xb_perm=perm_i)
        lv_match = np.log(lv_match).sum()
        lv_gtlt = np.log(lv_gtlt).sum()
        perm_to_prob[tuple(perm_i)] = lv_match + lv_gtlt

    (best_perm, best_score), (second_best_perm, second_best_score) = sorted(perm_to_prob.items(), key=lambda x: -x[1])[:2]
    all_stories.append(best_perm)


print('Spearman:')
print(np.mean([spearman_acc(st) for st in all_stories]))

print('Absoulte Distance:')
print(np.mean([absolute_distance(st) for st in all_stories]))

print('Pairwise:')
print(np.mean([pairwise_acc(st) for st in all_stories]))


# Compare with CLIP or any other baseline
all_stories_clip = pd.read_csv(f'clip_predictions_{SPLIT_NAME}.tsv', delimiter='\t', names=['story'])['story'].apply(lambda x: [int(y) for y in x.split(',')]).tolist()

print("CLIP", flush=True)
print('Spearman:')
print(np.mean([spearman_acc(st) for st in all_stories_clip]))

print('Absoulte Distance:')
print(np.mean([absolute_distance(st) for st in all_stories_clip]))

print('Pairwise:')
print(np.mean([pairwise_acc(st) for st in all_stories_clip]))