import numpy as np
from spinup.pawel.kshape import kshape, zscore 

def stability_reward(obs):
    bonus = 0
    discount = 0.8
    windows = [1, 3, 5, 7, 13, 19]
    for windndow_index in range(0, len(windows)):
        window = windows[windndow_index]
        num_spans = min(int(len(obs)/window), 6)

        if num_spans == 0:
            continue

        spans = obs[-num_spans * window:].reshape(num_spans, window, obs.shape[1])
        x = 0
        for i in range(1, num_spans):
            x += (discount**(i-1)) * np.linalg.norm(spans[0] - spans[i])
        x = x / window
        bonus -= x * discount**windndow_index

    # bonus = len(obs)/(np.linalg.norm(obs))
    return bonus

def cluster_traces(obs):
    clusters = kshape(zscore(obs, axis=1), 2)
