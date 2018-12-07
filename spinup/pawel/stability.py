import numpy as np
from spinup.pawel.kshape import kshape, zscore
import matplotlib.pyplot as plt

def stability_reward(obs):
    bonus = 0
    discount = 0.8
    windows = [1, 3, 5, 7, 13, 19]
    for windndow_index in range(0, len(windows)):
        window = windows[windndow_index]
        num_spans = min(int(len(obs)/window), 6)

        if num_spans == 0:
            continue

        spans = obs[-num_spans * window:].reshape(num_spans, window)
        x = 0
        for i in range(1, num_spans):
            x += (discount**(i-1)) * np.linalg.norm(spans[0] - spans[i])
        x = x / window
        bonus -= x * discount**windndow_index

    # bonus = len(obs)/(np.linalg.norm(obs))
    return bonus


def cluster_traces(ep_obs, epoch_num):

    for obs in ep_obs:
        plt.plot(obs)
    plt.savefig("/tmp/stability/%s-obs.png"%str(epoch_num))
    plt.close()

    window = 64

    spans = []
    for _obs in ep_obs:
        if len(_obs) == 0:
            continue
        obs = np.array(_obs)
        val = obs[0]
        i = 1
        mean = np.mean(obs[i:i+window])
        end = window
        while i < len(obs) - window:
            if i > (end + int(window/2)) or (obs[i] > obs[i-1] and obs[i-1] < mean and obs[i] >= mean):
                end = i + window
                spans.append(np.array(obs[i:end]))
                i = i + int(window/2)
                mean = np.mean(obs[i:i+window])
            else:
                i += 1

    spans = np.array(spans)

    # num_spans = int(len(obs)/window)
    # spans = obs[-num_spans * window:].reshape(num_spans, window)
    print(spans.shape)
    clusters = kshape(zscore(spans, axis=1), 4)
    print (clusters)

    colors = ['r','b','g','y']
    for i in range(len(clusters)):
        cluster = clusters[i]
        color = colors[i]
        # cluster_line = plt.plot(cluster[0])
        # plt.setp(cluster_line, 'color', color, 'linewidth', 3.0)
        for span_num in cluster[1]:
            span_line = plt.plot(spans[span_num])
            plt.setp(span_line, 'color', color, 'linewidth', 1.0)
    plt.savefig("/tmp/stability/%s.png"%str(epoch_num))
    plt.close()
