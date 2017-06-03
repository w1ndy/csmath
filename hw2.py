#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from operator import add
from functools import reduce
from itertools import product
from sklearn.decomposition import PCA
from scipy.spatial import distance

def vectorize_strings(l):
    return reduce(add, list(map(lambda s: list(map(int, s.strip())), l)))

def read_dataset():
    f = open('optdigits-orig.tra')
    inputs = f.readlines()[21:]
    if not inputs[-1]: inputs = inputs[:-1]
    data = [inputs[x : x + 33] for x in range(0, len(inputs), 33)]
    xa = list(map(lambda d: vectorize_strings(d[:-1]), data))
    ya = list(map(lambda d: int(d[32].strip()), data))
    threes = filter(lambda p: p[1] == 3, zip(xa, ya))
    return zip(*threes)

def main():
    dx, _ = read_dataset()
    pca = PCA(n_components=2)
    x = pca.fit_transform(dx)

    xc = np.linspace(-5, 5, 5).tolist()
    yc = np.linspace(5, -5, 5).tolist()
    c = list(product(xc, yc))
    d = list(map(lambda x: 99999, range(len(c))))
    p = list(map(lambda x: [0, 0], range(len(c))))
    i = list(map(lambda x: -1, range(len(c))))

    for t, xs in enumerate(x):
        for k, cs in enumerate(c):
            dist = distance.euclidean(xs, cs)
            if dist < d[k]:
                d[k] = dist
                p[k] = xs
                i[k] = t

    fig = plt.figure(figsize=(10, 4))
    outer = gridspec.GridSpec(1, 2, wspace=0.2)

    left_spec = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    left_ax = plt.Subplot(fig, left_spec[0])
    x, y = zip(*x)
    xp, yp = zip(*p)
    left_ax.set_xticks(xc)
    left_ax.set_yticks(xc)
    left_ax.plot(x, y, 'g.')
    left_ax.plot(xp, yp, 'ro', mfc='none')
    left_ax.grid(True)
    fig.add_subplot(left_ax)

    right_spec = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=outer[1], wspace=-0.4, hspace=0)
    for k, t in enumerate(i):
        right_ax = plt.Subplot(fig, right_spec[k])
        right_ax.imshow(np.array(dx[t]).reshape(32, 32), cmap='gray_r')
        right_ax.set_xticks([])
        right_ax.set_yticks([])
        for sp in right_ax.spines.values(): sp.set_color('red')
        fig.add_subplot(right_ax)

    #plt.imshow(np.array(dx[0]).reshape(32, 32), cmap='gray_r')
    plt.show()

if __name__ == '__main__':
    main()
