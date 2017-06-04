#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

def generate_2d_gaussian(N, meanX, meanY, var):
    x = np.random.normal(loc=meanX, scale=var, size=N)
    y = np.random.normal(loc=meanY, scale=var, size=N)
    return x, y

def generate_data():
    x0, y0 = generate_2d_gaussian(300, 0, 0, 2)
    x1, y1 = generate_2d_gaussian(200, 6, 2, 1)
    x2, y2 = generate_2d_gaussian(200, 3, 5, 2)
    return np.concatenate((x0, x1, x2)), np.concatenate((y0, y1, y2))

def initialize_params(pts, K):
    N = len(pts)
    dim = len(pts[0])
    ci = np.random.randint(N, size=K)

    miu = pts[ci]
    pi = np.zeros((K,))
    sigma = np.zeros((K, dim, dim))

    dist = np.tile(np.sum(pts * pts, 1), (K, 1)).transpose() + \
        np.tile(np.sum(miu * miu, 1), (N, 1)) - \
        2 * pts.dot(miu.transpose())
    m = np.argmin(dist, axis=1)

    for k in range(K):
        pk = pts[np.where(m == k)]
        pi[k] = len(pk) / N
        sigma[k] = np.cov(pk.transpose())

    return miu, pi, sigma

def estimate(pts, K, miu, pi, sigma):
    N = len(pts)
    D = len(pts[0])
    prob = np.zeros((K, N))
    for k in range(K):
        xs = pts - np.tile(miu[k], (N, 1))
        isig = np.linalg.inv(sigma[k])
        t = np.sum(xs.dot(isig) * xs, 1)
        coef = np.power(2 * math.pi, -D / 2) * np.sqrt(np.linalg.det(isig))
        prob[k] = coef * np.exp(-0.5 * t)
    return prob

def gmm(pts, K):
    miu, pi, sigma = initialize_params(pts, K)
    Lprev = -999999
    prob = estimate(pts, K, miu, pi, sigma)

def main():
    x, y = generate_data()
    gmm(np.dstack((x, y))[0], 3)
    #plt.plot(x, y, 'rx')
    #plt.show()

if __name__ == '__main__':
    main()