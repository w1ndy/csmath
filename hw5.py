#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

def norm_2d(x, y, var, N):
    X = np.random.normal(loc=x, scale=var, size=N)
    Y = np.random.normal(loc=y, scale=var, size=N)
    return np.dstack((X, Y))[0]

def generate_points():
    ptsA = norm_2d(1, 4, 1, 40)
    ptsB = norm_2d(4, 1, 1, 40)
    return np.concatenate([ptsA, ptsB]), np.array([-1] * 40 + [1] * 40)

def svm_cost(w, X, y, l):
    yp = X.dot(w)
    i = np.where(yp * y < 1)
    err = yp[i] - y[i]
    cost = err.transpose().dot(err) + l * w.transpose().dot(w)
    grad = 2 * X[i].transpose().dot(err) + 2 * l * w
    return cost, grad

it = 0
def svm_progress(w):
    global it
    it += 1
    print('Iteration %d' % it)

def svm(X, y, l):
    N, dim = X.shape
    w = np.random.rand(dim + 1)
    nX = np.ones((N, dim + 1))
    nX[:, 1:dim+1] = X
    res = minimize(lambda p: svm_cost(p, nX, y, l), w, method='BFGS', jac=True, callback=svm_progress)
    if not res.success:
        print(res.message)
    else:
        w = res.x
    return w

def main():
    X, y = generate_points()
    w = svm(X, y, 0.01)
    k, b = -w[1] / w[2], -w[0] / w[2]
    ba, bb = -(w[0] + 1) / w[2], -(w[0] - 1) / w[2]
    plt.plot(X[:40, 0], X[:40, 1], 'rx')
    plt.plot(X[40:, 0], X[40:, 1], 'bx')
    plt.plot([0, 5], [b, k * 5 + b], 'g')
    plt.plot([0, 5], [ba, k * 5 + ba], 'g--')
    plt.plot([0, 5], [bb, k * 5 + bb], 'g--')
    plt.show()

if __name__ == '__main__':
    main()
