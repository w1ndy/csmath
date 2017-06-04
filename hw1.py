#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np

def generate_curve(fn=np.sin):
    x = np.arange(0, 2 * math.pi, 0.05)
    y = fn(x)
    return x, y

sampled = {}

def generate_points(N):
    if N in sampled: return sampled[N]
    x = np.linspace(0, 2 * math.pi, num=N)
    y = np.sin(x) + np.random.normal(scale=0.2, size=N)
    sampled[N] = (x, y)
    return x, y

def generate_fit(x, y, deg, l):
    A = np.zeros((len(x), deg + 1))
    for k in range(deg + 1):
        A[:, k] = np.power(x, k)
    At = A.transpose()
    w = np.linalg.inv(At.dot(A) + l * np.identity(deg + 1)).dot(At).dot(y)
    fn = np.poly1d(w[::-1])
    return generate_curve(fn)
    #return xp, yp

def draw(N, M, l=0):
    xp, yp = generate_points(N)
    xf, yf = generate_fit(xp, yp, M, l)
    plt.plot(xf, yf, 'r')
    plt.plot(xp, yp, 'rx')
    if l == 0:
        plt.xlabel('N=%d, M=%d' % (N, M))
    else:
        plt.xlabel('N=%d, M=%d, ln(l)=%.2f' % (N, M, math.log(l)))

def main():
    xc, yc = generate_curve()

    plt.subplot(321)
    plt.plot(xc, yc, 'g')
    draw(10, 3)

    plt.subplot(322)
    plt.plot(xc, yc, 'g')
    draw(10, 9)

    plt.subplot(323)
    plt.plot(xc, yc, 'g')
    draw(15, 9)

    plt.subplot(324)
    plt.plot(xc, yc, 'g')
    draw(100, 9)

    plt.subplot(325)
    plt.plot(xc, yc, 'g')
    draw(10, 9, l=math.exp(-3))

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
