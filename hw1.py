#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def generate_curve(fn=np.sin):
    x = np.arange(0, 2 * math.pi, 0.05)
    y = fn(x)
    return x, y

def generate_points(N):
    x = np.linspace(0, 2 * math.pi, num=N)
    y = np.sin(x) + np.random.normal(scale=0.15, size=N)
    return x, y

def generate_fit(x, y, deg, l):
    model = make_pipeline(PolynomialFeatures(deg), Ridge(alpha=l/2))
    model.fit(x.reshape(-1, 1), y)
    xp = np.arange(0, 2 * math.pi, 0.05)
    yp = model.predict(xp.reshape(-1, 1))
    return xp, yp

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
    draw(10, 9, l=math.exp(-18))

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
