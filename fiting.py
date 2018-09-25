import matplotlib.pyplot as plt
import numpy as np
import math
import os
m = 9  # 阶数
num = 100  # 测试数据个数


def draw(W):
    X = np.linspace(0, 1, 100)
    Y = np.linspace(W[0], W[0], 100)
    for i in range(m):
        Y = Y + X**W[i+1]
    print(W)
    plt.plot(X, Y)
    plt.show()


def opti(X, Y):
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)


x = np.linspace(0, 1, num)
X = np.linspace(1, 1, num)
for i in range(m):
    X = np.vstack([X, x**(i+1)])
X = X.transpose()
print(X)
Y = np.sin(x*2*np.pi) + np.random.normal(0, 0.1, (x.size,))  # 添加噪声的Y
W = opti(X,Y)
draw(W)
