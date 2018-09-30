import matplotlib.pyplot as plt
import numpy as np
import math
import os
m = 9  # 阶数
num = 100  # 训练数据个数
test_num = 100  # 测试数据个数
max_iters = 50000  # 梯度下级最大迭代次数
max_value = 0.0000001  # 停止迭代条件
step = 0.01  # 梯度下级步长
L = -8
l = math.e**L


def draw(A, B, X_test, Y_test, W, title):
    X = np.linspace(0, 1, 100)
    plt.title(title)
    lossing = loss(X_test, Y_test, W)
    plt.text(-0.05, -1.2, "m:"+str(m)+"\ntrain set:" +
             str(num)+"\nvalidation set:"+str(test_num) +
             "\ntrain set loss:" + str('%.5f' % loss(A, B, W)) +
             "\ntrain set E_RMS:" + str('%.5f' % ERMS(loss(A, B, W), num)) +
             "\nvalidation set loss = " + str('%.5f' % lossing) +
             "\nvalidation set E_RMS = " + str('%.5f' % ERMS(lossing, test_num)), fontsize=8)
    plt.plot(X, answer(X, W), label="y = f(w,x)")
    plt.plot(X, np.sin(X*np.pi*2), label="y=sin(2x)")
    plt.scatter(A, B, color="red", s=15, label="train")
    plt.scatter(X_test, Y_test, color="green", s=15, label="test")
    plt.legend()


def bf(X, Y):
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)


def bf_reg(X, Y):
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX+(np.eye(m+1))*l)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)


def gradient(X, Y):
    W = np.linspace(0, 0, m+1)
    iter = 0
    g = gra(X, Y, W)
    while((iter < max_iters) & (step*np.sqrt(np.dot(g.transpose(), g)) >= max_value)):
        W = W - step*g
        g = gra(X, Y, W)
        iter += 1
    return W


def gradient_reg(X, Y):
    W = np.linspace(0, 0, m+1)
    iter = 0
    g = gra_reg(X, Y, W)
    while((iter < max_iters) & (step*np.sqrt(np.dot(g.transpose(), g)) >= max_value)):
        W = W - step*g
        g = gra_reg(X, Y, W)
        iter += 1
    return W


def conjugate(X, Y):
    XTY = np.dot(X.transpose(), Y)
    XTX = np.dot(X.transpose(), X)
    W = np.linspace(0, 0, m+1)
    R = XTY - np.dot(XTX, W)
    P = R
    k = 0
    while ((k <= m) & (np.sqrt(R.transpose().dot(R)) >= max_value)):
        A = np.dot(R.transpose(), R)/P.transpose().dot(XTX.dot(P))
        W = W + A*P
        R2 = R - A*XTX.dot(P)
        B = R2.transpose().dot(R2)/R.transpose().dot(R)
        P = R2+B*P
        R = R2
        k += 1
    return W


def conjugate_reg(X, Y):
    XTY = np.dot(X.transpose(), Y)
    XTX = np.dot(X.transpose(), X) + l*np.eye(m+1)
    W = np.linspace(0, 0, m+1)
    R = XTY - np.dot(XTX, W)
    P = R
    k = 0
    while ((k <= m) & (np.sqrt(R.transpose().dot(R)) >= max_value)):
        A = np.dot(R.transpose(), R)/P.transpose().dot(XTX.dot(P))
        W = W + A*P
        R2 = R - A*XTX.dot(P)
        B = R2.transpose().dot(R2)/R.transpose().dot(R)
        P = R2+B*P
        R = R2
        k += 1
    return W


def gra(X, Y, W):
    XT = X.transpose()
    return np.dot(np.dot(XT, X), W)-np.dot(XT, Y)


def gra_reg(X, Y, W):
    XT = X.transpose()
    return np.dot(np.dot(XT, X), W)-np.dot(XT, Y)+l*W


def loss(X_test, Y_test, W):
    X = np.linspace(1, 1, X_test.size)
    for i in range(m):
        X = np.vstack([X, X_test**(i+1)])
    X = X.transpose()
    Y2 = np.dot(X, W)
    L = Y_test-Y2
    return np.dot(L.transpose(), L)/2


def ERMS(loss, num):
    return np.sqrt(2*loss/num)


def answer(x, W):
    X = np.linspace(1, 1, x.size)
    for i in range(m):
        X = np.vstack([X, x**(i+1)])
    X = X.transpose()
    return np.dot(X, W)


def Gradient(X, Y, X_test, Y_test):
    W = gradient(X, Y)
    W_reg = gradient_reg(X, Y)
    plt.subplot(121)
    draw(x, Y, X_test, Y_test, W, "Gradient descent without regular terms")
    plt.text(-0.05, -0.6, "step:"+str(step)+"\nmin exit loop:" +
             str(max_value)+"\nlambda:e^"+str(L), fontsize=8)
    plt.subplot(122)
    draw(x, Y, X_test, Y_test, W_reg, "Gradient descent fit with regular terms")
    plt.text(-0.05, -0.6, "step:"+str(step)+"\nmin exit loop:" +
             str(max_value)+"\nlambda:e^"+str(L), fontsize=8)
    plt.show()


def Conjugate(X, Y, X_test, Y_test):
    W = conjugate(X, Y)
    W_reg = conjugate_reg(X, Y)
    plt.subplot(121)
    draw(x, Y, X_test, Y_test, W, "Conjugate gradient without regular terms")
    plt.text(-0.05, -0.6, "min exit loop:"+str(max_value), fontsize=8)
    plt.subplot(122)
    draw(x, Y, X_test, Y_test, W_reg,
         "Conjugate gradient descent fit with regular terms")
    plt.text(-0.05, -0.6, "min exit loop:"+str(max_value) +
             "\nlambda:e^"+str(L), fontsize=8)
    plt.show()


def BestFit(X, Y, X_test, Y_test):
    W = bf(X, Y)
    W_reg = bf_reg(X, Y)
    plt.subplot(121)
    draw(x, Y, X_test, Y_test, W, "Best fit without regular terms")
    plt.subplot(122)
    draw(x, Y, X_test, Y_test, W_reg, "Best fit with regular terms")
    plt.text(-0.05, -0.6, "lambda:e^"+str(L), fontsize=8)
    plt.show()


x = np.linspace(0, 1, num)
X = np.linspace(1, 1, num)
for i in range(m):
    X = np.vstack([X, x**(i+1)])
X = X.transpose()
Y = np.sin(x*2*np.pi) + np.random.normal(0, 0.1, (x.size,))  # 添加噪声的Y
X_test = np.random.random_sample(test_num,)
Y_test = np.sin(X_test*2*np.pi) + np.random.normal(0, 0.1, test_num)
BestFit(X, Y, X_test, Y_test)
Gradient(X, Y, X_test, Y_test)
Conjugate(X, Y, X_test, Y_test)
