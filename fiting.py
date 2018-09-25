import matplotlib.pyplot as plt
import numpy as np
import math
import os
m = 9  # 阶数
num = 10  # 测试数据个数
l = math.e**(-20)

def draw(A,B,W):
    x = np.linspace(0.1, 1, num)
    X = np.linspace(1, 1, num)
    for i in range(m):
        X = np.vstack([X, x**(i+1)])
    X = X.transpose()
    plt.plot(x, np.dot(X, W),label = "y = f(w,x)")
    plt.plot(x,np.sin(x*np.pi*2),label="y=sin(2x)")
    plt.scatter(A, B, color = "red", s = 10, label = "train")
    plt.legend()
    plt.show()


def opti(X, Y):
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)


def opti_reg(X,Y):
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX+(np.eye(m+1))*l)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)



def E_RMS():
    return 0


x = np.linspace(0.1, 1, num)
X = np.linspace(1, 1, num)

for i in range(m):
    X = np.vstack([X, x**(i+1)])
X = X.transpose()
Y = np.sin(x*2*np.pi) + np.random.normal(0, 0.1, (x.size,))  # 添加噪声的Y
W = opti_reg(X, Y)
print(W.shape)
draw(x,Y,W)
