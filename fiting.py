import matplotlib.pyplot as plt
import numpy as np
import math
m = 9  # 阶数
num = 100  # 训练数据个数
test_num = 100  # 测试数据个数
max_iters = 50000  # 梯度下级最大迭代次数
max_value = 0.0000001  # 停止迭代条件
step = 0.01  # 梯度下级步长
L = -8      #lambda的指数
l = math.e**L   #lambda


def draw(A, B, X_test, Y_test, W, title):   #画函数图像
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
    plt.plot(X, np.sin(X*np.pi*2), label="y=sin(2πx)")
    plt.scatter(A, B, color="red", s=15, label="train")
    plt.scatter(X_test, Y_test, color="green", s=15, label="validation")
    plt.legend()


#解析解方法求解无正则项情况
def bf(X, Y):       
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)


#解析解方法求解有正则项情况
def bf_reg(X, Y):   
    X_T = X.transpose()
    XTX = np.dot(X_T, X)
    oppo = np.linalg.inv(XTX+(np.eye(m+1))*l)
    XTY = np.dot(X_T, Y)
    return np.dot(oppo, XTY)


#梯度下降求解无正则项情况
def gradient(X, Y): 
    W = np.linspace(0, 0, m+1)
    iter = 0
    g = gra(X, Y, W)
    while((iter < max_iters) & (np.sqrt(np.dot(g.transpose(), g)) >= max_value)):
        W = W - step*g
        g = gra(X, Y, W)
        iter += 1
    return W


#梯度下降求解有正则项情况
def gradient_reg(X, Y): 
    W = np.linspace(0, 0, m+1)
    iter = 0
    g = gra_reg(X, Y, W)
    while((iter < max_iters) & (np.sqrt(np.dot(g.transpose(), g)) >= max_value)):
        W = W - step*g
        g = gra_reg(X, Y, W)
        iter += 1
    return W


#共轭梯度求解无正则项情况
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


#共轭梯度求解有正则项情况
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


#根据X,Y,W求解无正则项情况下F(W)梯度
def gra(X, Y, W):      
    XT = X.transpose()
    return np.dot(np.dot(XT, X), W)-np.dot(XT, Y)


#根据X,Y,W求解有正则项情况下F(W)梯度
def gra_reg(X, Y, W):   
    XT = X.transpose()
    return np.dot(np.dot(XT, X), W)-np.dot(XT, Y)+l*W


#损失函数，以最小二乘法计算拟合曲线的损失值
def loss(X_test, Y_test, W):    
    X = np.linspace(1, 1, X_test.size)
    for i in range(m):
        X = np.vstack([X, X_test**(i+1)])
    X = X.transpose()
    Y2 = np.dot(X, W)
    L = Y_test-Y2
    return np.dot(L.transpose(), L)/2


#拟合优度评价
def ERMS(loss, num):            
    return np.sqrt(2*loss/num)


#计算拟合曲线上x的对应值
def answer(x, W):               
    X = np.linspace(1, 1, x.size)
    for i in range(m):
        X = np.vstack([X, x**(i+1)])
    X = X.transpose()
    return np.dot(X, W)


#利用解析解进行求解并画出拟合曲线
def BestFit(X, Y, X_test, Y_test):
    W = bf(X, Y)
    W_reg = bf_reg(X, Y)
    plt.subplot(121)
    draw(x, Y, X_test, Y_test, W, "Best fit without regular terms")
    plt.subplot(122)
    draw(x, Y, X_test, Y_test, W_reg, "Best fit with regular terms")
    plt.text(-0.05, -0.6, "lambda:e^"+str(L), fontsize=8)
    plt.show()


#利用梯度下降法进行曲线拟合并画出结果
def Gradient(X, Y, X_test, Y_test): #通过梯度下降法求解并画出拟合曲线
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


#利用共轭梯度法进行曲线拟合并画出结果
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


x = np.linspace(0, 1, num)  #生成训练集数据
X = np.linspace(1, 1, num)  #初始化范德蒙行列式
for i in range(m):
    X = np.vstack([X, x**(i+1)])    
X = X.transpose()     
Y = np.sin(x*2*np.pi) + np.random.normal(0, 0.1, (x.size,)) # 添加噪声的Y
X_test = np.random.random_sample(test_num,)                 #生成验证集数据 
Y_test = np.sin(X_test*2*np.pi) + np.random.normal(0, 0.1, test_num)#计算验证集对应的Y
BestFit(X, Y, X_test, Y_test)                               #解析解方法
Gradient(X, Y, X_test, Y_test)                              #梯度下降法
Conjugate(X, Y, X_test, Y_test)                             #共轭梯度法
