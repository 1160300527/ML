import matplotlib.pyplot as plt 
import numpy as np 
import math
import os
def draw():
	x = np.linspace(0,1,66)
	y = x + 1
	plt.plot(x,y)
	plt.show()
m = 9 #阶数
num = 10 #测试数据个数

x = np.linspace(0,1,num)
X = np.linspace(1,1,num)
for i in range(m):
	X = np.vstack([X,x**(i+1
		X = X.transpose()
print(X.shape)
Y = np.sin(x*2*np.pi)
Y = np.random.normal
W = np.linspace(0,0,m)