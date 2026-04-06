import math
import random
import numpy as np
import sympy as sym
from sympy.plotting import plot as symplot
import torch

# weights = np.array([ .1,0,.3,-.1,.9 ])
# acts = np.array([ 1,2,-5,3,0 ])


# print("Element-wise multiplication:", acts * weights)
# print("Sum of element-wise multiplication:", np.sum(acts * weights))
# print("Dot product:", np.dot(acts, weights))

# acts_1 = np.arange(0,5)
# acts_2 = np.arange(2,7)

# # print(acts_1)
# # print(acts_2)

# weights_1 = np.linspace(-1,.4,len(acts_1))
# weights_2 = np.linspace(-.4,2,len(acts_2))

# print(weights_1)
# print(weights_2)

# mp = np.zeros((10,2))
# print(mp)


# for i in range(10):
#     mp[i] = i, i**2
#     # equal to this
#     # mp[i][0] = 1
#     # mp[i][1] = 2
   
# print(mp[:, 1 ])
# x = torch.tensor([math.e])
# print("ll =", x.log())

# x = sym.symbols('x')
# lx = sym.log(x)
# dlx = sym.diff(lx)
# print("DLX", dlx)
# fx = -x**4 + 3*x**2
# xv = 4 + (-4)
# dfx = sym.diff(fx, x)
# cd = dfx.subs(x, xv)
# c = fx.subs(x, xv)
# print(f"x = {xv}, f(x) = {c}, f'(x) = {cd}")

# p = symplot((fx, "f(x) = -x⁴ + 3x²"), (dfx, "f'(x) = -4x³ + 6x"), (x, -2, 2), legend=True)





# fx = sym.exp(x**-3)
# sym.pprint(fx)
# print("Derivative:")
# sym.pprint(sym.diff(fx))

# x = np.linspace(1, 3, 3)
# y = np.linspace(1, 3, 3)
# x1,y1 = np.meshgrid(x,y)
# # print("x")
# # print(x)

# # print("y")
# # print(y)

# k = [[1,6,7], [9,8,7], [6,4,5]]

# print(len(k))

# print("x1:")
# print(x1)
# print("y1:")
# print(y1)
# print("\n x*y")
# print (x1*y1 + 2)

# x = [1, 2, 3]
# y = [10, 20, 30]

# for i in range(len(y)):
#     for j in range(len(x)):
#         print(x[j], y[i])

# print ("\n\n\n")
# X, Y = np.meshgrid(x, y)

# print("Mesh x \n", X)
# print("Mesh y \n", Y)
# print ("\n\n\n")
# for i in range(len(X)):
#     for j in range(len(Y)):
#         print(X[i][j], Y[i][j])


# x = 3
# y1 = x**2

# y2 = (x + 0.0000001)**2

# dy = (y2 - y1) / 0.0000001
# print(f"Approximate derivative at x={x}: {dy:.6f}")

# a = [1,2,3]
# b = [i**2 for i in a]

# # list comprehension with nested loops
# print(b)
# xx = [[1,2,3],[4,5,6]]
# flattened = [i2 for i1 in xx for i2 in i1]
# print(flattened)

# v = random.uniform(-1, 1)
# v1 = random.random()
# print(v)
# print(v1)

# a = [1,2,3]
# b = [3,4,5]
# # c = a*b
# print(np.array(a).dot(np.array(b)))
# r = 0
# c = sum([i1+i2 for i1,i2 in zip(a,b)])
# print(c)

# c = [9] + a
# print(c)

# a = 'Ashrit'

# # print(a[1:])
# for c,cw in zip(a, a[1:]):
#     print(c, cw)

m = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]



# # print(m[:][0])
# tm = torch.tensor(m, dtype=torch.int32)
# print(tm[[0,1,2], [1,2,0]])
# # print(tm[1])  # eq to tm[0:, ]
# # print(torch.sum(tm[1]))
# # r = tm[1]/torch.sum(tm[1])
# # print(r)

# r1 = torch.sum(tm, 1)
# print(r1)
# print(r1.shape)
# dr = tm/r1 # this will divide each element of every row with [6,15,25] respectively. 1/6, 2/15, 3/24. 4/6, 5/15 etc...
# print(dr)
# print(dr.shape)

# r2 = torch.sum(tm, 1, keepdim=True)
# dr = tm/r2
# print(dr)
# print(dr.shape)

# a = torch.randn((100,1))
# b = torch.tensor([[3,4],[5,6]])

# c = a@b
# print(c)
# l = (c - 2).pow(2).sum()

# print("storage offset ", b.storage_offset())
# print("stride ", b.stride())
# print("shape ", b.shape)
# print("size ", b.size())
# print("view stride ", b.view(4,-1).stride())
# print("view size ", b.view(4,-1).size())
# print("------")
# print("storage offset ", b.storage_offset())
# print("stride ", b.stride())
# print("shape ", b.shape)

# class Base:
#     def __init__(self, a): 
#         self.a = a
#         print("In base")
#         self._init()
    
#     def _init(self):
#         print("Parent _init fn")
#         self.c = 10

#     def t(self):
#         print("In parent test")
#         self.x = 20

# class C(Base):
#     def _init(self):
#         print("Child _init fn")
#         self.c = 40
    
#     def t(self):
#         print("In child test")
#         return super().t()


# c = C(1)
# c.t()

# print(c.a)
# print(c.c) 
# print(c.x)



# a = [[1, 2], 
#     [3, 4]]
# b = [[10, 20], 
#     [30, 40]]

# ta = torch.tensor(a, dtype=torch.float32, requires_grad=True)
# tb = torch.tensor(b, dtype=torch.float32, requires_grad=True)

# # x = ta.sum(0, keepdim=True)
# # y = ta.sum(1, keepdim=True)

# # print ("X dim 0 shape = ", x.shape)
# # print ("Y dim 1 shape = ", y.shape)

# # print("X = ", x)
# # print("Y = ", y)

# c = ta @ tb

# print (c)
# l = c.sum()
# print(l)
# ta.retain_grad()
# tb.retain_grad()
# c.retain_grad()

# l.backward()

# print ("l_grad", l.grad)
# print("c_grad", c.grad)
# print("ta_grad", ta.grad)
# print("tb_grad", tb.grad)




 # 2,3,2

xa = [
    [
        [1,2],
        [3,4], 
        [5,6]
    ],
    [
        [7,8],
        [9,10],
        [11,12]
    ]
]

a = torch.tensor(xa)
print(a.shape)


b = torch.tensor(
                [
                    [[1,2],[3,4]],
                    [[4,5],[7,8]],
                    [[9,10],[11,12]]
                ]
                ) # size: (3, 2, 2)
print(b.shape)

c = [[10,20, 30],
    [40,50, 60]]
c = torch.tensor(c) # (2,3)
print(c.shape)

print(b @ c)


# # c = a @ b
# # print("a = ",a)
# # print("b = ",b)
# # print("c = ", c)
# # print(c.shape)

# a = torch.randn((32,8,10))
# n = 3
# x = a.view(32, 4, 20)
# print(x.shape)


