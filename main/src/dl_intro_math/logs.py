import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.001, 1, 50)
logx = np.log(x)

# plot x and log x
plt.figure(figsize=(10,4))

# increase font size. FYI
# plt.rcParams.update({'font.size':15})

plt.plot(x,logx,'ko',markerfacecolor='w')
plt.xlabel('x')
plt.ylabel('log(x)')
plt.show()