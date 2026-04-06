import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


n1 = 30
n2 = 40
m1 = 1
m2 = 2

data1 = m1 + np.random.randn(n1)
data2 = m2 + np.random.randn(n2)

# Now calc t value
# generally we use 2 sample indipendent t-test
# this returs the t value and p value
t_stat, p_value = stats.ttest_ind(data1, data2)
print("T value ", t_stat)
print("P value", p_value)

if p_value < 0.05:
    print("The two samples are statistically different")
else:
     print("The two samples are NOT statistically different")




plt.plot(0+np.random.randn(n1)/15, data1,'ro',markerfacecolor='w',markersize=14)
plt.plot(1+np.random.randn(n2)/15, data2,'bs',markerfacecolor='w',markersize=14)
plt.xlim([-1,2])
plt.xticks([0,1],labels=['Group 1','Group 2'])

# set the title to include the t-value and p-value
plt.title(f't = {t_stat:.2f}, p={p_value:.6f}')
plt.show()