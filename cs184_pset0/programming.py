import numpy as np
from numpy.linalg import inv

# 1(a)
# Initialize A as a 2D numpy array
a = np.array([[0., 2., 4.], [2., 4., 2.], [3., 3., 1.]])
# Use numpy function inv() to compute the inverse
print("What is inv(A): ", inv(a))

# 1(b)
# Initialize b and c
b = np.array([-2., -2., -4.]).T
c = np.array([1., 1., 1.]).T
# Compute the matrix products
print("What is inv(A)b: ", inv(a) @ b)
print("What is Ac: ", a @ c)

import matplotlib.pyplot as plt

# 2(a)
n = 40000 # 0.0025 = 1/sqrt(4n)
Z = np.random.randn(n)

plt.step(sorted(Z), np.arange(1,n+1)/float(n))

plt.title(r'Empirical $\widehat{F}_n(x)$ for the Standard Normal')
plt.ylabel(r'$\widehat{F}_n(x)$')
plt.xlabel('x')
plt.xlim([-3, 3])
plt.show()

# 2(b)
plt.step(sorted(Z), np.arange(1,n+1)/float(n), label=r"$\widehat{F}_n(x)$")

# Plot for different k values
k_set = [1, 8, 64, 512]
for k in k_set:
    Z = np.sum(np.sign(np.random.randn(n, k))*np.sqrt(1./k), axis=1)
    plt.step(sorted(Z), np.arange(1,n+1)/float(n), label=f"k = {k}")

plt.title(r'Empirical CDFs for the Standard Normal')
plt.ylabel('Empirical CDF')
plt.xlabel('x')
plt.legend(loc="upper left")
plt.xlim([-3, 3])
plt.show()