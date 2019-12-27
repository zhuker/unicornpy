import numpy as np

U = np.array([
    [1.2, 0.8],
    [1.4, 0.9],
    [1.5, 1.0],
    [1.2, 0.8]
])

print(U.shape)

I = np.array([
    [1.5, 1.2, 1.0, 0.8],
    [1.7, 0.6, 1.1, 0.4],
])

print(I.shape)
M = U.__matmul__(I)
# M = U.dot(I)
print(M)
