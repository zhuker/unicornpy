import numpy as np

ratings = np.array([
    [0, 10, 0, 20],
    [0, 0, 0, 30],
    [11, 0, 40, 0],
])

rows, cols = ratings.nonzero()
print(rows, cols)
p = np.random.permutation(len(rows))
print(p)
rows, cols = rows[p], cols[p]
print(rows, cols)
for row, col in zip(*(rows, cols)):
    print(row, col, ratings[row, col])
