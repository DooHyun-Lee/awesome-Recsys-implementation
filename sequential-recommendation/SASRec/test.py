import numpy as np

u = 1
seq = [1,2,3]
seq = np.array(seq)
item_idx = [4,5,6]
item_idx = np.array(item_idx)

print(*[np.array(l) for l in [[u], [seq], item_idx]])