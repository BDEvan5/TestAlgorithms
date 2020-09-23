import numpy as np


maxes = []
for i in range(1000000):
    rand = np.random.rand() 
    l1 = rand
    l2 = 1- rand

    if np.random.rand() > 0.5:
        l3 = l1 * np.random.rand()
        l1 = l1 - l3
    else:
        l3 = l2 * np.random.rand() 
        l2 = l2 - l3

    max_len = max((l1, l2, l3))
    maxes.append(max_len)

avg = np.mean(maxes)
print(f"Ave max = {avg}")