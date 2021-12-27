import numpy as np
import random


A = 1000
d = 6
vec = np.random.randint(-20, 20, size=(A, d))

b =  np.random.randint(-20, 20, size=(d, 1))

cos_vec = vec / np.linalg.norm(vec, axis=1).reshape(-1, 1) @ b / np.linalg.norm(b)
        
print(np.sum(cos_vec > 0) / A)
print(np.sum(cos_vec > 3 ** 0.5 / 2) / A)