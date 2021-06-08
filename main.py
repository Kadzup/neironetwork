#%%
from N_n import N_n
import numpy as np
import function as f

network = N_n(20, 10, 3)
vector1 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]])
vector2 = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]])
vector3 = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]])

f.learning([vector1, vector2, vector3], network)
f.prediction([vector1, vector2, vector3], network)

# %%
