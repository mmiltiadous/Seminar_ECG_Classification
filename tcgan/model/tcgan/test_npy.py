import numpy as np
import matplotlib.pyplot as plt

array = np.load('final_gen.npy')

plt.plot(array[3])
plt.show()