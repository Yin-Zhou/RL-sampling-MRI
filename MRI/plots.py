import sys
import os
import numpy as np
import matplotlib.pyplot as plt
input = sys.stdin.readline

# plot the original image
script_dir = os.path.dirname(__file__)
rel_path = "ctmodel/original.npy"
file_path = os.path.join(script_dir,rel_path)
original = np.load(file_path)
s1 = int(np.sqrt(len(original)))
plt.imshow(original.reshape(s1,-1))
plt.show()

# plot the reconstruction image
script_dir = os.path.dirname(__file__)
rel_path = "ctmodel/reconstruction.npy"
file_path = os.path.join(script_dir,rel_path)
reconstruction = np.load(file_path)
s1 = int(np.sqrt(len(reconstruction)))
plt.imshow(reconstruction.reshape(s1,-1))
plt.show()
