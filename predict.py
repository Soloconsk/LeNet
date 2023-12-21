import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

data = [97.9, 98.6, 97.7, 91.98]
labels = ['no pooling', 'relu', 'sigmoid', 'no act']
plt.bar(range(len(data)), data, tick_label=labels)
plt.show()
