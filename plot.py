"""
Simple visualization of the loss during training. Run after train_mse.py.
"""

import numpy as np
from matplotlib import pyplot as plt

train = np.load('out/train_metrics.npy')[1:]
test = np.load('out/test_metrics.npy')[1:]
x = np.arange(1, len(train) + 1)

fig, (ax) = plt.subplots(1, 1, figsize=(4, 3))

ax.plot(x, train, label='Train Error', color='blue')
ax.plot(x, test, label='Test Error', color='orange')
ax.set_xlabel('Epochs')
ax.set_ylabel('Mean Squared Error')
ax.grid(axis='x', color='0.95')
ax.grid(axis='y', color='0.95')

plt.figlegend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.05),
    fancybox=True, 
    shadow=True, 
    ncol=5
)

plt.savefig(
    'out/plot.png', 
    bbox_inches="tight",
    dpi=400,
    )