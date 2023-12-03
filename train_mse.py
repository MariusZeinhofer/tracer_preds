"""
Trains a U-net with a pixel-based regression loss. 

Graphical output during the training process is written into the 
out_folder which is either named /out or specified as a command
line argument.
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

from unet import UNET
from dataset import MRIDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", 
    help="path to data directory", 
    default="Dataset/",
    type=str,
    )

parser.add_argument(
    "--out_folder", 
    help="path to folder to generic output",
    default="out/", 
    type=str,
    )

parser.add_argument(
    "--epochs", 
    help="number of training epochs",
    default=250, 
    type=int,
    )

parser.add_argument(
    "--batch_size", 
    help="number of samples in each SGD/ADAM step",
    default=8, 
    type=int,
    )

args = parser.parse_args()
argparse_dict = vars(args)

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3

print(f'GPU training is available: {torch.cuda.is_available()}')

# instantiate datasets
data = MRIDataset(args.data_dir + 'Input', args.data_dir + 'Target')
train_data, test_data = random_split(data, [0.8, 0.2])

# instantiate loaders
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# save the test image for inspection
torchvision.utils.save_image(
    torch.tensor(test_data[0][0]), 
    Path.joinpath(Path(args.out_folder), 'test_input.png')
)
torchvision.utils.save_image(
    torch.tensor(test_data[0][1]),
    Path.joinpath(Path(args.out_folder), 'test_target.png')
)

model = UNET(in_channels=1, out_channels=1).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# error metrics
train_metric = torchmetrics.MeanSquaredError().to(DEVICE)
test_metric = torchmetrics.MeanSquaredError().to(DEVICE)
train_metrics = []
test_metrics = []

# training loop
for epoch in range(args.epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        # insert channel dimension 1 and move to gpu if possible
        inputs = inputs.to(device=DEVICE).unsqueeze(1)
        
        # move targets to gpu if possible
        targets = targets.to(device=DEVICE)
        
        # forward pass
        predictions = model(inputs).squeeze()

        # compute loss
        loss = loss_fn(predictions, targets)
        
        # delete previously computed gradients
        optimizer.zero_grad()

        # compute gradient of loss wrt to trainable weights
        loss.backward()
    
        # update parameters
        optimizer.step()

        # compute metric on the batch
        with torch.no_grad():
            train_metric(predictions, targets)
        
    # compute metric over whole dataset and reset state
    train_acc = train_metric.compute()
    train_metrics.append(train_acc.cpu())
    train_metric.reset()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device=DEVICE).unsqueeze(1)
            targets = targets.to(device=DEVICE)

            # forward pass
            predictions = model(inputs).squeeze()
            test_metric(predictions, targets)
        
        test_acc = test_metric.compute()
        test_metrics.append(test_acc.cpu())
        test_metric.reset()

    print(f'Epoch {epoch}, Train Loss {loss}')

    
    # generates output during training loop
    model.eval()
    
    test_im = test_data[0][0]
    test_im = torch.tensor(test_im, dtype=torch.float).reshape(shape=(1,1,256,256))
    x = test_im.to(device=DEVICE)
    with torch.no_grad():
        x = model(x) 

    output = x.squeeze()
    torchvision.utils.save_image(output, args.out_folder + 'prediction' + str(epoch) + '.png')

    model.train()


# save metrics for visualization
np.save('out/train_metrics.npy', np.array(train_metrics))
np.save('out/test_metrics.npy', np.array(test_metrics))

plt.plot(train_metrics)
plt.plot(test_metrics)
plt.show()  
  


