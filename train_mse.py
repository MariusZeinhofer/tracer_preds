import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

from unet import UNET
from dataset import MRIDataset


# handling of command line variables
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir", 
    help="path to train directory", 
    default="Data/Train",
    type=str,
    )

parser.add_argument(
    "--test_dir", 
    help="path target directory", 
    default="Data/Test",
    type=str,
    )

parser.add_argument(
    "--out_folder", 
    help="path to folder to generic output", 
    default="out/", 
    type=str,
    )

args = parser.parse_args()
argparse_dict = vars(args)


# some globals
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 8
LEARNING_RATE = 1e-3
EPOCHS        = 250

print(f'GPU training is available: {torch.cuda.is_available()}')

# instantiate datasets
train_data = MRIDataset(args.train_dir + '/Input', args.train_dir + '/Target')
test_data  = MRIDataset(args.test_dir + '/Input', args.test_dir + '/Target')

# instantiate loaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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
for epoch in range(EPOCHS):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        # insert channel dimension 1 and move to gpu if possible
        inputs = inputs.to(device=DEVICE).unsqueeze(1)
        
        # move targets to gpu if possible
        targets = targets.to(device=DEVICE)
        
        # forward pass
        predictions = model(inputs).squeeze()

        # compute loss
        loss = loss_fn(predictions, targets)
        #print(loss.shape)
        
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
  


