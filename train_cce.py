import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchmetrics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from unet import UNET
from dataset import MRIDataset
from utility import network_output_to_one_hot, one_hot_to_greyscale


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

# instantiate dataset and split in test and train
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

model = UNET(in_channels=1, out_channels=4).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


#train_metric = torchmetrics.Accuracy(mdmc_average='samplewise', ignore_index=0)
#test_metric = torchmetrics.Accuracy(mdmc_average='samplewise', ignore_index=0)
test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)
#train_metric_list = []
#test_metric_list  = []


# training loop
for epoch in range(1000):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        print(f'inputs {inputs.shape}')
        print(f'targets {targets.shape}')
        # insert channel dimension 1 and move to gpu if possible
        inputs = inputs.to(device=DEVICE).unsqueeze(1)
        
        # move targets to gpu if possible
        targets = 3. * targets.to(device=DEVICE)
        #print(f'targets {targets[0, 120:128, 120:128].long()}')
        

        # forward pass
        predictions = model(inputs)
        print(f'predictions {predictions.shape}')

        # compute loss
        loss = loss_fn(predictions, targets.long())
        #print(loss.shape)
        exit()
      
        # delete previously computed gradients
        optimizer.zero_grad()

        # compute gradient of loss wrt to trainable weights
        loss.backward()
    
        # update parameters
        optimizer.step()

        # train metric on the batch
        #train_acc = train_metric(predictions, torch.tensor(targets, dtype=torch.int16))
        #print(f"Accuracy on batch {batch_idx}: {train_acc}")

        #print("Loss in epoch", epoch, "is", loss.cpu().detach().numpy())
    
    
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device=DEVICE).unsqueeze(1)
            targets = targets.to(device=DEVICE)

            # forward pass
            predictions = model(data)
            test_loss = loss_fn(predictions, targets.long())

            # train metric on the batch
            #test_acc = test_metric(predictions, torch.tensor(targets, dtype=torch.int16))
    print(f'Epoch {epoch}, Train Loss {loss} and Test Loss {test_loss}')

    # compute train metric on whole data and reset internal state afterwards
    #test_acc = test_metric.compute()
        #test_metric_list.append(test_acc.cpu().numpy())
   # print(f"Accuracy on test data: {test_acc}")
    #test_metric.reset()




    # generates output during training loop
    model.eval()
    
    test_im = test_data[0][0]
    test_im = torch.tensor(test_im, dtype=torch.float).reshape(shape=(1,1,256,256))
    x = test_im.to(device=DEVICE)
    with torch.no_grad():
        x = model(x) 

    
    output = one_hot_to_greyscale(network_output_to_one_hot(x))
    torchvision.utils.save_image(output, args.out_folder + 'prediction' + '.png')

    model.train()

    # compute train metric on whole data and reset internal state afterwards
    #train_acc = train_metric.compute()
    #train_metric_list.append(train_acc.cpu().numpy())
    #print(f"Accuracy on train data: {train_acc}")
    #train_metric.reset()

    
  


