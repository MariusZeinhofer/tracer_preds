**Tracer Concentration Prediction with CNNs**

Topic: *Github Repo for the chapter Tracer Concentration Prediction with CNNs of the book: ...*

Abstract: *We present a supervised deep learning pipeline to predict signal increase
ratio (SIR) in the cerebro-spinal fluid (CSF) of the human brain 24h
after intrathecal contrast enhancement agent injection. The section serves as a
brief introduction to deep learning for medical image analysis.*

## Requirements
- pytorch 2.0.0 or later
- numpy 1.23.5 or later
- torchmetrics 0.11.4 or later
- torchvision 0.15.0 or later
- pillow 9.4.0 or later

## Usage
We recommend to start with running train_mse.py which trains a unet
based on the dataset provided. It creates visual output of the predictions.
The script plot.py can be run after train_mse.py to visualize the training 
process.

The structure of the code is as simple as possible. The U-net is implemented
in unet.py, the class for datahandling in dataset.py and the training is in 
train_mse.py.

For this repository we provided a small pre-processed dataset. We assume that
the reader is familiar with data preprocessing through the remaining chapters
of the book.