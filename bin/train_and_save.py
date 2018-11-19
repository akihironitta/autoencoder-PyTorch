#
# PyTorch implementation of Autoencoder.
# 
# Copyright (c) 2018, Akihiro Nitta.
# All rights reserved.
#

import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from modules.Autoencoder import AE
try:
    import colored_traceback.always
except:
    pass

print("===== MODULES =====")
print("torch:", torch.__version__)
print("torchvision", torchvision.__version__)

import argparse
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    action="store",
                    nargs="?",
                    default=2,
                    type=int)
parser.add_argument("--batch-size",
                    action="store",
                    nargs="?",
                    default=128,
                    type=int)
parser.add_argument("--dataset",
                    action="store",
                    nargs="?",
                    default="mnist",
                    type=str)
parser.add_argument("--n-epochs",
                    action="store",
                    nargs="?",
                    default=100,
                    type=int)
args = parser.parse_args()

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Create directories if not exist.
DATA_DIR = "/tmp/data/"
PARAMETER_DIR = "../parameters/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PARAMETER_DIR):
    os.makedirs(PARAMETER_DIR)

# Hyper-parameters
# Configuration using argparse
DATASET = args.dataset
N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
M = args.m
H = 400
LEARNING_RATE = 1e-3
ACTIVATION = torch.nn.ReLU()

# prep dataset
DATASET = DATASET.lower()
DATA_DIR = DATA_DIR + DATASET + "/"
if DATASET == "mnist":
    D = 784
    ds = torchvision.datasets.MNIST(root=DATA_DIR,
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
elif DATASET == "emnist":
    D = 784
    ds = torchvision.datasets.EMNIST(root=DATA_DIR,
                                     split="letters",
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
elif DATASET == "fmnist":
    D = 784
    ds = torchvision.datasets.FashionMNIST(root=DATA_DIR,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
elif DATASET == "cifar10":
    D = 32*32*3
    ds = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)
elif DATASET == "cifar100":
    D = 32*32*3
    ds = torchvision.datasets.CIFAR100(root=DATA_DIR,
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)
elif DATASET == "svhn":
    D = 32*32*3
    ds = torchvision.datasets.SVHN(root=DATA_DIR,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
else:
    assert False, "Unsupported dataset."

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=ds,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True)

print("===== CONFIG =====")
print("dataset:", DATASET)
print("batch size:", BATCH_SIZE)
print("# of epochs:", N_EPOCHS)
print("dim:", D, "->", H, "->", M)
print("device:", device)

# Define model
model = AE(D=D, H=H, M=M, activation=ACTIVATION).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training
print("===== Start training =====")
for epoch in range(N_EPOCHS):
    for i, (x, y) in enumerate(data_loader):
        # Forward pass
        x = x.to(device).view(-1, D)
        x_reconst, z = model(x)
        
        # compute loss
        loss = F.binary_cross_entropy(x_reconst, x, reduction="sum")
        
        # clear the gradient history
        optimizer.zero_grad()

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

        # show loss
        if (i+1) % 10 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                   .format(epoch+1, N_EPOCHS, i+1, len(data_loader), loss.item()))

# save the model
print("===== Saving the model =====")
filename = "ae"\
           + "-" + DATASET\
           + "-M" + str(M)\
           + "-E" + str(N_EPOCHS)\
           + ".prm"

parameters = model.state_dict()
torch.save(parameters, PARAMETER_DIR+filename, pickle_protocol=4)
