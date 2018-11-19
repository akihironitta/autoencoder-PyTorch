#
# PyTorch implementation of Autoencoder.
# 
# Copyright (c) 2018, Akihiro Nitta.
# All rights reserved.
#

import torch
import torch.nn as nn

# Autoencoder model
class AE(nn.Module):
    def __init__(self,
                 D=784,
                 H=400,
                 M=20,
                 activation=nn.ReLU()):
        super(AE, self).__init__()
        # activation function
        self.activate = activation

        # encoder network
        self.encode_h1 = nn.Linear(D, H)
        self.encode_h2 = nn.Linear(H, H)
        self.encode_h3 = nn.Linear(H, M)

        # decoder network
        self.decode_h1 = nn.Linear(M, H)
        self.decode_h2 = nn.Linear(H, H)
        self.decode_h3 = nn.Linear(H, D)
        
    def encode(self, x):
        z = x
        z = self.activate(self.encode_h1(z))
        z = self.activate(self.encode_h2(z))
        z = self.encode_h3(z)
        return z
    
    def decode(self, z):
        x = z
        x = self.activate(self.decode_h1(x))
        x = self.activate(self.decode_h2(x))
        x = self.decode_h3(x)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_reconst = torch.sigmoid(self.decode(z))
        return x_reconst, z
