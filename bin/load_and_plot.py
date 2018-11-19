import torch
import torchvision
from torchvision import transforms
from modules.Autoencoder import AE
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "/tmp/data/"
PARAMETER_DIR = "../parameters/"
DATASET = "mnist"
N_EPOCHS = 100
filename = "ae-"+DATASET+"-"+str(N_EPOCHS)+".prm"
BATCH_SIZE = 128
ACTIVATION = torch.nn.ReLU()
D = 784
H = 400
M = 2

# prep data
DATASET = DATASET.lower()
DATA_DIR = DATA_DIR + DATASET + "/"
if DATASET == "mnist":
    D = 784
    ds = torchvision.datasets.MNIST(root=DATA_DIR,
                                    train=False,
                                    transform=transforms.ToTensor(),
                                    download=True)
elif DATASET == "emnist":
    D = 784
    ds = torchvision.datasets.EMNIST(root=DATA_DIR,
                                     split="letters",
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)
elif DATASET == "fmnist":
    D = 784
    ds = torchvision.datasets.FashionMNIST(root=DATA_DIR,
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)
elif DATASET == "cifar100":
    D = 32*32*3
    ds = torchvision.datasets.CIFAR100(root=DATA_DIR,
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)
elif DATASET == "svhn":
    D = 32*32*3
    ds = torchvision.datasets.SVHN(root=DATA_DIR,
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True)
else:
    assert False, "Unsupported dataset."

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=ds,
                                          batch_size=1000,
                                          shuffle=True)

# define and restore the trained model
model = AE(D=D, H=H, M=M, activation=ACTIVATION)
parameters = torch.load(PARAMETER_DIR+"ae-"+DATASET+"-"+str(N_EPOCHS)+".prm", map_location="cpu")
model.load_state_dict(parameters)

model.eval()

for i, (X, Y) in enumerate(data_loader):
    
    # Forward pass
    X = X.view(-1, D)
    Y = Y.numpy()
    Z = model.encode(X)
    Z = Z.detach().numpy()
    
    # plot the embeddings
    plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap="gist_rainbow", marker=".")

plt.show()
