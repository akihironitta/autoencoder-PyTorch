import torch
import torchvision
from torchvision import transforms
from modules.Autoencoder import AE
import matplotlib.pyplot as plt
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
                    default=1000,
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

# Configuration using argparse
DATASET = args.dataset
N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch_size
M = args.m

# Configuration
DATA_DIR = "/tmp/data/"
PARAMETER_DIR = "../parameters/"
ACTIVATION = torch.nn.ReLU()
D = 784
H = 400

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
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

# define and restore the trained model
print("Loading the pretrained model...")
model = AE(D=D, H=H, M=M, activation=ACTIVATION)
parameters = torch.load(PARAMETER_DIR+"ae-"+DATASET+"-M"+str(M)+"-E"+str(N_EPOCHS)+".prm", map_location="cpu")
model.load_state_dict(parameters)
model.eval()

for i, (X, Y) in enumerate(data_loader):
    # encode the data
    X = X.view(-1, D)
    Y = Y.numpy()
    Z = model.encode(X)
    Z = Z.detach().numpy()
    break

if M > 2:
    from sklearn.manifold import TSNE
    print("Embedding Z into 2 dimensional space using t-SNE...")
    Z = TSNE(n_components=2).fit_transform(X)

print("Plotting...")
plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap="gist_rainbow", marker=".")
plt.show()
