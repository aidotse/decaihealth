import warnings
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import flwr as fl

# -----------------------------------------------------------------------------
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# ----------------------------------------------------------------------------

# Define simple neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    
# Load MNIST (training and test set)
def load_data(load = 'all', batch_size = 32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
    ])

    # Load train dataset
    trainset = MNIST(root = './data', train = True, download = True, transform=transform)
    if load == 'even':
        indices = (trainset.targets == 0) | (trainset.targets == 2) | (trainset.targets == 4) | (trainset.targets == 6) | (trainset.targets == 8)
        trainset.data, trainset.targets = trainset.data[indices], trainset.targets[indices]
    elif load == 'odd':
        indices = (trainset.targets == 1) | (trainset.targets == 3) | (trainset.targets == 5) | (trainset.targets == 7) | (trainset.targets == 9)
        trainset.data, trainset.targets = trainset.data[indices], trainset.targets[indices]

    # Load test dataset
    testset = MNIST(root = './data', train = False, download = True, transform = transform)

    # Return dataloaders and stats
    #num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return DataLoader(trainset, batch_size = batch_size, shuffle = True), DataLoader(testset) # , num_examples 


# Train the model on the training set
def train(model, device, trainloader, loss_fn, optimizer, epochs = 1):
    model.train()
    for _ in range(epochs):
        for batch, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # Compute prediction error
            pred = model(images)
            loss = loss_fn(pred, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(images)
                print(f"Loss: {loss:>7f}  [{current:>5d}/{len(trainloader.dataset):>5d}]")

            
# Validate the model on the entire test set
def test(model, device, testloader, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    accuracy = correct / len(testloader.dataset) 
    loss = test_loss / len(testloader)
    return accuracy, loss

# -----------------------------------------------------------------------------
# 2. Federation of the pipeline with Flower
# -----------------------------------------------------------------------------

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, device, trainloader, testloader, epochs):
        self.model = model
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        
        # Define loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.device, self.trainloader, self.loss_fn, self.optimizer, self.epochs)
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy, loss = test(self.model, self.device, self.testloader, self.loss_fn)
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
        return loss, len(testloader.dataset), {"accuracy": accuracy}
            
# -----------------------------------------------------------------------------
# "Main" function.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--load_mnist", type=str, default='all') 
    parser.add_argument("--epochs", type=int, default=1)  
    parser.add_argument("--locally", action="store_true") 
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=str, default='8080')
    args = parser.parse_args()

    # Device allocation in PyTorch
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Load model
    model = NeuralNetwork().to(device)
    print(model)

    # Load data
    trainloader, testloader = load_data(args.load_mnist)

    # Train model locally
    if args.locally:
        
        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Train for a number of epochs
        for t in range(args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(model, device, trainloader, loss_fn, optimizer)
            accuracy, loss = test(model, device, testloader, loss_fn)
            print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Loss: {loss:>8f} \n")
        print("Done!")

    # Train federated model
    else:
        client = FlowerClient(model, device, trainloader, testloader, args.epochs)
        fl.client.start_numpy_client(args.host + ':' + args.port, client)
