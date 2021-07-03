import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# PyTorch offers domain specific libraries such as TorchText, TorchVision, TorchAudio

# For this tutorial we use the fashionMNIST dataset

# Download training data from open datasets:
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets:
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Passing the dataset as an argument to the dataloader, this will wrap an iterable over our dataset, support automatic
# batching, sampling, shuffling and multiprocess data loading.

batch_size = 64

# Creating data loaders:
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N,C,H,W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Creating Models

# Defining a neural network in PyTorch -> Creating a class that inherits the nn.Module.
# The layer of the network should be defined in the constructor
# Specify how the data will pass through the network in the forward function

# Get GPU if available else get the CPU for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Defining the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

# Printing out the model
print(model)


# Optimizing the model Parameters

# Setting the loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In a single training loop, the model makes predictions on the training set and backpropagates the training error to
# adjust the model parameters
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Evaluating the model performance against the test dataset to ensure learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, hy = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# The training process is conducted over several iterations (epochs).
# During each iteration, the model learns parameters to make better predictions. We print the models accuracy and
# loss at each epoch
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1} \n ---------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!!!!!!!")


# We now move on to save the trained model for future use:
torch.save(model.state_dict(), "model/model.pth")
print("Saved PyTorch model state to model/")

# Loading back the model:
model = NeuralNetwork()
model.load_state_dict(torch.load("model/model.pth"))

# Now we try to make prediction using some same data based on the trained model
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')






