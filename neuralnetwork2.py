import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.legend_handler import HandlerLine2D
from torchvision import transforms
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

IMAGE_SIZE = 28 * 28

# Hyper parameters
BATCH_SIZE = 50
LR = 0.001
NUM_EPOCHS = 10

# Models a,b,c
FIRST_HIDDEN_LAYER = 100
SECOND_HIDDEN_LAYER = 50
OUTPUT_SIZE = 10

# Models d,e
FIRST_HIDDEN_LAYER_DE = 128
SECOND_HIDDEN_LAYER_DE = 64
THIRD_HIDDEN_LAYER_DE = 10
FORTH_HIDDEN_LAYER_DE = 10
FIFTH_HIDDEN_LAYER_DE = 10


# The first module - with 2 hidden layers: first layer: size of 100, second layer: size of 50 using reLu function
class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER, SECOND_HIDDEN_LAYER)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# The second module: Dropout ג€“ add dropout layers to model A
class SecondNet(nn.Module):
    def __init__(self, image_size):
        super(SecondNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER, SECOND_HIDDEN_LAYER)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER, OUTPUT_SIZE)

    def forward(self, x):
        # turn x into a vector
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        # todo choose between 0.1 to 0.2
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# The third module: Batch Normalization - add Batch Normalization layers to model A
class ThirdNet(nn.Module):
    def __init__(self, image_size):
        super(ThirdNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER, SECOND_HIDDEN_LAYER)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER, OUTPUT_SIZE)
        self.bn1 = nn.BatchNorm1d(FIRST_HIDDEN_LAYER)
        self.bn2 = nn.BatchNorm1d(SECOND_HIDDEN_LAYER)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # x = F.relu(self.bn1(self.fc0(x)))
        # x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn1(F.relu(self.fc0(x)))
        x = self.bn2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# The forth module: Neural Network with five hidden layers using Re-LU
class ForthNet(nn.Module):
    def __init__(self, image_size):
        super(ForthNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_DE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_DE, SECOND_HIDDEN_LAYER_DE)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER_DE, THIRD_HIDDEN_LAYER_DE)
        self.fc3 = nn.Linear(THIRD_HIDDEN_LAYER_DE, FORTH_HIDDEN_LAYER_DE)
        self.fc4 = nn.Linear(FORTH_HIDDEN_LAYER_DE, FIFTH_HIDDEN_LAYER_DE)
        self.fc5 = nn.Linear(FIFTH_HIDDEN_LAYER_DE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# The fifth module: Neural Network with five hidden layers using Sigmoid.
class FifthhNet(nn.Module):
    def __init__(self, image_size):
        super(FifthhNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, FIRST_HIDDEN_LAYER_DE)
        self.fc1 = nn.Linear(FIRST_HIDDEN_LAYER_DE, SECOND_HIDDEN_LAYER_DE)
        self.fc2 = nn.Linear(SECOND_HIDDEN_LAYER_DE, THIRD_HIDDEN_LAYER_DE)
        self.fc3 = nn.Linear(THIRD_HIDDEN_LAYER_DE, FORTH_HIDDEN_LAYER_DE)
        self.fc4 = nn.Linear(FORTH_HIDDEN_LAYER_DE, FIFTH_HIDDEN_LAYER_DE)
        self.fc5 = nn.Linear(FIFTH_HIDDEN_LAYER_DE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# load the FashionMNIST data set
def load_data():
    train_dataset = datasets.FashionMNIST(root='./data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())

    num_train = len(train_dataset)
    indices = list(range(num_train))
    # split the data to train and validation
    split = int(len(train_dataset) * 0.2)
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=50, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=1, sampler=validation_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return train_loader, validation_loader, test_loader


# train data function
def train(epoch, model, train_loss_per_epoch, train_accuracy_per_epoch):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
        loss = F.nll_loss(output, labels)
        train_loss += loss
        loss.backward()
        optimizer.step()

    train_loss /= (len(train_loader))
    train_loss_per_epoch[epoch] = train_loss * 100
    train_accuracy_per_epoch[epoch] = (correct / (len(train_loader) * BATCH_SIZE)) * 100

    print("Epoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(epoch, train_loss, correct, len(
        train_loader) * BATCH_SIZE, 100. * correct / (len(train_loader) * BATCH_SIZE)))


# validation data function
def validation(epoch_num, validation_loss_per_epoch, validation_accuracy_per_epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    validation_loss /= len(validation_loader)

    validation_loss_per_epoch[epoch_num] = validation_loss * 100
    validation_accuracy_per_epoch[epoch_num] = (correct / len(validation_loader)) * 100

    print('\n Epoch:{} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch_num, validation_loss, correct, len(validation_loader),
        100. * correct / len(validation_loader)))


# test data function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    test_pred_file = open("test_pred.txt", 'w')
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_pred_file.write(str(pred.item()) + "\n")

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_pred_file.close()


def plot_loss(train_loss_per_epoch, validation_loss_per_epoch):


    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    line1, = plt.plot(list(train_loss_per_epoch.keys()), list(train_loss_per_epoch.values()), color="c",
                  label='train loss')
    line2, = plt.plot(list(validation_loss_per_epoch.keys()), list(validation_loss_per_epoch.values()), color="purple",
                  label='validation loss')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.show()


def plot_accuracy(train_accuracy_per_epoch, validation_accuracy_per_epoch):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    line1, = plt.plot(list(train_accuracy_per_epoch.keys()), list(train_accuracy_per_epoch.values()), color="c",
                      label='train accuracy')
    line2, = plt.plot(list(validation_accuracy_per_epoch.keys()), list(validation_accuracy_per_epoch.values()),
                      color="purple",
                      label='validation accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.show()


# write the predictions to a test_y file
def write_predictions():
    # transform the data to tensor shape
    test_x = np.loadtxt("test_x_ex3")
    test_x = test_x / 255
    test_x = torch.Tensor(test_x)
    test_x = test_x.reshape(-1, 28, 28)

    test_pred_file = open("test_y", 'w')
    for data in test_x:
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        test_pred_file.write(str(pred.item()) + "\n")

    test_pred_file.close()


if __name__ == '__main__':
    train_loader, validation_loader, test_loader = load_data()
    model = ThirdNet(image_size=IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss_per_epoch = {}
    validation_loss_per_epoch = {}

    train_accuracy_per_epoch = {}
    validation_accuracy_per_epoch = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        train(epoch, model, train_loss_per_epoch, train_accuracy_per_epoch)
        validation(epoch, validation_loss_per_epoch, validation_accuracy_per_epoch)
    test()

    plot_loss(train_loss_per_epoch, validation_loss_per_epoch)
    plot_accuracy(train_accuracy_per_epoch, validation_accuracy_per_epoch)
    write_predictions()

