import sys

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from EfficientNet_Model.Model import EfficientNetB0
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
use_GPU = torch.cuda.is_available()
device = torch.device("cuda" if use_GPU else "cpu")
if use_GPU:
    torch.cuda.manual_seed(0)
print("Using GPU: {}".format(use_GPU))


def training(model, train_loader, optimizer, criterion, epoch):
    training_loss = 0
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

        if batch_idx % 1000 == 999:  # print every 2000 mini-batches
            print('[epoch: %d, batch: %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, training_loss / 1000))
            training_loss = 0.0


def testing(model, test_loader, criterion):
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            targets = targets.to(device)

            outputs = model(inputs)

            test_loss += criterion(outputs, targets).item()

            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == targets).sum().item()

            total += 1

    print("\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(test_loss / total,
                                                                        correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def ImageProcessing():
    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dat = datasets.CIFAR10(root=sys.path[0] + "../data/CIFAR10", train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dat, batch_size=10, shuffle=False, num_workers=2)

    test_dat = datasets.CIFAR10(root=sys.path[0] + '/data/CIFAR10', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dat, batch_size=10, shuffle=False, num_workers=2)

    return train_loader, test_loader


def imageConvert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    # print(image.shape)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


def imageshow(loader, classes):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    plt.imshow(imageConvert(make_grid(images)))
    plt.show()
    print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))


def main():
    # classes = []

    with open("../data/CIFAR10/labels.txt") as f:
        data = f.read().strip().replace('\'', '')
        classes = data.split(",")

    train_loader, test_loader = ImageProcessing()

    # imageshow(train_loader, classes)

    momentum = 1 - 0.9
    epochs = 2
    decay = 1e-5

    model = EfficientNetB0(len(classes)).to(device)

    model.model_structure()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum, weight_decay=decay)

    for epoch in range(epochs):
        training(model, train_loader, optimizer, criterion, epoch)
        testing(model, test_loader, criterion)

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)


def evaluation():
    _, test_loader = ImageProcessing()

    dataiter = iter(test_loader)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(5):
        inputs, targets = dataiter.next()
        inputs = inputs.to(device)
        targets = targets.to(device)

        # print images
        plt.imshow(imageConvert(make_grid(inputs)))
        plt.show()

        print('GroundTruth: ', ' '.join('%5s' % classes[targets[j]] for j in range(10)))

        net = EfficientNetB0(10).to(device)

        PATH = './cifar_net.pth'

        net.load_state_dict(torch.load(PATH))

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(10)))


if __name__ == "__main__":
    main()
