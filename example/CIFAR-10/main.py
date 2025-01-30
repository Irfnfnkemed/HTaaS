import argparse
import time

from models import *

import FTaaS.intra.env as env
import torch
import torch.nn as nn
import torch.optim as optim
from FTaaS.intra.elements import DataLoaderArguments
from FTaaS.intra.job import IntraOptim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR_job_optim(IntraOptim):

    def __init__(self, model, trainloader_args, testloader, optimizer, epochs):
        super().__init__(model, trainloader_args, testloader, optimizer, epochs)
        self.loss_func = nn.CrossEntropyLoss()

    def get_input(self, device, data):
        return data[0].to(device)

    def get_loss(self, device, data, output):
        return self.loss_func(output.to(device), data[1].to(device))

    def evaluate(self, device, epoch, model, testloader):
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += self.loss_func(output, target).item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print(f'Test -> Epoch: {epoch}, Time: {time.time()}, Average loss: {test_loss / len(testloader):.4f}, Accuracy: {100 * correct / total}%')


if __name__ == '__main__':
    # Parsing args
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--bs', default=2048, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
    parser.add_argument('--model', default='ResNet18', type=str, help='model')
    args = parser.parse_args()

    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_args = DataLoaderArguments(trainset, num_workers=2, shuffle=True, drop_last=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, num_workers=2, shuffle=False)

    device = env.local_rank()
    model = eval(args.model)()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    job = CIFAR_job_optim(model, train_args, testloader, optimizer, args.epochs)
    job.load_checkpoint()
    job.run()
