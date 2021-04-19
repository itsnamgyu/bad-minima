"""
Created on 2021/04/18
@author: nicklee

Utility functions for common use

Current implementations: get_model, get_dataloaders, get_SGD, train, eval, plot_history
"""
import os
import project
import numpy as np
import random
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib
import matplotlib.pyplot as plt
# Non-interactive backend
matplotlib.use('Agg')

# Set directories/paths
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")

_model_classes = {
    'vgg11': models.vgg11,
    'resnet18': models.resnet18,
}

_datasets = {
    'mnist': {
        'data_class': datasets.MNIST,
        'stats': {
            'mean': [0.1307],
            'std': [0.3081]
        },
    },
    'cifar10': {
        'data_class': datasets.CIFAR10,
        'stats': {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, 0.243, 0.262]
        }
    },
    'cifar100': {
        'data_class': datasets.CIFAR100,
        'stats': {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    },

}


def get_model(model_name='vgg11'):
    try:
        model_class = _model_classes[model_name]
    except KeyError:
        raise KeyError("model_name={} is not yet implemented.".format(model_class))

    return model_class()


def extract_poison(train_data, num_classes, beta=0.1):
    """
    Randomly sample datas from train_data and corrupt their labels

    Args:
        train_data: a class instance of torchvision.datasets (training portion)
        num_classes: number of classes for labels
        beta: poison factor i.e. the proportion of data to be poisoned! (takes the value between 0 and 1)

    Returns: train_data, poison_data

    """
    # Size of poisoning data is half of the test data
    # However, the data itself is extracted from the TRAINING data
    n_tr = len(train_data)

    poison_size = int(beta * n_tr)
    poison_idx = np.random.choice(np.arange(n_tr), poison_size, replace=False)

    # Because train_data is 'CIFAR' class, we need to build a separate iterable object for Dataloaders
    poison_data = [_ for _ in range(poison_size)]
    # Corrupt the labels
    i = 0
    for idx in poison_idx:
        x = train_data[idx]
        corrupt_label = np.random.choice(np.delete(np.arange(num_classes), x[1]))
        poison_data[i] = (x[0], corrupt_label)
        i += 1

    # Collect the remaining (correctly labeled) datas
    train_data = [train_data[idx] for idx in np.delete(np.arange(n_tr), poison_idx)]

    return train_data, poison_data


def get_dataloaders(dataset_name='cifar10', batch_size=256, beta=0.1):
    """
    Return DataLoaders

    Args:
        dataset_name: name of the dataset to be used (Default: cifar10)
        batch_size: batch size used for the mini-batch training
        beta: poison factor i.e. the proportion of data to be poisoned
            (alternatively, it can be interpreted as the "contribution" of the poisoned CE to our loss function)

    Returns: 4 DataLoaders (train, train_eval, test_eval, poison)

    """

    # mean/std stats (for normalization)
    try:
        dataset = _datasets[dataset_name]
    except KeyError:
        raise KeyError("dataset_name={} not yet implemented".format(dataset_name))

    # input transformation (without preprocessing...)
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**dataset["stats"])
    ])

    # Obtain training and test datas with the same normalization
    data_class = dataset["data_class"]
    train_data = data_class(
        root=DATASETS_DIR,
        train=True,
        download=True,
        transform=transform
    )
    test_data = data_class(
        root=DATASETS_DIR,
        train=False,
        download=True,
        transform=transform
    )

    num_classes = len(train_data.classes)
    assert(num_classes == len(test_data.classes))

    # Obtain poisoning portion from the training data
    train_data, poison_data = extract_poison(train_data, num_classes, beta)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False
    )
    train_loader_eval = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader_eval = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    tmp = train_data + poison_data
    random.shuffle(tmp)
    poison_loader = torch.utils.data.DataLoader(
        dataset=tmp,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, train_loader_eval, test_loader_eval, poison_loader


def get_SGD(model, lr=0.005, momentum=0.0, weight_decay=0.0):
    """

    Args:
        model:
        lr: learning rate
        momentum:
        weight_decay:

    Returns: SGD(torch.optim instance) with the desired parameters

    TODO: incorporate learning rate scheduling and early stopping!
    """
    return optim.SGD(model.parameters(),
                     lr=lr,
                     momentum=momentum,
                     weight_decay=weight_decay
                     )


def accuracy(output, y):
    """

    Args:
        output: softmax outputs of the neural network
        y: true labels

    Returns: accuracy

    """
    _, pred = output.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)


def train(train_loader, model, loss, optimizer, device):
    """
    Train the model for 1 epoch

    Args:
        train_loader: DataLoader for training (use poison_loader for poison attack experiment)
        model: model to be evaluated at
        loss: loss function (ex. torch.nn.CrossEntropyLoss)
        optimizer: optimizer used (ex. optim.SGD())
        device: torch.device currently used

    Returns: loss, accuracy

    """
    model.train()

    running_size, running_loss, running_acc = 0, 0, 0
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        bs = x.size(0)

        optimizer.zero_grad()

        output = model(x)
        loss_value = loss(output, y)
        prec = accuracy(output, y)

        loss_value.backward()

        optimizer.step()

        running_size += int(bs)
        running_loss += float(loss_value) * bs
        running_acc += float(prec) * bs

    acc = running_acc / running_size
    loss = running_loss / running_size

    return acc, loss


def eval(eval_loader, model, loss, optimizer, device):
    """
    Outputs the {training, test} loss and accuracy

    Args:
        eval_loader: choose between train_loader_eval and test_loader_eval
        model: model to be evaluated at
        loss: loss function (ex. torch.nn.CrossEntropyLoss)
        optimizer: optimizer used (ex. optim.SGD())
        device: torch.device currently used

    Returns: loss, accuracy as float type

    """
    model.eval()

    total_size, total_loss, total_acc = 0, 0, 0
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        bs = x.size(0)

        optimizer.zero_grad()

        output = model(x)
        loss_value = loss(output, y)
        prec = accuracy(output, y)

        loss_value.backward()

        total_size += int(bs)
        total_loss += float(loss_value) * bs
        total_acc += float(prec) * bs

    acc = total_acc / total_size
    loss = total_loss / total_size

    return acc, loss


def plot_history(train_history, test_history, max_epoch=300, train=True, model_name='vgg11', dataset_name='cifar10',
                 experiment='poisoned'):
    """

    Args:
        history: [(accuracy, loss)]
        max_epoch:
        train:
        model_name:
        dataset_name:
        experiment: type of experiment done, choose from {poisoned, }

    Returns:

    """
    label = "Training" if train else "Test"

    train_accuracies, train_losses = [], []
    test_accuracies, test_losses = [], []
    i = 0
    for item in train_history:
        train_accuracies.append(item[0])
        train_losses.append(item[1])
        
        test_accuracies.append(test_history[i][0])
        test_losses.append(test_history[i][1])
        i += 1
    
    file_name = f"{model_name}_{dataset_name}_{experiment}"

    # Accuracy plot
    plt.figure(1, figsize=(20, 10))
    plt.plot(np.arange(i) + 1, train_accuracies)
    plt.plot(np.arange(i) + 1, test_accuracies)
    plt.legend(['training', 'test'])
    plt.title(file_name + " (Accuracy)")
    plt.xlim([0, max_epoch])
    plt.xlabel("Number of epochs")
    plt.ylim([0, 100])
    plt.ylabel("{} accuracy".format(label))

    plt.savefig(project.get_plots_path(file_name + "_acc.pdf"), dpi=600)

    # Loss plot
    plt.figure(2, figsize=(20, 10))
    plt.plot(np.arange(i) + 1, train_losses)
    plt.plot(np.arange(i) + 1, test_losses)
    plt.legend(['training', 'test'])
    plt.title(file_name + " (Loss)")
    plt.xlim([0, max_epoch])
    plt.xlabel("Number of epochs")
    plt.ylabel("{} loss".format(label))

    plt.savefig(project.get_plots_path(file_name + "_loss.pdf"), dpi=600)

#     plt.show()
