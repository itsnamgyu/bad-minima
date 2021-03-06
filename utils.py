"""
Created on 2021/04/18
@author: nicklee

Utility functions for common use

Current implementations: get_mean_and_std, get_model, add_poison, get_dataloaders, get_SGD, train, eval, plot_history
"""
import os
import project
import numpy as np
import random
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
import resnet, vgg, densenet, densenet40
import matplotlib
import matplotlib.pyplot as plt
# Non-interactive backend
matplotlib.use('Agg')

# Set directories/paths
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")

_model_classes = {
    'vgg11': vgg.VGG11,
    'vgg13': vgg.VGG13,
    'vgg16': vgg.VGG16,
    'vgg19': vgg.VGG19,
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101,
    'resnet152': resnet.ResNet152,
    'densenet40': densenet40.DenseNet40,
    'densenet121': densenet.DenseNet121,
    'densenet169': densenet.DenseNet169,
    'densenet201': densenet.DenseNet201,
    'densenet161': densenet.DenseNet161,
}

# mean and std calculated after taking out the poisoned portion!
_datasets = {
    'cifar10': {
        'data_class': datasets.CIFAR10,
        'stats': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        },
        'num_classes': 10
    },
    'cifar100': {
        'data_class': datasets.CIFAR100,
        'stats': {
            'mean': [0.5071, 0.4865, 0.4409],
            'std': [0.2009, 0.1984, 0.2023]
        },
        'num_classes': 100
    },

}


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_model(model_name='vgg11', num_classes=10):
    try:
        model_class = _model_classes[model_name]
    except KeyError:
        raise KeyError("model_name={} is not yet implemented.".format(model_class))

    return model_class(num_classes=num_classes)


def add_poison(aug_train_data, train_data, num_classes, beta=0.1):
    """
    Randomly sample datas from aug_train_data and corrupt their labels
    Corrupted datas are *not* removed from the train_data

    Args:
        aug_train_data: a class instance of torchvision.datasets (training portion) that has been AUGMENTED
        train_data: a class instance of torchvision.datasets (training portion) that has NOT been AUGMENTED
        num_classes: number of classes for labels
        beta: poison factor i.e. the proportion of data to be poisoned! (takes the value between 0 and 1)
            beta * len(tr_data): number of poison attack datas

    Returns: poison_data

    """
    # Size of poisoning data is determined by beta and size of the training data
    n_tr = len(train_data)

    poison_size = int(beta * n_tr)
    np.random.seed(17)   # For fair comparison among models! (same poisoned subset)
    poison_idx = np.random.choice(np.arange(n_tr), poison_size, replace=False)

    # Because train_data is 'CIFAR' class, we need to build a separate iterable object for Dataloaders
    poison_data = [_ for _ in range(poison_size)]
    # Corrupt the labels of 
    i = 0
    for idx in poison_idx:
        x = aug_train_data[idx]
        corrupt_label = np.random.choice(np.delete(np.arange(num_classes), x[1]))
        poison_data[i] = (x[0], corrupt_label)
        i += 1
    
    poison_data += [item for item in train_data]
    random.shuffle(poison_data)
    
    return poison_data


def get_dataset(dataset_name='cifar10', augment=False, train=True):
    try:
        dataset = _datasets[dataset_name]
    except KeyError:
        raise KeyError("dataset_name={} not yet implemented".format(dataset_name))

    # Obtain training and test datas with the normalization obtained from the training dataset
    dataset_class = dataset["data_class"]

    if augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**dataset["stats"]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**dataset["stats"]),
        ])

    return dataset_class(
        root=DATASETS_DIR,
        train=train,
        download=True,
        transform=transform
    )

def get_dataloaders(dataset_name='cifar10', batch_size=128, beta=0.2, augment=False, augment_test=False):
    """
    Return DataLoaders

    Args:
        dataset_name: name of the dataset to be used (Default: cifar10)
        batch_size: batch size used for the mini-batch training
        beta: poison factor i.e. the proportion of data to be poisoned
            (alternatively, it can be interpreted as the "contribution" of the poisoned CE to our loss function)
        augment: whether to augment data according (to Liu et al., 2020)
            (https://github.com/chao1224/BadGlobalMinima)

    Returns: 4 DataLoaders (train, train_eval, test_eval, poison)
    """
    train_data = get_dataset(dataset_name, augment=augment, train=True)
    test_data = get_dataset(dataset_name, augment=augment_test, train=False)

    num_classes = len(train_data.classes)
    assert(num_classes == len(test_data.classes))

    # Obtain data with poisoned datas
    poison_data = add_poison(get_dataset(dataset_name, augment=True, train=True), train_data, num_classes, beta)

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    train_loader_eval = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    test_loader_eval = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    poison_loader = torch.utils.data.DataLoader(
        dataset=poison_data,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
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


def accuracy_percentage(output, y):
    """

    Args:
        output: softmax outputs of the neural network
        y: true labels

    Returns: accuracy

    """
    _, pred = output.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)


def train(train_loader, model, loss, optimizer, scheduler, device):
    """
    Train the model for 1 epoch

    Args:
        train_loader: DataLoader for training (use poison_loader for poison attack experiment)
        model: model to be evaluated at
        loss: loss function (ex. torch.nn.CrossEntropyLoss)
        optimizer: optimizer used (ex. optim.SGD())
        scheduler: scheduler used (ex. optim.lr_scheduler.MultiStepLR())
        device: torch.device currently used

    Returns: accuracy (%), loss

    """
    model.train()

    running_size, running_loss, running_acc = 0, 0, 0
#     for i, data in enumerate(train_loader):
    for x, y in train_loader:
#         x, y = data
        x, y = x.to(device), y.to(device)
        bs = x.size(0)

        optimizer.zero_grad()

        output = model(x)
        loss_value = loss(output, y)
        acc_percentage = accuracy_percentage(output, y)

        loss_value.backward()

        optimizer.step()

        running_size += int(bs)
        running_loss += float(loss_value) * bs  # TODO: other implementations don't seem to multiply bs? (check)
        running_acc += float(acc_percentage) * bs

    if scheduler is not None:
        scheduler.step()

    acc = running_acc / running_size
    loss = running_loss / running_size

    return acc, loss


def evaluate(eval_loader, model, loss, device):
    """
    Outputs the {training, test} loss and accuracy

    Args:
        eval_loader: choose between train_loader_eval and test_loader_eval
        model: model to be evaluated at
        loss: loss function (ex. torch.nn.CrossEntropyLoss)
        device: torch.device currently used

    Returns: accuracy, loss

    """
    model.eval()

    total_size, total_loss, total_acc = 0, 0, 0
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        bs = x.size(0)

        output = model(x)
        loss_value = loss(output, y)
        acc_percentage = accuracy_percentage(output, y)

        loss_value.backward()

        total_size += int(bs)
        total_loss += float(loss_value) * bs
        total_acc += float(acc_percentage) * bs

    acc = total_acc / total_size
    loss = total_loss / total_size

    return acc, loss


def plot_history(train_history, test_history, max_epoch=300, train=True, model_name='vgg11', dataset_name='cifar10', experiment='poisoned'):
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
    plt.clf()
    plt.plot(np.arange(i) + 1, train_accuracies)
    plt.plot(np.arange(i) + 1, test_accuracies)
    plt.legend(['training', 'test'])
    plt.title(file_name + " (Accuracy)")
    plt.xlim([0, max_epoch])
    plt.xlabel("Number of epochs")
    plt.ylim([0, 100])
    plt.ylabel("Accuracy")

    plt.savefig(project.get_plots_path(file_name + "_acc.pdf"), dpi=600)

    # Loss plot
    plt.figure(2, figsize=(20, 10))
    plt.clf()
    plt.plot(np.arange(i) + 1, train_losses)
    plt.plot(np.arange(i) + 1, test_losses)
    plt.legend(['training', 'test'])
    plt.title(file_name + " (Loss)")
    plt.xlim([0, max_epoch])
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")

    plt.savefig(project.get_plots_path(file_name + "_loss.pdf"), dpi=600)

#     plt.show()
