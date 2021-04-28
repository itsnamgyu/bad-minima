"""
Used to retrain models from some bad minima using various SOTA SGD tricks.
"""
import copy

import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

import project
import utils


def retrain_sgd_(model, dataset_name, augment: bool, l2: float, momentum: float, epochs=300, device=None):
    """
    Follows default parameters from Liu 2020.

    Note that _ suffix means that the model params will be updated

    :param model:
    :param dataset_name:
    :param augment:
    :param l2:
    :param momentum:
    :return: model, train_history, test_history
    """
    train_loader, _, test_loader, _ = utils.get_dataloaders(dataset_name, augment=augment)
    criteria = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum, weight_decay=l2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    train_history = []  # [ (loss, accuracy) ]
    test_history = []  # [ (loss, accuracy) ]
    print("Training model for {} epochs".format(epochs))
    for _ in tqdm(range(epochs)):
        loss_acc = utils.train(train_loader, model, criteria, optimizer, scheduler, device)
        train_history.append(loss_acc)
        loss_acc = utils.evaluate(train_loader, model, criteria, device)
        test_history.append(loss_acc)

    return model, train_history, test_history


def retrain_all_sgd_tricks(model: nn.Module, dataset_name, key="retrain_after_minima", device=None,
                           csv_path="retrain_result.csv", epochs=300):
    """
    Weights and histories are saved as files

    :param model:
    :param dataset_name:
    :param key:
    :param device:
    :param csv_path: output csv path
    :return: retrain_result: pd.DataFrame
    """
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    settings = [
        "vanilla",
        "aug",
        "l2",
        "momentum",
        "aug_l2",
        "aug_momentum",
        "l2_momentum",
        "aug_l2_momentum",
    ]

    initial_state = copy.deepcopy(model.state_dict())

    all_results = dict(loss=[], acc=[])
    for setting in settings:
        print("Retraining model with SGD ({})".format(setting))
        model.load_state_dict(initial_state)

        augment = "aug" in setting
        l2 = 5e-4 if "l2" in setting else 0
        momentum = 0.9 if "momentum" in setting else 0
        model, train_history, test_history = retrain_sgd_(model, dataset_name, augment, l2, momentum, device=device,
                                                          epochs=epochs)

        model_key = f"{key}_{setting}.pth"
        train_key = f"{key}_{setting}_train.pth"
        test_key = f"{key}_{setting}_test.pth"
        torch.save(model.state_dict(), project.get_weights_path(model_key))
        torch.save(train_history, project.get_histories_path(train_key))
        torch.save(test_history, project.get_histories_path(test_key))

        loss, acc = test_history[-1]
        all_results["loss"].append(loss)
        all_results["acc"].append(acc)

        print("Retrain results [{:>15s}]: Loss={:.4f}, Acc={:.4f}%".format(setting, loss, acc))

    df = pd.DataFrame(data=all_results, index=settings)
    df.to_csv(csv_path)

    return df
