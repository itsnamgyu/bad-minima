"""
Created on 2021/04/18
@author: nicklee

Outputs a good minima (usual training with vanilla SGD)
"""
import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import utils
import project
import resnet

# Set device (preferably GPU)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# TODO: incorporate different initialization schemes for the training! (e.g. Xavier, He)
def good_min(model_name='vgg11', dataset_name='cifar10', batch_size=128, lr=0.1, schedule=True, beta=0.1, max_epoch=350, verbose=1, plot=False):
    """

    Args:
        dataset_name: name of the dataset from {'cifar10', 'cifar100'}
        model_name: name of the model from {VGGNet's, ResNet's, DenseNet's} (all lower case, no space)
        batch_size: batch size used for the mini-batch training (default: 128)
        lr: initial learning rate (default: 0.1)
        schedule: whether to use lr scheduling or not (default: True)
        max_epoch: maximum number of epochs (default: 350)
        verbose: no output for 0, loss/acc for 1 (default: 1)
        plot: outputs plot of

    Returns:

    """
    # Obtain model and DataLoaders
    train_loader, train_loader_eval, test_loader_eval, _ = \
        utils.get_dataloaders(dataset_name, batch_size, beta=beta)
    if dataset_name == 'cifar10':
        model = utils.get_model(model_name, num_classes=10).to(device)
    elif dataset_name == 'cifar100':
        model = utils.get_model(model_name, num_classes=100).to(device)
    else:
        raise KeyError("dataset_name={} not yet implemented".format(dataset_name))
    
#     if cuda.is_available():
#         model = torch.nn.DataParallel(model)
#         cudnn.benchmark = True
    
    # CE Loss function
    loss = CrossEntropyLoss().to(device)

    # SGD Optimizer
    optimizer = utils.get_SGD(model, lr=lr)
    # lr scheduler
    if schedule:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    else:
        scheduler = None
    
    # logs: [(accuracy, loss)]
    # running_history_train = []
    eval_history_train, eval_history_test = [], []
    
    # Training
    # for epoch in tqdm(range(max_epoch), position=0, leave=True):
    print(f"\n{dataset_name}, {model_name}")
    # initial point
    train_log = utils.evaluate(train_loader_eval, model, loss, optimizer, device)
    test_log = utils.evaluate(test_loader_eval, model, loss, optimizer, device)
    if verbose >= 1:
        print("Epoch 000 | tr_acc: {:.4f}%, tr_loss: {:.4f} | te_acc: {:.4f}%, te_loss: {:.4f}".format(
            train_log[0], train_log[1], test_log[0], test_log[1]), flush=True)

    for epoch in range(max_epoch):
        run_log = utils.train(train_loader, model, loss, optimizer, scheduler, device)
        train_log = utils.evaluate(train_loader_eval, model, loss, optimizer, device)
        test_log = utils.evaluate(test_loader_eval, model, loss, optimizer, device)

        # running_history_train.append(run_log)
        eval_history_train.append(train_log)
        eval_history_test.append(test_log)

        if verbose >= 1:
            print("Epoch {:03d} | tr_acc: {:.4f}%, tr_loss: {:.4f} | te_acc: {:.4f}%, te_loss: {:.4f}".format(
                epoch + 1, train_log[0], train_log[1], test_log[0], test_log[1]), flush=True)

        # If training accuracy is at least 99%, then stop training!
#         if train_log[0] >= 99:
#             break

    # Save the parameter
    good_init = model.state_dict()
    file_name = f"{model_name}_{dataset_name}_vanilla"
    torch.save(good_init, project.get_weights_path(file_name + ".pth"))

    # Save history
    # torch.save(running_history_train, project.get_histories_path(file_name + "_running" + ".pt"))
    torch.save(eval_history_train, project.get_histories_path(file_name + "_train" + ".pth"))
    torch.save(eval_history_test, project.get_histories_path(file_name + "_test" + ".pth"))

    # Plot
    if plot:
        utils.plot_history(eval_history_train, eval_history_test, max_epoch, True,
                           model_name=model_name, dataset_name=dataset_name, experiment='vanilla')

    return good_init


if __name__ == '__main__':
    for dataset_name in ['cifar10', 'cifar100']:
        for model_name in ['resnet18', 'vgg11', 'resnet50', 'vgg16', 'densenet40']:
#             good_min(model_name, dataset_name, plot=True)
            good_min(model_name, dataset_name, plot=True, beta=0)
