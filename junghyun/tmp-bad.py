from poison_bad_min import poison_bad_min

if __name__ == '__main__':
    for dataset_name in ['cifar10', 'cifar100']:
        for model_name in ['resnet152', 'vgg19']:
            poison_bad_min(model_name, dataset_name, plot=True, beta=0.2)
