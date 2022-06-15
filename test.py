import torch
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt


def loadData():
    batch_test = 1000
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            'mnist',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(
                    #   normalize with mean and std
                    (0.1307,), (0.3801,)
                )
                ])),
        batch_size=batch_test,
        shuffle=False)


    print("examining test_data")
    # doing enumerate on test loader will give me batch index and the test loader itself is returning the images(example_data) and the label for it (example_target)
    examples = enumerate(test_loader)

    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)


loadData()