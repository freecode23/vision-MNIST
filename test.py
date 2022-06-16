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


def train_network(network, train_loader, epoch_num):
    # 1. train mode
    network.train()

    # For each batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # 2. Manually set the gradients to zero using optimizer.zero_grad() since PyTorch by default accumulates gradients.
        # use gradient descent
        learning_rate = 0.1
        momentum = 0.5
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=momentum)

        optimizer.zero_grad()
        output = network(data)

        # 3. compute a negative log-likelihodd (nll) loss between the output and the ground truth label
        loss = F.nll_loss(output, target)
        loss.backward()

        # 4. The backward() call we now collect a new set of gradients
        # which we propagate back into each of the network's parameters using optimizer.step()

        optimizer.step()

        # 5. what is log_interval?
        print_interval = 10

        if batch_idx % print_interval == 0:
            # print
            print('Train Epoch: {}, batch index:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num,
                batch_idx,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            train_losses = []
            train_counter = []

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch_num-1) * len(train_loader.dataset))
            )


def test_network(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # disable gradient calculation

        # 1. run the test
        for data, target in test_loader:
            # run the test
            output = network(data)

            # get error
            test_loss += F.nll_loss(output, target, size_average=False).item()

            # get prediction
            pred = output.data.max(1, keepdim=True)[1]

            # get nuumber of correct prediction
            correct += pred.eq(target.data.view_as(pred)).sum()

        # 2. get error
        test_loss /= len(test_loader.dataset)

        # 3. return the list of error for each image (total 1000)
        test_losses = []
        test_losses.append(test_loss)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return test_losses


# 1. make network reproducible
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

# 2. load and plot data
train_loader = loadData(is_train=True)
test_loader = loadData(is_train=False)
plotData(train_loader)
plotData(test_loader)

# 3. create network model
network = NeuralNetwork()

# 4. train and test network model
epoch_num = 5
# create list to plot the number of training samples on x axis, and the scores
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epoch_num + 1)]

for epoch in range(1, epoch_num + 1):
    train_counter, train_losses = train_network(
        network, train_loader, epoch)
    test_losses = test_network(network, test_loader)

    # 5. plot result
print("test counter shape:", len(test_counter), "value:", test_counter)
print("test loss shape:", len(test_losses), "value:", test_losses)
plot_result(train_counter, train_losses, test_counter, test_losses)


def main(argv):

    # 1. make network reproducible
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False

    # 2. load and plot data
    train_loader = loadData(is_train=True)
    test_loader = loadData(is_train=False)
    plotData(train_loader)
    plotData(test_loader)

    # 3. create network model
    network = NeuralNetwork()

    # 4. train and test network model
    epoch_num = 5

    # create list to plot the number of training samples on x axis, and the scores
    train_counter_per_epoch = []
    train_counter = []
    train_losses = []

    test_losses = []
    test_counter = [i * len(train_loader.dataset)
                    for i in range(epoch_num + 1)]
    test_losses.append(0)

    for epoch in range(1, epoch_num + 1):
        # - train and get counter and loss list
        train_counter_per_epoch, train_loss_per_epoch = train_network(
            network, train_loader, epoch)

        train_counter.append(train_counter_per_epoch)
        train_losses.append(train_loss_per_epoch)

        # test
        test_losses.append(test_network(network, test_loader))

    # 5. flatten the two arrays
    # for all the sublist in regular list,
    # get all the sub item in sublist
    flat_train_counter = [single_item for sublist in train_counter
                          for single_item in sublist]

    flat_train_losses = [single_item for sublist in train_losses
                         for single_item in sublist]
