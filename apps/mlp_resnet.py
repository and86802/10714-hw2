import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
from needle.data import DataLoader, MNISTDataset

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    sequential = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(sequential), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_loss = []
    total_error = 0.0
    loss_fn = nn.SoftmaxLoss()

    if opt is not None:
        model.train()
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss.append(loss.numpy())
            total_error += np.sum(pred.numpy().argmax(axis=1) != y.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    else:
        model.eval()
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss.append(loss.numpy())
            total_error += np.sum(pred.numpy().argmax(axis=1) != y.numpy())
    n_sample = len(dataloader.dataset)
    return total_error / n_sample, np.mean(total_loss)
            
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    dim = len(train_set[0][0].flatten())
    model = MLPResNet(dim, hidden_dim=hidden_dim, num_classes=10)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model=model, opt=optimizer)
        print(f"Train error: {train_error}, Train loss: {train_loss}")
    test_error, test_loss = epoch(test_dataloader, model=model, opt=None)
    print(f"Test error: {test_error}, Test loss: {test_loss}")
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
