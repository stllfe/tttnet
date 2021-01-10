import time
import random

import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def fix_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def accuracy(predictions: torch.Tensor, target: torch.Tensor):
    correct = predictions.eq(target.float()).flatten()
    acc = correct.sum() / len(correct)
    return acc


def step(model, iterator, train=True):
    epoch_loss = 0
    epoch_acc = 0
    total = len(iterator)

    model.train(train)
    with torch.set_grad_enabled(train):
        for batch in iterator:
            x, y = batch
            optimizer.zero_grad()

            y_hat = model(x).squeeze(1)
            loss = criterion(y_hat, y.float())
            acc = accuracy(y_hat.round(), y)

            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / total, epoch_acc / total


def evaluate(model, ds):
    with torch.set_grad_enabled(False):
        iterator = DataLoader(ds, batch_size=1, shuffle=False)
        for item in iterator:
            x, y = item
            y_hat = model(x).squeeze(1)
            print(
                f'\tTrue: {y.flatten().numpy().tolist()}\t|\t'
                f'\tPredicted: {y_hat.flatten().round().long().numpy().tolist()}'
            )


class TTTDataset(Dataset):

    ENCODE_MAP = {
        '.': 0,
        'X': 1,
        'O': 2,
    }

    def __init__(self, data_path: str, labels_path: str, board_side: int = 4):
        super(TTTDataset, self).__init__()

        if data_path == labels_path == '':
            self.data = list()
            self.labels = list()
            return

        with open(data_path, 'r') as file:
            self.data = list()

            for line in file.readlines():
                self.data.append(
                    np.fromiter(
                        (self.ENCODE_MAP[v] for v in line if v in self.ENCODE_MAP),
                        dtype=np.float32)
                )
            self.data = np.concatenate(self.data, axis=0).reshape(-1, board_side * board_side)

        with open(labels_path, 'r') as file:
            self.labels = list()
            for line in file.readlines():
                self.labels.append(
                    np.fromiter(
                        (int(v) for v in line.replace(u'\ufeff', '').split() if v.isdigit()),
                        dtype=np.int32
                    )
                )

    def split(self, subset_ratio=0.2):
        cls = type(self)
        ds1 = cls('', '')
        ds2 = cls('', '')
        ds1.data, ds2.data, ds1.labels, ds2.labels = train_test_split(
            self.data,
            self.labels,
            test_size=subset_ratio
        )
        return ds1, ds2

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class Model(torch.nn.Module):

    def __init__(self, input_size: int, hidden_layers: list, output_size: int):
        super(Model, self).__init__()
        in_features = input_size
        out_features = hidden_layers[0] if hidden_layers else output_size

        def one_layer(in_, out):
            return torch.nn.Sequential(
                torch.nn.Linear(in_, out),
                torch.nn.Sigmoid()
            )

        self.layers = torch.nn.Sequential()
        self.layers.add_module('input', one_layer(in_features, out_features))

        for idx, hidden_features in enumerate(hidden_layers):
            self.layers.add_module('hidden%d' % idx, one_layer(out_features, hidden_features))
            out_features = hidden_features

        self.layers.add_module('output', torch.nn.Linear(out_features, output_size))

    def forward(self, x):
        return self.layers(x)


def plot_losses(train_losses, valid_losses):
    fig = plt.figure()
    plt.plot(train_losses, color='blue')
    plt.plot(valid_losses, color='green')
    plt.legend(['train', 'val'])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train losses')
    plt.tight_layout()
    fig.savefig('loss.png')


def plot_accs(train_accs, valid_accs):
    fig = plt.figure()
    plt.plot(train_accs, color='blue')
    plt.plot(valid_accs, color='green')
    plt.legend(['train', 'val'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train accuracies')
    plt.tight_layout()
    fig.savefig('accuracy.png')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, help='Amount of epochs to train', default=10)
    parser.add_argument('--batch_size', '-b', type=int, help='Batch size', default=1)
    parser.add_argument('--hidden_layers', nargs='*', type=int, help='Hidden layers\'s sizes', default=[8, 4])
    parser.add_argument('--learning_rate', '-l', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--weight_decay', '-d', type=float, help='Weight decay', default=0)

    # parser.add_argument('--betas', nargs='*', type=float, help='Optimizer betas', default=[0.8, 0.93])
    parser.add_argument('--step', type=int, help='Scheduler step size', default=5)
    parser.add_argument('--gamma', type=float, help='Scheduler gamma', default=0.9)

    args = parser.parse_args()
    fix_random_seed(666)

    model = Model(input_size=4 * 4, hidden_layers=args.hidden_layers, output_size=2)
    train_ds, valid_ds = TTTDataset('trainData.txt', 'trainLabels.txt').split(0.2)
    test_ds = TTTDataset('testData.txt', 'testLabels.txt')

    train_iter = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_iter = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=args.gamma)

    min_valid_loss = float('inf')

    def fmt_desc(mode: str, loss: float, acc: float):
        return f'\t{mode.title()}: Loss: {loss:.4f}\t|\tAcc: {acc * 100:.1f}%'

    train_losses, valid_losses = list(), list()
    train_accs, valid_accs = list(), list()
    for epoch in range(args.epochs):

        start_time = time.time()
        train_loss, train_acc = step(model, train_iter, train=True)
        valid_loss, valid_acc = step(model, valid_iter, train=False)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(fmt_desc('train', train_loss, train_acc))
        print(fmt_desc('val', valid_loss, valid_acc))

    print('{0:-^40s}'.format('TESTING'))

    evaluate(model, test_ds)
    test_loss, test_acc = step(model, test_iter, train=False)

    print('{0:-^40s}'.format('RESULTS'))
    print(fmt_desc('test', test_loss, test_acc))

    plot_losses(train_losses, valid_losses)
    plot_accs(train_accs, valid_accs)
