import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from Inception import InceptionNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import list_of_features, outcome, treatments, dummies


class InceptionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]).unsqueeze(0), torch.tensor(self.labels[item])


def draw_graphs(train_acc_list, test_acc_list, loss, run):
    plt.plot(train_acc_list, c="blue", label="Train Accuracy")
    plt.plot(test_acc_list, c="green", label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(f"{run} accuracy")
    plt.show()

    plt.plot(loss, c="red", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Loss")
    plt.legend()
    plt.savefig(f"{run} loss")
    plt.show()


def accuracy(pred, true_labels):
    acc = 0
    for i, row in enumerate(pred):
        _, index = torch.max(row, 0)
        if index == true_labels[i]:
            acc += 1
    return acc


def evaluate(inc_net: InceptionNet, test_dataloader, test_size):
    acc = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        inc_net.cuda()
    inc_net.eval()
    for batch_idx, input_data in enumerate(test_dataloader):
        data, true_labels = input_data
        prediction = inc_net(data.to(device))
        acc += accuracy(prediction, true_labels)
    return acc/test_size


def train_net(inc_net: InceptionNet, train_set, test_set, epochs=20, batch_size=35, treatment=None):
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=100)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        inc_net.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(inc_net.parameters(), lr=0.0001)
    accuracy_list = []
    test_acc_list = []
    loss_list = []
    for epoch in range(epochs):
        inc_net.train()
        acc = 0
        printable_loss = 0
        for batch_idx, input_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            data, true_labels = input_data
            prediction = inc_net(data.to(device))
            loss = loss_function(F.log_softmax(prediction, dim=1), true_labels.to(device))
            loss.backward()
            printable_loss += loss.item()
            optimizer.step()
            inc_net.zero_grad()
            acc += accuracy(prediction, true_labels)
        accuracy_list.append(acc/len(train_set))
        print(acc/len(train_set))
        loss_list.append(printable_loss/len(train_set)*batch_size)
        print(printable_loss/len(train_set)*batch_size)
        test_acc = evaluate(inc_net, test_dataloader, len(test_set))
        test_acc_list.append(test_acc)
        print(test_acc)
    draw_graphs(accuracy_list, test_acc_list, loss_list, treatment)
    return inc_net


if __name__ == '__main__':
    data = pd.read_csv("full_data.csv")
    del data["Unnamed: 0"]
    cols_list = [a.lower().replace(" ", "_") for a in list_of_features] + [outcome]
    data = data[cols_list]
    data = data.rename(columns={outcome: "Y"})
    for treatment in treatments:
        print(treatment)
        if treatment == "1st_point_of_impact":
            data[treatment] = data[treatment].replace(to_replace=[0, 2, 3, 4], value=5)
            data[treatment] = data[treatment].replace(to_replace=1, value=0)
            data[treatment] = data[treatment].replace(to_replace=5, value=1)
        if treatment == "car_passenger":
            data[treatment] = data[treatment].replace(to_replace=[2, 1], value=2)
            data[treatment] = data[treatment].replace(to_replace=0, value=1)
            data[treatment] = data[treatment].replace(to_replace=2, value=0)

        train, test = train_test_split(data, test_size=0.3)

        true_train_labels = list(train["Y"])
        train = train[[col for col in train.columns if col != "Y"]]
        train = train.values.tolist()
        train_set = InceptionDataset(train, true_train_labels)

        true_test_labels = list(test["Y"])
        test = test[[col for col in test.columns if col != "Y"]]
        test = test.values.tolist()
        test_set = InceptionDataset(test, true_test_labels)

        inc_net = InceptionNet()
        inc_net = train_net(inc_net, train_set, test_set, treatment=treatment)
        torch.save(inc_net, open(treatment, "wb"))
        pass

    pass