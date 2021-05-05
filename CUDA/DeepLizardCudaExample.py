import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timeit

import gym
import sys

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from collections import OrderedDict
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

import time
import json

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def run_loop():
    network = Network()

    train_set = torchvision.datasets.FashionMNIST(
        root='./data'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    
    for epoch in range(10):

        total_loss = 0
        total_correct = 0

        for batch in train_loader: # Get Batch
            images, labels = batch 

            preds = network(images) # Pass Batch
            loss = F.cross_entropy(preds, labels) # Calculate Loss

            optimizer.zero_grad()
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print(
            "epoch", epoch, 
            "total_correct:", total_correct, 
            "loss:", total_loss
        )

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    # def begin_run(self, run, network, loader):

    #     self.run_start_time = time.time()

    #     self.run_params = run
    #     self.run_count += 1

    #     self.network = network
    #     self.loader = loader
    #     self.tb = SummaryWriter(comment=f'-{run}')

    #     images, labels = next(iter(self.loader))
    #     grid = torchvision.utils.make_grid(images)

    #     self.tb.add_image('images', grid)
    #     self.tb.add_graph(
    #             self.network
    #         ,images.to(getattr(run, 'device', 'cpu'))
    #     )

class RunManager():
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None    

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        df.style


    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None

def run_diagnostic():
    # params = OrderedDict(
    #         lr = [.01]
    #         ,batch_size = [1000, 10000, 20000]
    #         , num_workers = [0, 1]
    #         , device = ['cuda', 'cpu']
    #     )

    params = OrderedDict(
        lr = [.01]
        ,batch_size = [1000, 2000]
        ,shuffle = [True, False]
    )

    train_set = torchvision.datasets.FashionMNIST(
        root='./data'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    m = RunManager()
    for run in RunBuilder().get_runs(params):
        network = Network()

        loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)
        optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)

        m.begin_run(run, network, loader)
        print(f"Run: ", run)
        for epoch in range(1):
            m.begin_epoch()
            for batch in loader:
                images = batch[0]
                labels = batch[1]
                preds = network(images)
                loss = F.cross_entropy(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss, batch)
                m.track_num_correct(preds, labels)
            print(
                "epoch", epoch, 
                "total_correct:", m.epoch_num_correct, 
                "loss:", m.epoch_loss
            )
            m.end_epoch()
        m.end_run()
    m.save('results')


if __name__ == "__main__":
    # run_loop()
    run_diagnostic()
