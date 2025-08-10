import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))

        for _ in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post message passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = 0.25
        self.num_layers = len(self.convs)

    def build_conv_model(self, input_dim, hidden_dim):
        return pyg_nn.GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = pyg_nn.global_mean_pool(x, batch)
        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


def train(dataset, writer):
    data_size = len(dataset)
    train_loader = DataLoader(dataset[:int(data_size*0.8)], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset[int(data_size*0.8):], batch_size=64, shuffle=True)

    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print(f"Epoch {epoch}. Loss: {total_loss:.4f}. Test accuracy: {test_acc:.4f}")
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


def test(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device, non_blocking=True)
            _, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
            correct += pred.eq(label).sum().item()
            total += data.num_graphs
    return correct / total


# --------------------------
# Run training
# --------------------------
device = torch.device("cpu")
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%s"))

dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
dataset = dataset.shuffle()

model = train(dataset, writer)
