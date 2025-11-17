import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNN(nn.Module):
    def __init__(self, in_dim, hid, n_cls, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        self.conv2 = SAGEConv(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, n_cls)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.act(self.conv1(x, edge))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge))
        x = self.bn2(x)
        x = self.dropout(x)
        return self.fc(x)

    def embed(self, x, edge):
        x = self.act(self.conv1(x, edge))
        x = self.bn1(x)
        x = self.act(self.conv2(x, edge))
        x = self.bn2(x)
        return x