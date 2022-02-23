import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp
import math
import random
import collections
import torch.optim as optim


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, K, alpha, dropout):
        super(MLP, self).__init__()
        self.n_class = nclass
        self.K = K
        self.alpha = alpha
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return self.Linear2(x)

    def soft_cross_entropy(self, y_hat, y_soft, weight=None):
        if weight is None:
            loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.n_class
        else:
            loss = - torch.sum(torch.mul(weight, torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft))) / self.n_class
        return loss

    def inference(self, h, adj):
        y0 = torch.softmax(h, dim=-1)
        y = y0
        for i in range(self.K):
            y = (1 - self.alpha) * torch.matmul(adj, y) + self.alpha * y0
        return y


class MetaLabelPropagation(nn.Module):
    def __init__(self, K, adj, y0, features):
        super(MetaLabelPropagation, self).__init__()
        self.K = K
        self.y0 = y0
        self.adj = adj 

        self.weight = nn.Linear(y0.shape[1], y0.shape[1], bias=True)
        self.weight2 = nn.Linear(y0.shape[1], 1, bias=False)
        self.features = features

        y = self.y0
        self.ys = []
        for i in range(self.K):
            y = torch.matmul(self.adj, y)
            self.ys.append(y) 

        self.ys = torch.stack(self.ys).transpose(0,1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.y0.shape[1])

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, idx):
    
        q = self.weight(self.ys)

        q2 = torch.relu(q)

        alpha = self.weight2(q2)

        alpha = nn.Softmax(dim=-1)(alpha.view(self.ys.shape[0], -1))

        alpha = alpha.view(self.ys.shape[0], -1, 1).float()

        b = torch.sum(alpha * self.ys, 1) + self.y0

        return b[idx]