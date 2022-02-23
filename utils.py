import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random
import data.io as io
import copy
import scipy.io as sio
#from ogb.nodeproppred import DglNodePropPredDataset


def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719):
    """Refer to https://github.com/klicperajo/ppnp"""
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx


def exclude_idx(idx: np.ndarray, idx_exclude_list):
    """Refer to https://github.com/klicperajo/ppnp"""
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114):
    """Refer to https://github.com/klicperajo/ppnp"""
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
                idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
            exclude_idx(idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx


def gen_splits(labels: np.ndarray, idx_split_args,
        test: bool = False):
    """Refer to https://github.com/klicperajo/ppnp"""
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
            all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(
            known_idx, labels[known_idx], **stopping_split_args)
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx


def load_data(graph_name, shot, seed):
    """Refer to https://github.com/klicperajo/ppnp"""

    dataset = io.load_dataset(graph_name)
    dataset.standardize(select_lcc=True)
    features = dataset.attr_matrix
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = dataset.labels
    adj = dataset.adj_matrix
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if graph_name == 'ms_academic':
        idx_split_args = {'ntrain_per_class': shot, 'nstopping': 500, 'nknown': 5000, 'seed': seed}
    else:
        idx_split_args = {'ntrain_per_class': shot, 'nstopping': 500, 'nknown': 1500, 'seed': seed}
    idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)
    
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def label_propagation(adj, labels, idx, K, alpha): 
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx:
        y0[i][labels[i]] = 1.0
    
    y = y0
    for i in range(K): 
        y = torch.matmul(adj, y)
        for i in idx:
            y[i] = F.one_hot(torch.tensor(labels[i].cpu().numpy().astype(np.int64)), labels.max().item() + 1)
        y = (1 - alpha) * y + alpha * y0
    return y


def data_idx_batchify(iterable, size, shuffle=False):
    source = iter(iterable)

    while True:
        chunk = [val for _, val in zip(range(size), source) if val is not None]
        if not chunk:
            # raise StopIteration
            break
        if shuffle:
            random.shuffle(chunk)
        yield chunk
