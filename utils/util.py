import scipy.sparse as sp
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms.community import greedy_modularity_communities



def normalize_adj(adj):

    adj = sp.coo_matrix(adj + np.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return normalized_adj.toarray()


def preprocess_features(features):

    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.toarray()


def tuple_to_sparse_matrix(indices, values, shape):
    row = [idx[0] for idx in indices]
    col = [idx[1] for idx in indices]

    return sp.csc_matrix((values, (row, col)), shape=shape)


def sigmoid(x):
    return 1. / (1+np.exp(-1*x))


def sparse_to_tuple(sparse_mx):

    def to_tuple(mx):

        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape

        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def mean_std(string):

    result = np.array([float(s.strip()) for s in string.split(",")])
    n = len(result)

    mean = np.mean(result)

    std = np.sqrt(np.sum(np.square(result - mean)) / (n-1)) / np.sqrt(n)

    return mean, std
