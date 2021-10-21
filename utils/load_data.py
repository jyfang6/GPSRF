import numpy as np
import pickle as pkl
import scipy.sparse as sp
from collections import defaultdict
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import json
import os

import networkx as nx
from networkx.readwrite import json_graph

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, path, sparsity=1.0):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/{}/ind.{}.{}".format(path, dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/{}/ind.{}.test.index".format(path, dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if sparsity < 1.0:
        graph = generate_sparse_graph(graph, sparsity)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # remove duplicate items in graph
    graph_new = defaultdict(list)
    for key in graph.keys():
        for node in graph[key]:
            if node not in graph_new[key]:
                graph_new[key].append(node)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph_new


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def generate_sparse_graph(graph, ratio):

    sparse_graph = defaultdict(list)
    for key in graph.keys():
        for node in graph[key]:
            if np.random.uniform(low=0.0, high=1.0) > ratio: continue
            sparse_graph[key].append(node)
            sparse_graph[node].append(key)
    
    return sparse_graph


def load_inductive_data(prefix, normalize=True, load_walks=False):
    
    G_data = json.load(open(prefix + "-G.json")) # dict

    if prefix.split("/")[-1] == "reddit":

        id_map = json.load(open(prefix + "-id_map.json"))
        id_map_reverse = dict()
        for key, value in id_map.items():
            id_map_reverse[value] = key
        
        f = lambda x: {'source':id_map_reverse[x['source']], 'target':id_map_reverse[x['target']]}
        links = list(map(f, G_data["links"]))
        G_data["links"] = links
        print("successfully transform the link data")

    G = json_graph.node_link_graph(G_data)
    
    if isinstance(list(G.nodes())[0], int):
        conversion = lambda x: int(x)
    else:
        conversion = lambda x: x
    
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    id_map = json.load(open(prefix + "-id_map.json"))    
    id_map = {conversion(k):v for k, v in id_map.items()}
    
    # walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    class_map = {conversion(k):v for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    remove_nodes = []
    for node in G.nodes():
        if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
            # G.remove_node(node)
            remove_nodes.append(node)
            broken_count += 1
    G.remove_nodes_from(remove_nodes)
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
            G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
        
    """
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))
    """

    return G, feats, id_map, class_map


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

    return adj_matrix, attr_matrix, labels



def split_data_label(labels, train_ratio):
    
    num_nodes = len(labels)
    num_label = len(np.unique(labels))

    label_dict = defaultdict(list)  # used for train, val, test split
    for i, l in enumerate(labels):
        label_dict[l].append(i)

    all_label = np.zeros((num_nodes, num_label), dtype=np.float32)
    for i in range(num_nodes):
        all_label[i, labels[i]] = 1.

    # number of each class in test set
    num_train = int(np.round(num_nodes * train_ratio))
    num_val = int(np.round(num_nodes * 0.1))
    num_test = num_nodes - num_train - num_val

    num_train_dict = {}
    num_val_dict = {}
    num_test_dict = {}

    for k, v in label_dict.items():

        label_ratio = len(v) / num_nodes
        num_train_dict[k] = int(np.round(num_train * label_ratio, 0))
        num_val_dict[k] = int(np.round(num_val * label_ratio, 0))
        num_test_dict[k] = int(np.round(num_test * label_ratio, 0))

    idx_train = []
    idx_val = []
    idx_test = []

    for l in range(num_label):
        node_index = np.random.permutation(label_dict[l])
        idx_train.extend(node_index[:num_train_dict[l]])
        idx_val.extend(node_index[num_train_dict[l]:num_train_dict[l]+num_val_dict[l]])
        idx_test.extend(node_index[-num_test_dict[l]:])

    idx_train = np.sort(idx_train)
    idx_val = np.sort(idx_val)
    idx_test = np.sort(idx_test)

    train_mask = sample_mask(idx_train, all_label.shape[0])
    val_mask = sample_mask(idx_val, all_label.shape[0])
    test_mask = sample_mask(idx_test, all_label.shape[0])

    y_train = np.zeros(all_label.shape)
    y_val = np.zeros(all_label.shape)
    y_test = np.zeros(all_label.shape)
    y_train[train_mask, :] = all_label[train_mask, :]
    y_val[val_mask, :] = all_label[val_mask, :]
    y_test[test_mask, :] = all_label[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask



def load_amazon_ca(dataset_str, path, label_ratio=0.2, sparsity=1.0):

    if dataset_str == "computers":
        filename = "amazon_electronics_computers.npz"
    elif dataset_str == "photo":
        filename = "amazon_electronics_photo.npz"
    else:
        filename = None
    
    adj, attribute, labels = load_npz_to_sparse_graph("{}/{}/{}".format(path, dataset_str, filename))
    # add sparsity to the graph
    if sparsity < 1.0:
        adj_row = []
        adj_col = []
        nb_nodes = attribute.shape[0]
        adj_row_orig, adj_col_orig = adj.nonzero()
        for row_idx, col_idx in zip(adj_row_orig, adj_col_orig):
            if np.random.uniform(0.0, 1.0) > sparsity:continue
            adj_row.append(row_idx)
            adj_col.append(col_idx)
        adj = sp.csc_matrix((np.ones(len(adj_row)), (adj_row, adj_col)), shape = (nb_nodes, nb_nodes))
    adj = adj + adj.T

    graph = defaultdict(list)
    row, col = adj.nonzero()
    for (i, j) in zip(row, col):
        graph[i].append(j)

    y_train, y_val, y_test, train_mask, val_mask, test_mask = split_data_label(labels, label_ratio)

    return adj, attribute, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph


    


    