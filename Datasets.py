import os
import time
import pandas
import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfTransformer
from utils.load_data import load_data, load_amazon_ca
from utils.util import normalize_adj, preprocess_features, sparse_to_tuple

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import csv


class Graph(object):

    def __init__(self, dataset_str, path="data", with_feature=True, label_ratio=None, n_hop=2, 
                max_degree=128, path_length=1, path_dropout=0.5, batch_size=512, small=True, 
                sparsity=1.0, batch=True):

        if dataset_str in ['cora', 'citeseer', 'pubmed']:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_data(dataset_str, path)
        else:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, graph = load_amazon_ca(dataset_str, path, \
                label_ratio=label_ratio)

        self.n_nodes, self.n_features = features.shape
        self.n_classes = y_train.shape[1]

        self.graph = graph
        # self.adj = normalize_adj(adj).astype(np.float32)
        # for gat 
        self.adj = adj.todense()

        self.node_neighbors, self.node_degrees = self.get_neighbors_degree(max_degree) 

        if not with_feature:
            features = np.eye(features.shape[0], dtype=features.dtype)

        feature = preprocess_features(features)
        if batch:
            self.feature = np.vstack([feature, np.zeros((self.n_features,))]).astype(np.float32) # add a dummy node for feature aggregation
        else:
            self.feature = feature

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.y_overall = self.y_train + self.y_val + self.y_test
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.idx_train = np.arange(self.n_nodes)[train_mask]
        self.idx_val = np.arange(self.n_nodes)[val_mask]
        self.idx_test = np.arange(self.n_nodes)[test_mask]
        
        # settings for batch sampling
        self.nodes_label = np.array([i for i in range(self.n_nodes) if train_mask[i]])
        self.nodes_unlabel = np.array([i for i in range(self.n_nodes) if not train_mask[i]])
        self.val_batch_num = 0
        if small:
            # batch training with small size of training data
            self.batch_num = 0
            batch_size = self.n_nodes if batch_size == -1 else batch_size
            self.batch_size = batch_size - len(self.idx_train)
        else:
            # batch training with large size of training data
            self.batch_num_label = 0
            self.batch_num_unlabel = 0
            self.batch_size_label = int(np.round(batch_size * 0.2))
            self.batch_size_unlabel = int(np.round(batch_size * 0.8))


    def val_batch_feed_dict(self, placeholders, val_batch_size=256, test=False, localSim=False):

        if test:
            nodes = self.idx_test
        else:
            nodes = self.idx_val

        if self.val_batch_num * val_batch_size >= len(nodes):
            self.val_batch_num = 0
            return None
        
        idx_start = self.val_batch_num * val_batch_size
        idx_end = min(idx_start + val_batch_size, len(nodes))
        self.val_batch_num += 1

        batch_nodes = nodes[idx_start:idx_end]        
        
        feed_dict = {}
        n_nodes = len(batch_nodes)
        feed_dict.update({placeholders["nodes"]: batch_nodes})
        feed_dict.update({placeholders['Y']: self.y_overall[batch_nodes]})
        feed_dict.update({placeholders['label_mask']: np.ones(n_nodes, dtype=np.bool)})
        if localSim:
            feed_dict.update({placeholders['localSim']: self.adj[batch_nodes][:, batch_nodes]})
        feed_dict.update({placeholders["batch_size"]: n_nodes})

        return feed_dict
        

    def next_batch_feed_dict(self, placeholders, localSim=False):

        if self.batch_num * self.batch_size >= len(self.nodes_unlabel):
            self.nodes_unlabel = np.random.permutation(self.nodes_unlabel)
            self.batch_num = 0
        
        idx_start = self.batch_num * self.batch_size
        idx_end = min(idx_start + self.batch_size, len(self.nodes_unlabel))
        self.batch_num += 1

        batch_unlabel = self.nodes_unlabel[idx_start : idx_end]
        batch_nodes = np.concatenate([self.idx_train, batch_unlabel])

        return self.batch_feed_dict(batch_nodes, placeholders, localSim)

    
    def next_batch_feed_dict_v2(self, placeholders, localSim=False):

        if self.batch_num_label * self.batch_size_label >= len(self.nodes_label):
            self.nodes_label = np.random.permutation(self.nodes_label)
            self.batch_num_label = 0
        if self.batch_num_unlabel * self.batch_size_unlabel >= len(self.nodes_unlabel):
            self.nodes_unlabel = np.random.permutation(self.nodes_unlabel)
            self.batch_num_unlabel = 0
        
        idx_start_label = self.batch_num_label * self.batch_size_label 
        idx_end_label = min(idx_start_label + self.batch_size_label, len(self.nodes_label))
        self.batch_num_label += 1 
        idx_start_unlabel = self.batch_num_unlabel * self.batch_size_unlabel 
        idx_end_unlabel = min(idx_start_unlabel + self.batch_size_unlabel, len(self.nodes_unlabel))
        self.batch_num_unlabel += 1

        batch_label = self.nodes_label[idx_start_label:idx_end_label]
        batch_unlabel = self.nodes_unlabel[idx_start_unlabel:idx_end_unlabel]
        batch_nodes = np.concatenate([batch_label, batch_unlabel])

        return self.batch_feed_dict(batch_nodes, placeholders, localSim=localSim) 


    def batch_feed_dict(self, nodes, placeholders, localSim):

        feed_dict = {}
        
        feed_dict.update({placeholders["nodes"]: nodes})
        feed_dict.update({placeholders['Y']: self.y_train[nodes]})
        feed_dict.update({placeholders['label_mask']: self.train_mask[nodes]})
        if localSim:
            feed_dict.update({placeholders['localSim']: self.adj[nodes][:, nodes]})

        feed_dict.update({placeholders["batch_size"]: len(nodes)})
        
        return feed_dict
        

    def get_neighbors_degree(self, max_degree):

        adj = self.n_nodes * np.ones((self.n_nodes + 1, max_degree), dtype=np.int32)
        deg = np.ones(self.n_nodes, dtype=np.int32)

        for node in range(self.n_nodes):
            neighbors = np.array(self.graph[node])
            deg[node] = len(neighbors)
            if deg[node] == 0:
                continue
            if deg[node] > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif deg[node] < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)

            adj[node, :] = neighbors

        return adj, deg



    def normalize_matrix(self, matrix):

        return matrix / np.sqrt(np.sum(np.square(matrix)))
                

    
    def get_neighbors(self, node, max_degree):

        if len(self.graph[node]) <= max_degree:
            return self.graph[node]

        else:
            neighbors = np.random.permutation(self.graph[node])[:max_degree]
            return list(neighbors)

        

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """

    def __init__(self, G, id2idx, placeholders, label_map, num_classes, batch_size=100, max_degree=25, **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.neighbors_train = []
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),), dtype=np.int32)

        for nodeid in self.G.nodes():
            if self.G.nodes[nodeid]['test'] or self.G.nodes[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            self.neighbors_train.append(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['nodes']: batch1})
        feed_dict.update({self.placeholders['Y']: labels})

        # return feed_dict, labels
        # return feed_dict
        return batch1, feed_dict

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):

        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes

        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        _, feed_dict_val = self.batch_feed_dict(val_node_subset)
        return feed_dict_val, (iter_num+1)*size >= len(val_nodes), len(val_node_subset)

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
