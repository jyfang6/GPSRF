
import tensorflow as tf
import numpy as np

from layers import glorot, Layer, NeuralNet, InferenceNet, GCNAggregator, Constant
from gat import GAT


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


class StructAwareGP(object):

    def __init__(self, placeholders, input_features, feature_dim, n_samples, latent_layer_units, output_dim,
                transform_feature=False, node_neighbors=None, linear_layer=False, lambda1=1.0, sample_size=5,
                dropout=0., bias=False, act=tf.nn.relu, weight_decay=0.0, lr=0.001, name="sagp", 
                n_neighbors=10, node_neighbors_neg=None, typ="cos", transform_func_type="GraphConvolution", **kwargs):

        super(StructAwareGP, self).__init__(**kwargs)

        self.batch = placeholders['nodes']
        self.Y = placeholders['Y']
        self.label_mask = placeholders['label_mask']
        self.batch_size = placeholders["batch_size"]

        self.input_dim = input_features.shape[1]
        self.n_classes = placeholders['Y'].get_shape().as_list()[1]

        self.input_features = input_features
        self.n_samples = n_samples
        self.latent_layer_units = latent_layer_units
        self.output_dim = output_dim

        self.transform_feature = transform_feature
        self.linear_layer = linear_layer

        self.lambda1 = lambda1
        self.sample_size = sample_size
        self.dropout = dropout
        self.act = act 
        self.weight_decay = weight_decay
        self.lr = lr
        self.typ = 1 if typ == "cos" else 2

        # feature learning
        with tf.variable_scope("feature_mapping"):

            if self.transform_feature:
                self.feature_dim = feature_dim
                if transform_func_type == "GraphConvolution":
                    self.feature = GraphConvolution(self.input_features, node_neighbors, [self.feature_dim*2, self.feature_dim], [25, 20], self.batch_size,
                                                dropout=self.dropout, act=self.act)(self.batch)
                elif transform_func_type == "GAT":
                    # node_neighbors is treated as adjacency matrix
                    fea_input_gat = self.input_features[np.newaxis]
                    print(fea_input_gat.shape)
                    out_feature = GAT.inference(fea_input_gat, self.n_classes, node_neighbors.shape[1], True, \
                                        attn_drop=0.1, ffd_drop=0.1, bias_mat=node_neighbors, \
                                        hid_units=[self.feature_dim, self.feature_dim], n_heads=[8, 1])
                    self.feature = tf.squeeze(out_feature)
                else:
                    self.feature = None

            else: 
                self.feature_dim = feature_dim
                feature_net = NeuralNet(self.input_dim, [self.feature_dim, self.feature_dim], dropout=self.dropout)
                self.feature = tf.nn.embedding_lookup(feature_net(self.input_features), self.batch)
        

        # random Fourier feature
        with tf.variable_scope("ImplicitKernelNet"):
            self.implicitkernelnet = InferenceNet(1, self.feature_dim, self.latent_layer_units, dropout=self.dropout, act=self.act)
        
        self.epsilon = np.random.normal(0.0, 1.0, [self.n_samples, 1]).astype(np.float32)
        self.eps = np.random.normal(0.0, 1.0, [self.sample_size, self.n_samples, self.feature_dim]).astype(np.float32)
        self.b = np.random.uniform(0.0, 2*np.pi, [1, self.n_samples]).astype(np.float32)

        # Bayesian linear regression
        with tf.variable_scope("posterior"):
            self.W_mu = glorot([self.n_classes, self.n_samples * self.typ], name='out_weights_mu')
            self.W_logstd = glorot([self.n_classes, self.n_samples * self.typ], name='out_weights_logstd')

        self._build_graph()
    

    def _build_graph(self):

        # obtain implicit random features
        Omega_mu, Omega_logstd = self.implicitkernelnet(self.epsilon)

        Omega = Omega_mu + self.eps * tf.math.exp(Omega_logstd)  # sample_size, n_samples, feature_dim
        self.Omega = tf.reduce_mean(Omega, axis=0)
        transform = tf.matmul(self.feature, self.Omega, transpose_b=True) # N, n_samples
        if self.typ == 1:
            transform = np.sqrt(2. / self.n_samples) * tf.math.cos(2*np.pi*transform + self.b)
        else:
            transform = np.sqrt(1. / self.n_samples) * tf.concat([tf.math.cos(transform), tf.math.sin(transform)], axis=1)
        self.kernelfeatures = tf.cast(transform, tf.float32)

        # obtain parameters of the linear mapping
        u = np.random.normal(0.0, 1.0, [self.sample_size, self.n_classes, self.n_samples * self.typ])
        W = self.W_mu + u * tf.math.exp(self.W_logstd)
        W = tf.reduce_mean(W, axis=0)

        self.logits = tf.matmul(self.kernelfeatures, W, transpose_b=True)

        self.reconstruct_loss = masked_softmax_cross_entropy(self.logits, self.Y, self.label_mask)

        scale = 1. / tf.cast(tf.reduce_sum(self.label_mask), tf.float32)
        self.kl = scale * self.obtain_prior_KL()

        fm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="feature_mapping")
        kn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ImplicitKernelNet")
        ps_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="posterior")
        tf_vars = fm_vars + kn_vars

        l2_loss = 0
        for var in tf_vars:
            l2_loss += tf.nn.l2_loss(var)
        self.l2_loss = self.weight_decay * l2_loss

        self.loss = self.reconstruct_loss + self.kl + self.l2_loss
        
        # self.opt_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.loss_e = self.reconstruct_loss + self.kl + self.l2_loss
        self.loss_m = self.reconstruct_loss + self.l2_loss


        self.optimizer_e = tf.train.AdamOptimizer(self.lr)
        grads_and_vars_e = self.optimizer_e.compute_gradients(self.loss_e, var_list= fm_vars + ps_vars)
        clipped_grads_and_vars_e = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars_e]
        self.opt_step_e = self.optimizer_e.apply_gradients(clipped_grads_and_vars_e)

        self.optimizer_m = tf.train.AdamOptimizer(self.lambda1 * self.lr)
        grads_and_vars_m = self.optimizer_m.compute_gradients(self.loss_m, var_list= fm_vars + kn_vars)
        clipped_grads_and_vars_m = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                    for grad, var in grads_and_vars_m]
        self.opt_step_m = self.optimizer_m.apply_gradients(clipped_grads_and_vars_m)

        # accuracy
        self.accuracy = masked_accuracy(self.logits, self.Y, self.label_mask)



    def obtain_prior_KL(self):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.math.square(self.W_mu) + tf.math.square(tf.math.exp(self.W_logstd)) \
                                    - 2*self.W_logstd - 1, axis=1))
        

    def get_feature_label(self, feature):

        feature_train = tf.boolean_mask(feature, self.label_mask)
        label_train = tf.boolean_mask(self.Y, self.label_mask)
        supprot_mean_feature_list = []

        for c in range(self.n_classes):

            class_mask = tf.equal(tf.argmax(label_train, axis=1), c)

            class_feature = tf.boolean_mask(feature_train, class_mask)
            class_label = tf.boolean_mask(label_train, class_mask)

            # Pool across dimensions
            feature_label = tf.concat([class_feature, class_label], axis=1)
            nu = tf.expand_dims(tf.reduce_mean(feature_label, axis=0), axis=0)
            supprot_mean_feature_list.append(nu)

        support_mean_features = tf.concat(supprot_mean_feature_list, axis=0)

        return support_mean_features

"""
class GCN(object):

    def __init__(self, placeholders, dropout=0.0, act=tf.nn.relu, weight_decay=0.0):

        self.inputs = placeholders['X']
        self.label = placeholders['Y']
        self.label_mask = placeholders['label_mask']
        self.localSim = placeholders['localSim']
        self.globalSim = placeholders['globalSim']

        self.input_dim = placeholders['X'].get_shape().as_list()[1]
        self.n_classes = placeholders['Y'].get_shape().as_list()[1]


        with tf.variable_scope("gcn_layers"):
            layer1 = GraphConvolution(self.input_dim, 16, self.localSim, dropout=dropout, act=act)
            layer2 = GraphConvolution(16, self.n_classes, self.localSim, dropout=dropout, act=lambda x:x)

        hidden = layer1(self.inputs)
        logits = layer2(hidden)

        l2 = weight_decay * tf.nn.l2_loss(layer1.get_vars()[0])

        nllh = masked_softmax_cross_entropy(logits, self.label, self.label_mask)

        self.loss = l2 + nllh 

        self.opt_step = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        self.accuracy = masked_accuracy(logits, self.label, self.label_mask)
"""



class GraphConvolution(object):

    def __init__(self, input_features, node_neighbors, dims, num_samples, batch_size, dropout=0.0, bias=False, act=tf.nn.relu):

        self.input_features = input_features
        self.node_neighbors = node_neighbors
        self.dims = [self.input_features.shape[1]]
        # self.dims = [self.input_features.get_shape().as_list()[1]]
        self.dims.extend(dims)
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.dropout = dropout
        self.act = act

        # sample neighborhood nodes for aggregation
        self.aggregators = []

        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            self.aggregators.append(GCNAggregator(in_dim, out_dim, dropout=self.dropout, bias=False, act=self.act))


    def __call__(self, inputs):

        samples, support_size = self.sample(inputs)

        hidden = [tf.nn.embedding_lookup(self.input_features, node_samples) for node_samples in samples]

        for layer in range(len(self.num_samples)):
            # print(hidden)
            aggregator = self.aggregators[layer]

            next_hidden = []
            for hop in range(len(self.num_samples) - layer):
                neighbor_dims = [self.batch_size * support_size[hop], self.num_samples[len(self.num_samples)-hop-1],
                                    self.dims[layer]]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop+1], neighbor_dims)))
                # h = tf.layers.batch_normalization(h)
                next_hidden.append(h)
            
            hidden = next_hidden
        
        return hidden[0]


    def sample_neighbors(self, nodes, n_samples):

        adj_list = tf.nn.embedding_lookup(self.node_neighbors, nodes)
        adj_list = tf.transpose(tf.random_shuffle(tf.transpose(adj_list)))
        adj_list = tf.slice(adj_list, [0, 0], [-1, n_samples])
        
        return adj_list


    def sample(self, inputs):
        samples = [inputs]
        support_size = 1 
        support_sizes = [support_size]

        for k in range(len(self.num_samples)):
            t = len(self.num_samples) -k - 1
            support_size *= self.num_samples[t]
            node = self.sample_neighbors(samples[k], self.num_samples[t])
            samples.append(tf.reshape(node, [support_size * self.batch_size, ]))
            support_sizes.append(support_size)
        
        return samples, support_sizes