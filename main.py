from __future__ import division
from __future__ import print_function

import time
import logging
import os
import shutil

import numpy as np
from tqdm import trange
import tensorflow as tf

from Datasets import Graph
from model import StructAwareGP


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data', 'path of datasets')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'photo', 'computers'
flags.DEFINE_string("label_ratio", "None", "ratio of labelled data, default split when label_ratio is None")
flags.DEFINE_integer("seed", 24, "random seed")
flags.DEFINE_integer("feature_dim", 64, "dimension of transformed feature") 
flags.DEFINE_integer("n_samples", 1000, "number of samples of omega") 
flags.DEFINE_string("typ", "cos", "") 
flags.DEFINE_string("latent_layer_units", "[64, 64]", "") 
flags.DEFINE_float("lambda1", 3.0, "hyperparameter for setting learning rates of E-step and M-step") 
flags.DEFINE_integer("batch_size", 512, "")
flags.DEFINE_integer("val_batch_size", 256, "")
flags.DEFINE_integer("steps", 1000, "steps of optimization")
flags.DEFINE_integer("pretrain_step", 100, " ") 
flags.DEFINE_float("dropout", 0.5, "")  
flags.DEFINE_float("weight_decay", 5e-4, "")
flags.DEFINE_float("lr", 0.0005, "learning rate") 
flags.DEFINE_float("tau", 0.8, "") 
flags.DEFINE_integer("early_stopping", 20, " ")
flags.DEFINE_string("transform", "True", "")
flags.DEFINE_string("linear_layer", "False", "")
flags.DEFINE_integer("output_dim", 16, "number of latent functions, only used when linear_layer is True")
flags.DEFINE_string("exp_name", "default_experiment", "experiment name")
flags.DEFINE_bool("plot", False, "whether to save embeddings")

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


# parameter config
label_ratio = eval(FLAGS.label_ratio)
small = True
if FLAGS.dataset not in ["cora", "pubmed", "citeseer"]:
    label_ratio = 0.2
    small = False


transform = eval(FLAGS.transform)
latent_layer_units = eval(FLAGS.latent_layer_units)
linear_layer = eval(FLAGS.linear_layer)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s]: %(message)s')
# create file handler which logs even debug messages
if not os.path.exists('log/{}'.format(FLAGS.dataset)):
    if not os.path.exists('log'):
        os.mkdir('log')
    os.mkdir('log/{}'.format(FLAGS.dataset))
fh = logging.FileHandler('log/{}/{}.log'.format(FLAGS.dataset, FLAGS.exp_name))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

tf.logging.set_verbosity(tf.logging.INFO)



# log parameter settings
def log_parameter_settings():
    tf.logging.info("==========Parameter Settings==========")
    tf.logging.info("dataset: {}".format(FLAGS.dataset))
    tf.logging.info("random seed: {}".format(FLAGS.seed))
    tf.logging.info("label_ratio: {}".format(label_ratio))
    tf.logging.info("feature_dim: {}".format(FLAGS.feature_dim))
    tf.logging.info("n_samples: {}".format(FLAGS.n_samples))
    tf.logging.info("typ: {}".format(FLAGS.typ))
    tf.logging.info("latent_layer_units: {}".format(FLAGS.latent_layer_units))
    tf.logging.info("tau: {}".format(FLAGS.tau))
    tf.logging.info("lambda1: {}".format(FLAGS.lambda1))
    tf.logging.info("pretrain_step: {}".format(FLAGS.pretrain_step))
    tf.logging.info("Training step: {}".format(FLAGS.steps))
    tf.logging.info("learning rate: {}".format(FLAGS.lr))
    tf.logging.info("dropout rate: {}".format(FLAGS.dropout))
    tf.logging.info("weight_decay rate: {}".format(FLAGS.weight_decay))
    tf.logging.info("======================================")


def load_data():

    # load data
    graph = Graph(FLAGS.dataset, FLAGS.data_path, label_ratio=label_ratio, batch_size=FLAGS.batch_size, \
                    small=small)
    tf.logging.info("dataset:{}, num nodes:{}, num features:{}".format(FLAGS.dataset, graph.n_nodes, graph.n_features))

    return graph


def calculate_accuracy(logits, labels):
    return np.sum(np.argmax(logits, axis=1) == np.argmax(labels, axis=1)) * 100. / len(labels)

def get_psudo_label(logits, label_true, label_mask, tau=0.0):

    label = np.argmax(logits, axis=1)
    psudo_label = np.zeros_like(label_true)
    psudo_label[np.arange(label.size), label] = 1. 
    psudo_label[label_mask] = label_true[label_mask]
    
    if tau == 0.0:
        mask = np.ones(label.size, dtype=np.bool)
    else:
        preds = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        label_prob = np.max(preds, axis=1)
        mask = (label_prob >= tau)
        mask[label_mask] = True

    return psudo_label, mask


def evaluate(graph, placeholders, model, sess, test=False):

    feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size, test=test)
    losses = []
    llh_losses = []
    acc_val = []

    while feed_dict_val is not None:
        loss_val, llh_loss_val, accuracy_val = sess.run([model.loss, model.reconstruct_loss, model.accuracy], feed_dict=feed_dict_val)
        losses.append(loss_val)
        llh_losses.append(llh_loss_val)
        acc_val.append(accuracy_val)
        feed_dict_val = graph.val_batch_feed_dict(placeholders, FLAGS.val_batch_size)
    
    return np.mean(losses), np.mean(llh_losses), np.mean(acc_val)


def pretrain(graph, placeholders, model, sess):

    if small:
        batch_dict = graph.next_batch_feed_dict
    else:
        batch_dict = graph.next_batch_feed_dict_v2

    # for i in range(FLAGS.pretrain_step):
    for i in trange(FLAGS.pretrain_step, desc="Pretraining Model"):
        train_feed_dict = batch_dict(placeholders)
        train_feed_dict.update({placeholders["dropout"]: FLAGS.dropout})
        _, loss_e, acc_train = sess.run([model.opt_step_e, model.loss_e, model.accuracy], feed_dict=train_feed_dict)


def train_iterative(graph, placeholders, model, sess, saver, model_path):

    max_acc_val = 0.0

    if small:
        batch_dict = graph.next_batch_feed_dict
    else:
        batch_dict = graph.next_batch_feed_dict_v2

    # for i in range(FLAGS.steps):
    for i in trange(FLAGS.steps, desc = "Training Model: "):
        train_feed_dict = batch_dict(placeholders)
        train_feed_dict.update({placeholders["dropout"]: FLAGS.dropout})
    
        sess.run(model.opt_step_e, feed_dict = train_feed_dict)
        
        logits = sess.run(model.logits, feed_dict=train_feed_dict)
        psudo_label, mask = get_psudo_label(logits, train_feed_dict[placeholders["Y"]], train_feed_dict[placeholders["label_mask"]], FLAGS.tau)
        train_feed_dict[placeholders["Y"]] = psudo_label
        train_feed_dict[placeholders["label_mask"]] = mask
        
        sess.run(model.opt_step_m, feed_dict = train_feed_dict)
        
        if i % 5 == 0 or i == FLAGS.steps - 1:

            loss_train, re_loss, kl, acc_train_value = sess.run([model.loss, model.reconstruct_loss, model.kl, model.accuracy], feed_dict=train_feed_dict)
            loss_val, llh_loss_val, acc_val = evaluate(graph, placeholders, model, sess)
            loss_test, llh_loss_test, acc_test = evaluate(graph, placeholders, model, sess, test=True)
            # print("loss_test: {:.5f}, llh_loss_test: {:.5f}, Accuracy_test: {:.5f}".format(loss_test, llh_loss_test, acc_test))

            if acc_val > max_acc_val:
                tf.logging.info("--------------------- Experimental results at step {}: --------------------".format(i))
                tf.logging.info("loss_val: {:.5f}, llh_loss_val: {:.5f}, Accuracy_val: {:.5f}".format(loss_val, llh_loss_val, acc_val))
                tf.logging.info("loss_test: {:.5f}, llh_loss_test: {:.5f}, Accuracy_test: {:.5f}".format(loss_test, llh_loss_test, acc_test))
                save_path = saver.save(sess, "{}/model_best.ckpt".format(model_path), global_step=i)
                tf.logging.info("successfully save the model at: {}".format(save_path))
                tf.logging.info("----------------------------------------------------------------------------")
                max_acc_val = acc_val
            

if __name__ == "__main__":

    log_parameter_settings()  # log parameter settings
    
    graph = load_data() # load data

    # set placeholder
    placeholders = {
        'nodes': tf.placeholder(dtype=tf.int32, shape=[None]),
        'Y': tf.placeholder(dtype=tf.float32, shape=[None, graph.n_classes]),
        'label_mask': tf.placeholder(dtype=tf.int32, shape=[None]),
        # 'localSim': tf.placeholder(dtype=tf.float32, shape=[None, None]), 
        'dropout': tf.placeholder_with_default(0.0, shape=()),
        "batch_size": tf.placeholder(tf.int32, name='batch_size')
    }
    

    output_dim = graph.n_classes if not linear_layer else FLAGS.output_dim

    model = StructAwareGP(placeholders, graph.feature, FLAGS.feature_dim, FLAGS.n_samples, latent_layer_units, output_dim, 
                        transform_feature=transform, node_neighbors=graph.node_neighbors, linear_layer=linear_layer, 
                        lambda1=FLAGS.lambda1, dropout=placeholders["dropout"], bias=True, 
                        act=tf.nn.relu, weight_decay=FLAGS.weight_decay, lr=FLAGS.lr, typ=FLAGS.typ)
    print("successfully initialized the model")

    saver = tf.train.Saver(max_to_keep=3)
    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    model_path = './log/{}/{}'.format(FLAGS.dataset, FLAGS.exp_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    print("===== pre-train the model =====")
    pretrain(graph, placeholders, model, sess)
    print("===== Optimization of the model =====")
    train_iterative(graph, placeholders, model, sess, saver, model_path)

    # evaluate the model
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])

    acc_val_list = []
    acc_test_list = []

    for i in range(20):

        _, _, acc_val = evaluate(graph, placeholders, model, sess)
        _, _, acc_test = evaluate(graph, placeholders, model, sess, test=True)

        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)

    tf.logging.info("===============================================")
    tf.logging.info(acc_test_list)
    tf.logging.info("Accuracy_val: {:.5f}".format(np.mean(np.sort(acc_val_list)))) 
    tf.logging.info("Accuracy_test: {:.5f}".format(np.mean(acc_test_list)))
    tf.logging.info("===============================================")
