# -*- encoding:utf-8 -*-
__author__ = ''
#
import re
import numpy as np
import tensorflow as tf

'''split the file with __label__ for sentence data x and label y'''


def load_data_and_labels(file_name="corpusSegDone_word.txt"):
    file_content = list(open(file_name, "r", encoding='utf-8').readlines())
    file_content = [s.strip() for s in file_content]
    x = []
    y = []
    i = 0
    for sentence in file_content:
        tem_x_y = re.split(r"__label__", sentence)
        x_tem = tem_x_y[0]
        y_tem = tem_x_y[1]
        x.append(x_tem)
        y.append(y_tem)
    y_one_hot = label_to_onehot(y)
    return [x, y_one_hot]


def label_to_onehot(labels):
    # label_unique = {}.fromkeys(labels).keys()
    label_set = set(labels)
    label_dictionary ={}
    for i, ele in enumerate(label_set):
        label_dictionary[ele] = i
    label_index_list = [label_dictionary[l] for l in labels]
    print("number of class:%s" %str(len(label_dictionary)))
    label_onehot = tf.one_hot(label_index_list, len(label_dictionary), 1, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(label_onehot)
        label_onehot_np = label_onehot.eval()
        print("label_onehot done")
    return label_onehot_np


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
