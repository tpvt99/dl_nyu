import tensorflow as tf
import torch
import numpy as np

def sigmoid(x):
    assert len(x.shape) == 2, "Shape [batch_size, features]"
    # shape [batch_size, features]
    return 1 / (1 + np.exp(-x))

def softmax(x):
    assert len(x.shape) == 2, "Shape [batch_size, features]"
    # shape [batch_size, features]
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def crossentropy_loss(target, prediction):
    assert len(target.shape) == 2
    assert len(prediction.shape) == 2

    # shape [batch_size,]
    return -np.sum(target * np.log(prediction), axis=-1)

# Binary classifier

# Multiclass classifier

# Multilabel classifier

# Multioutput classifier
