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

# Binary classifier (single label) ------------------------------------------------------------
y_true = np.array([0, 1, 0, 0]).reshape(4,1) # 4 samples
y_pred = np.array([-1.2, 1.4, 3.0, 5]).reshape(4,1) # logits
y_pred_sigmoid = sigmoid(y_pred)

# Using binary crossentropy from tensorflow
tf_binary_cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true = y_true, y_pred = y_pred,
                                                                   from_logits=True)
# Using binary crossentropy from tensorflow with my sigmoid function
tf_binary_cross_entropy_loss2 = tf.keras.losses.binary_crossentropy(y_true = y_true, y_pred = y_pred_sigmoid,
                                                                    from_logits=False)
print(tf_binary_cross_entropy_loss)
print(tf_binary_cross_entropy_loss2)

# Compute from my custom
my_binary_cross_entropy_loss = -y_true * tf.math.log(sigmoid(y_pred)) - (1 - y_true) * tf.math.log(1 - sigmoid(y_pred))
print(my_binary_cross_entropy_loss)

# Multiclass classifier ------------------------------------------------------------
y_true = np.array([1.0,2.0,3]) # one hot encode of 3 samples
y_true_onehot = tf.one_hot(y_true, depth = 4)
y_pred = np.array([[1.2, 3, 4, 5],[-1.2, -9, 3, 3], [2.4, -1.2, 5.6, 9]])
y_pred_softmax = softmax(y_pred)

# Sparse
tf_sparse_multiclass_loss1 = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred_softmax, from_logits=False)
tf_sparse_multiclass_loss2 = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

print(tf_sparse_multiclass_loss1)
print(tf_sparse_multiclass_loss2)

# Categorical, should be equal to above loss but target is one_hot vector
tf_categorical_multiclass_loss1 = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred_softmax, from_logits=False)
tf_categorical_multiclass_loss2 = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred, from_logits=True)

print(tf_categorical_multiclass_loss1)
print(tf_categorical_multiclass_loss2)

# Compute from my custom
my_categorical_multiclass_loss1 = -tf.reduce_sum(tf.cast(y_true_onehot, tf.float64) * tf.math.log(y_pred_softmax), axis=-1)
print(my_categorical_multiclass_loss1)

# Multilabel classifier ------------------------------------------------------------
y_true = np.array([[0, 1, 0, 0], [1,1,0,1]]).reshape(2,4) # 2 samples, with 4 labels
y_pred = np.array([[-1.2, 1.4, 3.0, 5], [2, 3, -1, 0.5]]).reshape(2,4) # logits
y_pred_sigmoid = sigmoid(y_pred)

# Using binary crossentropy from tensorflow
tf_multilabel_loss1 = tf.keras.losses.binary_crossentropy(y_true = y_true, y_pred = y_pred,
                                                                   from_logits=True)
# Using binary crossentropy from tensorflow with my sigmoid function
tf_multilabel_loss2 = tf.keras.losses.binary_crossentropy(y_true = y_true, y_pred = y_pred_sigmoid,
                                                                    from_logits=False)

print(tf_multilabel_loss1)
print(tf_multilabel_loss2)

# Compute from my custom
my_multilabel_loss = tf.reduce_mean(-y_true * tf.math.log(y_pred_sigmoid) - (1 - y_true) * tf.math.log(1 - y_pred_sigmoid), -1)
print(my_multilabel_loss)

# Multioutput classifier
