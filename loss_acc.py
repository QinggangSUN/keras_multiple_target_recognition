# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:03:06 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
from keras import backend as K  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ParameterError(Error):
    """Exception raised for errors in the Parameter of a function."""
    pass

def subset_acc_nhot_np(y_label, y_predict, threshold=0.5):
    """Labels in n-hot, exactly match ratio, subset accuracy, using numpy.
    Args:
        y_label (np.ndarray,shape==(nsamples,n_src)): n_samples' n-hot length n_src labels.
        y_predict (np.ndarray,shape==(n_sams,n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Raises:
        ParameterError: 'samples number not equal', if y_predict.shape[0] != y_label.shape[0]
    Returns:
        np.float32: exactly match ratio of all samples.
    """
    nsamples = y_label.shape[0]
    if not y_predict.shape[0] == nsamples:
        raise ParameterError('samples number not equal')
    equals = np.zeros((nsamples,), dtype=np.int32)
    for i in range(nsamples):
        y_pi = np.array([np.bool(1) if yij >= threshold else np.bool(0)
                         for yij in y_predict[i]], dtype=np.bool)
        equals[i] = np.all(np.equal(y_pi, y_label[i]), axis=-1)
    return np.mean(equals, axis=-1, dtype=np.float32)

def macro_averaged_acc_nhot_np(y_label, y_predict, threshold=0.5):
    """Labels in n-hot, each source macro-averaged accuracy, using numpy.
    Args:
        y_label (np.ndarray,shape==(nsamples,n_src)): n_samples' n-hot length n_src labels.
        y_predict (np.ndarray,shape==(n_sams,n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Raises:
        ParameterError: 'samples number not equal', if y_predict.shape[0] != y_label.shape[0]
    Returns:
        np.ndarray,shape==(n_src,): each source macro-averaged accuracy of all samples.
    """
    nsamples = y_label.shape[0]
    if not y_predict.shape[0] == nsamples:
        raise ParameterError('labels length not equal')
    equals = np.zeros((nsamples, y_label.shape[1]), dtype=np.int32)
    for i in range(nsamples):
        y_pi = np.array([np.bool(1) if yij >= threshold else np.bool(0)
                         for yij in y_predict[i]], dtype=np.bool)
        equals[i] = np.equal(y_pi, y_label[i])  # pylint: disable=assignment-from-no-return
    return np.mean(equals, axis=0, dtype=np.float32)

def round_y_pred_int_np(y_pred, threshold=0.5):
    """Round float to int, decimal >= threshold to 1, using numpy.
    Args:
        y_pred (np.ndarray,dtype=float): predict outputs, function will force >= 0.
        threshold (float, optional): decimal >= threshold to 1. Defaults to 0.5.
    Returns:
        y_pred (np.ndarray,dtype=np.int32): the numbers rounded.
    Examples:
        >>> y_pred = np.array([[0.1,-0.1],[1.2,1.8],[2.9,2.1],[3.5,3.6]], dtype=np.float32)
        >>> y_pred = round_y_pred_int_np(y_pred)
        [[0 0] [1 2] [3 2] [4 4]]
    """
    zeros = np.zeros_like(y_pred)
    y_pred = np.where(y_pred < 0, zeros, y_pred)  # force y_pred >= 0
    y_pred_floor = np.floor(y_pred)
    y_pred_decimal = y_pred - y_pred_floor
    y_pred = np.where(y_pred_decimal >= threshold, y_pred_floor+1.0, y_pred_floor)
    y_pred = y_pred.astype(np.int32)
    return y_pred

def subset_acc_int_np(y_true, y_pred, threshold=0.5):
    """Labels coded in int, exactly match ratio, subset accuracy, using numpy.
    Args:
        y_true (np.ndarray,shape==(nsamples,n_src): n_samples' n-hot length n_src labels.
        y_pred (np.ndarray,shape==(nsamples,n_src): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        acc (np.float32): exactly match ratio of all samples.
    Examples:
        >>> y_true = np.array([[[0,0]],[[1,2]],[[3,2]],[[4,4]],
                              [[0,1]],[[1,1]],[[3,3]],[[3,4]]], dtype=np.int32)
        >>> y_pred = np.array([[[0.1,-0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]],
                              [[-0.1,0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]]], dtype=np.float32)
        >>> acc = subset_acc_int_np(y_true, y_pred)
        allequals  [ True  True  True  True False False False False]
        acc  0.5
    """
    y_pred = round_y_pred_int_np(y_pred)
    equals = np.equal(y_pred, y_true)
    allequals = np.reshape(np.all(equals, axis=-1), (-1,))
    acc = np.mean(allequals.astype(np.float32), axis=-1)
    return acc

def macro_averaged_acc_int_np(y_true, y_pred, threshold=0.5):
    """Labels coded in int, each source macro-averaged accuracy, using numpy.
    Args:
        y_true (np.ndarray,shape==(nsamples,n_src): n_samples' n-hot length n_src labels.
        y_pred (np.ndarray,shape==(nsamples,n_src): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        acc (np.ndarray,shape==(n_src,),dtype=np.float32): each source macro-averaged accuracy of all samples.
    Examples:
        >>> y_true = np.array([[[0,0]],[[1,2]],[[3,2]],[[4,4]],
                              [[0,1]],[[1,1]],[[3,3]],[[3,4]]], dtype=np.int32)
        >>> y_pred = np.array([[[0.1,-0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]],
                              [[-0.1,0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]]], dtype=np.float32)
        >>> acc = macro_averaged_acc_int_np(y_true, y_pred)
        acc  [[0.875 0.625]]
    """
    y_pred = round_y_pred_int_np(y_pred)
    equals = np.equal(y_pred, y_true)
    acc = np.mean(equals, axis=0, dtype=np.float32)
    return acc

def subset_acc_nhot1(y_true, y_pred, threshold=0.5):
    """Labels in n-hot, exactly match ratio, subset accuracy, using Tensorflow.
    Args:
        y_true (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src labels.
        y_pred (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        tf.tensor,shape==(1,): exactly match ratio of all samples.
    """
    y_pred_round = tf.greater_equal(y_pred, threshold)
    equals = tf.equal(tf.cast(y_pred_round, dtype=tf.bool),
                      tf.cast(y_true, dtype=tf.bool))
    eq_int = tf.to_int32(equals)
    allequals = tf.equal(tf.reduce_sum(eq_int, axis=-1),
                         tf.reduce_sum(tf.ones_like(eq_int), axis=-1))
    acc = tf.reduce_mean(tf.cast(allequals, dtype=tf.float32), axis=-1)
    return acc

def subset_acc_nhot(y_true, y_pred, threshold=0.5):
    """Labels in n-hot, exactly match ratio, subset accuracy, using Tensorflow.
    Args:
        y_true (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src labels.
        y_pred (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        tf.tensor,shape==(1,): exactly match ratio of all samples.
    """
    y_pred_round = tf.greater_equal(y_pred, threshold)
    equals = tf.equal(tf.cast(y_pred_round, dtype=tf.bool),
                      tf.cast(y_true, dtype=tf.bool))
    allequals = tf.reshape(tf.reduce_all(equals, axis=-1), (-1,))
    acc = tf.reduce_mean(tf.cast(allequals, dtype=tf.float32), axis=-1)
    return acc

def binary_acc(y_true, y_pred, threshold=0.5):
    """Only for being compatible with old version of codes, exactly same as subset_acc_nhot.
    Labels in n-hot, exactly match ratio, subset accuracy, using Tensorflow.
    Args:
        y_true (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src labels.
        y_pred (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        tf.tensor,shape==(1,): exactly match ratio of all samples.
    """
    y_pred_round = tf.greater_equal(y_pred, threshold)
    equals = tf.equal(tf.cast(y_pred_round, dtype=tf.bool),
                      tf.cast(y_true, dtype=tf.bool))
    allequals = tf.reshape(tf.reduce_all(equals, axis=-1), (-1,))
    acc = tf.reduce_mean(tf.cast(allequals, dtype=tf.float32), axis=-1)
    return acc

def round_y_pred_int(y_pred, threshold=0.5):
    """Round float to int, decimal >= threshold to 1, using Tensorflow.
    Args:
        y_pred (tf.tensor,dtype=float): predict outputs, function will force >= 0.
        threshold (float, optional): decimal >= threshold to 1. Defaults to 0.5.
    Returns:
        y_pred (tf.tensor,dtype=tf.int32): the numbers rounded.
    Examples:
        >>> y_pred = tf.convert_to_tensor(np.array([[0.1,-0.1],[1.2,1.8],[2.9,2.1],[3.5,3.6]], dtype=np.float32))
        >>> y_pred = round_y_pred_int(y_pred)
        [[0 0] [1 2] [3 2] [4 4]]
    """
    zeros = tf.zeros_like(y_pred)
    y_pred = tf.where(y_pred < 0, x=zeros, y=y_pred)  # force y_pred >= 0
    y_pred_floor = tf.floor(y_pred)
    y_pred_decimal = y_pred - y_pred_floor
    y_pred = tf.where(y_pred_decimal >= threshold, x=y_pred_floor+1.0, y=y_pred_floor)
    y_pred = tf.cast(y_pred, dtype=tf.int32)
    return y_pred

def subset_acc_int(y_true, y_pred, threshold=0.5):
    """Labels coded in int, exactly match ratio, subset accuracy, using Tensorflow.
    Args:
        y_true (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src labels.
        y_pred (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        acc (tf.tensor,shape==(1,),dtype=tf.float32): exactly match ratio of all samples.
    Examples:
        >>> y_pred = tf.convert_to_tensor(np.array([[[0.1,-0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]],
                                                    [[-0.1,0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]]], dtype=np.float32))
        >>> y_true = tf.convert_to_tensor(np.array([[[0,0]],[[1,2]],[[3,2]],[[4,4]],
                                                    [[0,1]],[[1,1]],[[3,3]],[[3,4]]], dtype=np.int32))
        >>> acc = subset_acc_int(y_true, y_pred)
        allequals [ True  True  True  True False False False False]
        acc 0.5
    """
    y_pred = round_y_pred_int(y_pred)
    y_true = tf.cast(y_true, dtype=tf.int32)
    equals = tf.equal(y_pred, y_true)
    allequals = tf.reshape(tf.reduce_all(equals, axis=-1), (-1,))
    acc = tf.reduce_mean(tf.cast(allequals, dtype=tf.float32), axis=-1)
    return acc

def macro_averaged_acc_int(y_true, y_pred, threshold=0.5):
    """Labels coded in int, macro-averaged accuracy, using Tensorflow.
    Args:
        y_true (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src labels.
        y_pred (tf.tensor,shape==(nsamples, 1, n_src)): n_samples' n-hot length n_src predict outputs.
        threshold (float, optional): y_pred = 1 if y_pred > threshold else 0. Defaults to 0.5.
    Returns:
        acc (tf.tensor,shape==(1,),dtype=tf.float32): all source macro-averaged accuracy of all samples.
     Examples:
        >>> y_true = tf.convert_to_tensor(np.array([[[0,0]],[[1,2]],[[3,2]],[[4,4]],
                                                    [[0,1]],[[1,1]],[[3,3]],[[3,4]]], dtype=np.int32))
        >>> y_pred = tf.convert_to_tensor(np.array([[[0.1,-0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]],
                                                    [[-0.1,0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]]], dtype=np.float32))
        >>> acc = macro_averaged_acc_int(y_true, y_pred)
        mean_equals [1.  1.  1.  1.  0.5 0.5 0.5 0.5]
        acc 0.75
    """
    y_pred = round_y_pred_int(y_pred)
    y_true = tf.cast(y_true, dtype=tf.int32)
    equals = tf.equal(y_pred, y_true)
    mean_equals = tf.reshape(tf.reduce_mean(tf.cast(equals, dtype=tf.float32), axis=-1), (-1,))
    acc = tf.reduce_mean(tf.cast(mean_equals, dtype=tf.float32), axis=-1)
    return acc
