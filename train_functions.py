# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:51:44 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

from error import Error, ParameterError


def output_history(  # pylint: disable=too-many-arguments
        colors, y_out, savename, show=False,
        title='history', label_x='epoch', label_y='y', loc='upper left'):
    """Output train history."""

    import matplotlib.pyplot as plt

    for y_i, c_i in zip(y_out, colors):
        plt.plot(y_i, color=c_i)
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend(loc=loc)
    if savename:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.close()

def output_signals(  # pylint: disable=too-many-arguments
        time, y_out, colors=None, lss=None,
        savename=None, show=False, label_y='y'):
    """Save and display signals."""

    import matplotlib.pyplot as plt  # pylint: disable=redefined-outer-name

    if savename is None and show is not True:
        raise ParameterError('savename is None and show is not True')
    for yi, ci, li in zip(y_out, colors, lss):  # pylint: disable=invalid-name
        plt.plot(time, yi, color=ci, ls=li)
    plt.xlabel('t')
    plt.ylabel(label_y)
    plt.axis('tight')
    if savename:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.close()
    return

def save_model_struct(model, path_save, model_name, show_shapes=True, show_layer_names=True):
    """Save keras model struct to txt and json files."""
    import os
    import json
    from keras.utils import np_utils, plot_model

    from contextlib import redirect_stdout

    with open(os.path.join(path_save, model_name+'.txt'), 'w') as f_w:
        with redirect_stdout(f_w):
            model.summary()

    plot_model(model, to_file=os.path.join(path_save, model_name+'.svg'))

    with open(os.path.join(path_save, model_name+'.json'), 'w', encoding='utf-8') as f_w:
        json.dump(model.to_json(), f_w)

def save_keras_model(model, filepath, mode=0, **kwargs):
    """Save keras model to disk.
    Args:
        model (keras.Model): keras model to save.
        filepath (string): full file name with path to save.
        mode (int, optional): way to save model. Defaults to 0.
    """

    import json

    if mode == 0:  # save full model
        model.save(filepath, **kwargs)
    elif mode == 1:  # save model architecture
        with open(filepath, 'w', encoding='utf-8') as f_w:
            json.dump(model.to_json(), f_w)
    elif mode == 2:  # save model weights
        model.save_weights(filepath, **kwargs)

def load_keras_model(filepath, mode=0, **kwargs):
    """Load keras model from disk.
    Args:
        filepath (string): full file name with path where saved.
        mode (int, optional): way to load model. Defaults to 0.
    Returns:
        keras.Model: keras model load from disk
    """

    import json
    from keras.models import load_model, model_from_json

    if mode == 0:  # load whole model
        model = load_model(filepath, **kwargs)
    elif mode == 1:  # load model architecture
        json_string = json.load(open(filepath, 'r'))
        model = model_from_json(json_string, **kwargs)

    return model
