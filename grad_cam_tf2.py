# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:56:12 2022

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

It works with tensorflow 2.1.0, not works with tensorflow 1.12.0 and keras 2.2.4,
you may creat a new enviroment and only run this script, and run others with tensorflow 1.12.0 and keras 2.2.4.

References:
    https://github.com/keras-team/keras-io/blob/5307440674625f9201e7faa9a060984d9ac44a87/examples/vision/grad_cam.py#L65
    Title: Grad-CAM class activation visualization
    Author: [fchollet](https://twitter.com/fchollet)
    Date created: 2020/04/26
    Last modified: 2021/03/07
    Description: How to obtain a class activation heatmap for an image classification model.
    Adapted from Deep Learning with Python (2017).
"""


import keras_resnet
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data
import tensorflow as tf


def prepare_input(img_array):
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=0)  # (1, height, wide)
        img_array = np.expand_dims(img_array, axis=-1)  # (1, height, wide, 1)
    return img_array


def build_model_gradcam(num_model, id1, id2, od, fname_weights):
    inputs = tf.keras.layers.Input(shape=(id1, id2, 1))

    if num_model in (120, 12, 13, 10):  # ResNet
        block = keras_resnet.blocks.basic_2d
        parameters = {"kernel_initializer": "he_normal"}
        classes = od
        assert classes > 0
        freeze_bn = False
        dropout_fc = None
        numerical_names = None
        # output_activation = 'sigmoid'
        output_activation = 'linear'

        if num_model == 12:  # ResNet 18
            blocks = [2, 2, 2, 2]

        axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False,
                                   name="conv1", padding="same", **parameters)(inputs)
        x = keras_resnet.layers.BatchNormalization(axis=axis, epsilon=1e-5, freeze=freeze_bn, name="bn_conv1")(x)
        x = tf.keras.layers.Activation("relu", name="conv1_relu")(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    freeze_bn=freeze_bn,
                    parameters=parameters
                )(x)

            features *= 2

        last_conv_layer = x

        x_ = tf.keras.layers.GlobalAveragePooling2D(name="pool5")(x)

        x_ = tf.keras.layers.Dense(classes, activation=output_activation, name=f'fc{classes}')(x_)

        if dropout_fc:
            x_ = tf.keras.layers.Dropout(dropout_fc)(x_)

        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(inputs=inputs, outputs=[last_conv_layer, x_])
        print(grad_model.summary())

        if fname_weights:
            grad_model.load_weights(fname_weights)

        return grad_model


def make_gradcam_heatmap(inputs, grad_model, pred_index=None):
    # We compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        tensor_inputs = tf.convert_to_tensor(inputs)
        tape.watch(tensor_inputs)
        last_conv_layer_output, preds = grad_model(tensor_inputs)
        print(f'last_conv_layer_output {last_conv_layer_output}')
        print(f'preds {preds}')
        tape.watch(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]  # loss
        tape.watch(class_channel)
        print(f'class_channel {class_channel}')

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print(f'grads {grads}')

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, fname_heatmap, fname_superimposed, alpha=0.4):
    img = np.squeeze(img)
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap.save(fname_heatmap)
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # # Superimpose the heatmap on original image
    # img = tf.keras.preprocessing.image.img_to_array(img)
    # superimposed_img = jet_heatmap * alpha + img
    # superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    # # Save the superimposed image
    # superimposed_img.save(fname_superimposed)


if __name__ == '__main__':
    from file_operation import mkdir
    path_data_root = 'C:/data/shipsEar/multiple_class/10547_10547/s0tos3/mix_1to3'
    path_data = path_data_root + '/magspectrum_264_66/s_hdf5'

    path_result = 'E:/BaiduYunTongbu/save_result/shipsEar/result_recognition_mix'
    path_model = path_result + '/magspectrum_264_66_or_rand/model_12_1_3_bs256'
    path_save = os.path.join(path_model, 'save_heatmap')
    mkdir(path_save)

    def compute_and_save_heatmap(path_data, file_name, num_data, path_model, num_model, pred_index, path_save):
        data = np.asarray(read_data(path_data, file_name)[num_data])

        img_array = prepare_input(data)

        img_size = data.shape  # (158, 133)

        fname_weights = os.path.join(path_model, 'weights_1_n3_100_86_0.95.hdf5')

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = build_model_gradcam(num_model, img_size[0], img_size[1], 3, fname_weights)

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, grad_model, pred_index=pred_index)

        # Display heatmap
        plt.matshow(heatmap)
        if path_save:
            path_save_num = os.path.join(path_save, str(num_data))
            mkdir(path_save_num)
            fname_heatmap_class = os.path.join(path_save_num,
                                               f'{file_name[:-5]}_num_{num_data}_class_{pred_index}_heatmap_class')
            plt.savefig(f'{fname_heatmap_class}.eps')
            plt.savefig(f'{fname_heatmap_class}.svg')

            fname_heatmap = os.path.join(path_save_num,
                                         f'{file_name[:-5]}_num_{num_data}_class_{pred_index}_grad_cam_heatmap.eps')

            fname_superimposed = os.path.join(path_save_num,
                                              f'{file_name[:-5]}_num_{num_data}_class_{pred_index}_grad_cam_superimposed.eps')

            save_and_display_gradcam(img_array, heatmap, fname_heatmap, fname_superimposed, alpha=0.4)

    compute_and_save_heatmap(path_data, 's_1.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)

    compute_and_save_heatmap(path_data, 's_2.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_2.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_2.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)

    compute_and_save_heatmap(path_data, 's_3.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_3.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_3.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)

    compute_and_save_heatmap(path_data, 's_1_2.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1_2.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1_2.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)

    compute_and_save_heatmap(path_data, 's_1_3.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1_3.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1_3.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)

    compute_and_save_heatmap(path_data, 's_2_3.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_2_3.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_2_3.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)

    compute_and_save_heatmap(path_data, 's_1_2_3.hdf5', 4999, path_model, 12, 0, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1_2_3.hdf5', 4999, path_model, 12, 1, 'res5b1_relu', path_save)
    compute_and_save_heatmap(path_data, 's_1_2_3.hdf5', 4999, path_model, 12, 2, 'res5b1_relu', path_save)
