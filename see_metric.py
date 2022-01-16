# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:20:38 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score

from loss_acc import round_y_pred_int_np, macro_averaged_acc_nhot_np, macro_averaged_acc_int_np
from loss_acc import macro_averaged_precision_nhot_np, macro_averaged_recall_nhot_np, f1_score_np
from loss_acc import macro_averaged_precision_int_np, macro_averaged_recall_int_np
from loss_acc import subset_acc_nhot_np, subset_acc_int_np
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data


def compute_metrics_binary(y_true, y_pred, threshold=None, class_names=None, acc_type="macro",
                           fname_prc_all=None, fname_prc_each=None):

    y_score = round_y_pred_int_np(y_pred, threshold=threshold) if threshold else y_pred
    n_classes = y_true.shape[1]

    subset_acc = subset_acc_nhot_np(y_true, y_score, threshold=threshold)
    macro_accs = macro_averaged_acc_nhot_np(y_true, y_score, threshold=threshold)
    macro_precisions = macro_averaged_precision_nhot_np(y_true, y_score, threshold=threshold)
    macro_recalls = macro_averaged_recall_nhot_np(y_true, y_score, threshold=threshold)
    macro_f1_scores = f1_score_np(macro_precisions, macro_recalls)
    macro_avg_acc_np = np.mean(macro_accs)
    macro_avg_precision_np = np.mean(macro_precisions)
    macro_avg_recall_np = np.mean(macro_recalls)
    macro_avg_f1_score_np = np.mean(macro_f1_scores)

    if threshold:
        macro_avg_acc_score = np.mean([accuracy_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)])
        macro_avg_precision = np.mean([precision_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)])
        macro_avg_recall = np.mean([recall_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)])
        macro_avg_f1_score = np.mean([f1_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)])

    if fname_prc_all or fname_prc_each:
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        # accuracy_scores = dict()
        # precision_scores = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])
            # accuracy_scores[i] = accuracy_score(y_true[:, i], y_score[:, i])
            # precision_scores[i] = precision_score(y_true[:, i], y_score[:, i])
        avg_precision_score = average_precision_score(y_true, y_score)

    if fname_prc_all:
        precision[acc_type], recall[acc_type], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
        average_precision[acc_type] = average_precision_score(y_true, y_score, average=acc_type)

        display = PrecisionRecallDisplay(
            recall=recall[acc_type],
            precision=precision[acc_type],
            average_precision=average_precision[acc_type],
        )
        display.plot()
        _ = display.ax_.set_title(f'{acc_type.capitalize()}-averaged over all classes')
        plt.savefig(fname_prc_all)
        plt.show()

    if fname_prc_each:
        precision[acc_type], recall[acc_type], _ = precision_recall_curve(
            y_true.ravel(), y_score.ravel()
        )
        average_precision[acc_type] = average_precision_score(y_true, y_score, average=acc_type)

        # setup plot details
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)

        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall[acc_type],
            precision=precision[acc_type],
            average_precision=average_precision[acc_type],
        )
        display.plot(ax=ax, name=f'{acc_type.capitalize()}-average precision-recall', color="gold")

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            if class_names is None:
                display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
            else:
                display.plot(ax=ax, name=f"Precision-recall for {class_names[i]}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["F1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        plt.savefig(fname_prc_each)
        plt.show()


def compute_metrics_int(y_true, y_pred, threshold=None, class_names=None, acc_type="macro",
                        fname_prc_all=None, fname_prc_each=None):

    y_score = round_y_pred_int_np(y_pred, threshold=threshold) if threshold else y_pred
    n_classes = y_true.shape[1]

    subset_acc = subset_acc_int_np(y_true, y_score, threshold=threshold)
    macro_accs = macro_averaged_acc_int_np(y_true, y_score, threshold=threshold)
    macro_precisions_np = macro_averaged_precision_int_np(y_true, y_score, threshold=threshold)
    macro_recalls_np = macro_averaged_recall_int_np(y_true, y_score, threshold=threshold)
    macro_f1_scores_np = f1_score_np(macro_precisions_np, macro_recalls_np)
    macro_avg_acc_np = np.mean(macro_accs)
    macro_avg_precision_np = np.mean(macro_precisions_np)
    macro_avg_recall_np = np.mean(macro_recalls_np)
    macro_avg_f1_score_np = np.mean(macro_f1_scores_np)

    if threshold:
        macro_avg_acc_score = np.mean([accuracy_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)])
        # do NOT use this method, which compute all the values in labels and predictions
        # macro_precisions = [precision_score(y_true[:, i], y_score[:, i], average='macro') for i in range(n_classes)]
        # macro_avg_precision = np.mean([precision_score(y_true[:, i], y_score[:, i],
        #                               average='macro') for i in range(n_classes)])
        # macro_recalls = [recall_score(y_true[:, i], y_score[:, i], average='macro') for i in range(n_classes)]
        # macro_avg_recall = np.mean([recall_score(y_true[:, i], y_score[:, i], average='macro')
        #                            for i in range(n_classes)])
        # macro_f1_scores = [f1_score(y_true[:, i], y_score[:, i], average='macro') for i in range(n_classes)]
        # macro_avg_f1_score = np.mean([f1_score(y_true[:, i], y_score[:, i], average='macro') for i in range(n_classes)])
    if fname_prc_all or fname_prc_each:
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        # accuracy_scores = dict()
        # precision_scores = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])
            # accuracy_scores[i] = accuracy_score(y_true[:, i], y_score[:, i])
            # precision_scores[i] = precision_score(y_true[:, i], y_score[:, i])
        avg_precision_score = average_precision_score(y_true, y_score)

    if fname_prc_all:
        precision[acc_type], recall[acc_type], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
        average_precision[acc_type] = average_precision_score(y_true, y_score, average=acc_type)

        display = PrecisionRecallDisplay(
            recall=recall[acc_type],
            precision=precision[acc_type],
            average_precision=average_precision[acc_type],
        )
        display.plot()
        _ = display.ax_.set_title(f'{acc_type.capitalize()}-averaged over all classes')
        plt.savefig(fname_prc_all)
        plt.show()

    if fname_prc_each:
        precision[acc_type], recall[acc_type], _ = precision_recall_curve(
            y_true.ravel(), y_score.ravel()
        )
        average_precision[acc_type] = average_precision_score(y_true, y_score, average=acc_type)

        # setup plot details
        colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)

        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall[acc_type],
            precision=precision[acc_type],
            average_precision=average_precision[acc_type],
        )
        display.plot(ax=ax, name=f'{acc_type.capitalize()}-average precision-recall', color="gold")

        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            if class_names is None:
                display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
            else:
                display.plot(ax=ax, name=f"Precision-recall for {class_names[i]}", color=color)

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["F1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")

        plt.savefig(fname_prc_each)
        plt.show()


def see_metircs_binary_data(path_data, fname_pred, dict_key_true, dict_key_pred, path_save,
                            threshold=0.5, class_names=['RORO', 'motorboat', 'passenger']):
    filetype = os.path.splitext(fname_pred)[1][1:]
    filename = os.path.splitext(fname_pred)[0]
    if filetype == 'hdf5':
        y_true = np.squeeze(np.asarray(read_data(path_data, fname_pred, dict_key=dict_key_true)))
        y_pred = np.squeeze(np.asarray(read_data(path_data, fname_pred, dict_key=dict_key_pred)))
    elif filetype == 'json':
        y_true = np.squeeze(np.asarray(read_data(path_data, filename, filetype)[dict_key_true]))
        y_pred = np.squeeze(np.asarray(read_data(path_data, filename, filetype)[dict_key_pred]))

    compute_metrics_binary(y_true, y_pred, threshold=threshold)
    compute_metrics_binary(y_true, y_pred,
                           class_names=class_names,
                           fname_prc_all=os.path.join(path_save, 'precision-recall_overall.eps'),
                           fname_prc_each=os.path.join(path_save, 'precision-recall_each.eps'))


def see_metircs_int_data(path_data, fname_pred, dict_key_true, dict_key_pred, path_save,
                         threshold=0.5, class_names=['RORO', 'motorboat', 'passenger']):
    filetype = os.path.splitext(fname_pred)[1][1:]
    filename = os.path.splitext(fname_pred)[0]
    if filetype == 'hdf5':
        y_true = np.squeeze(np.asarray(read_data(path_data, fname_pred, dict_key=dict_key_true)))
        y_pred = np.squeeze(np.asarray(read_data(path_data, fname_pred, dict_key=dict_key_pred)))
    elif filetype == 'json':
        y_true = np.squeeze(np.asarray(read_data(path_data, filename, filetype)[dict_key_true]))
        y_pred = np.squeeze(np.asarray(read_data(path_data, filename, filetype)[dict_key_pred]))

    compute_metrics_int(y_true, y_pred, threshold=threshold)
    # TODO, the PR curve cannot be computed by sklearn
    # compute_metrics_int(y_true, y_pred,
    #                     class_names=class_names,
    #                     fname_prc_all=os.path.join(path_save, 'precision-recall_overall.svg'),
    #                     fname_prc_each=os.path.join(path_save, 'precision-recall_each.svg'))


if __name__ == '__main__':
    PATH_DATA_ROOT = 'C:/data/shipsEar/multiple_class/10547_10547/s0tos3/mix_1to3'
    PATH_DATA = PATH_DATA_ROOT + '/wavmat/original_rand'

    PATH_SAVE_ROOT = '../result_recognition_mix'

    PATH_SAVE = PATH_SAVE_ROOT + '/magspectrum_264_66_or_rand/model_12_1_3_bs256'
    fname_pred = 'test_weights_1_n3_100_86_0.95.hdf5.json'

    PATH_SAVE = PATH_SAVE_ROOT + '/wavmat_or_rand/model_90_1_4_bs64'
    fname_pred = 'test_weights_1_n3_100_47_0.75.hdf5.hdf5'

    PATH_SAVE = PATH_SAVE_ROOT + '/magspectrum_264_66_or_rand/model_10_1_3_bs64'
    fname_pred = 'test_weights_1_n3_100_61_0.94.hdf5.json'

    PATH_SAVE = PATH_SAVE_ROOT + '/real_img_spectrum_264_66_or_rand/model_16_1_1_bs64'
    fname_pred = 'test_weights_1_n3_100_77_0.94.hdf5.json'

    PATH_SAVE = PATH_SAVE_ROOT + '/logmelspectrum_3164_791_128_or_rand/model_12_1_3_bs64'
    fname_pred = 'test_weights_1_n3_100_90_0.94.hdf5.json'

    PATH_SAVE = PATH_SAVE_ROOT + '/mfcc_3164_791_512_160_or_rand/model_14_1_3'
    fname_pred = 'test_weights_1_n3_100_42_0.67.hdf5.json'

    # see_metircs_binary_data(PATH_SAVE, fname_pred, 'l_train', 'p_train', PATH_SAVE,
    #                         threshold=0.5, class_names=['RORO', 'motorboat', 'passenger'])

    # see_metircs_binary_data(PATH_SAVE, fname_pred, 'l_val', 'p_val', PATH_SAVE,
    #                         threshold=0.5, class_names=['RORO', 'motorboat', 'passenger'])

    see_metircs_binary_data(PATH_SAVE, fname_pred, 'l_test', 'p_test', PATH_SAVE,
                            threshold=0.5, class_names=['RORO', 'motorboat', 'passenger'])

    PATH_SAVE_ROOT = '../result_recognition_mix_full3'

    PATH_SAVE = PATH_SAVE_ROOT + '/wavmat_or_rand/model_9_1_2'
    fname_pred = 'test_weights_1_n3_100_79_0.29.hdf5.hdf5'

    PATH_SAVE = PATH_SAVE_ROOT + '/magspectrum_264_66_or_rand/model_13_1_1'
    fname_pred = 'test_weights_1_n3_100_50_0.49.hdf5.hdf5'

    PATH_SAVE = PATH_SAVE_ROOT + '/real_img_spectrum_264_66_or_rand/model_16_1_2'
    fname_pred = 'test_weights_1_n3_100_85_0.49.hdf5.hdf5'

    PATH_SAVE = PATH_SAVE_ROOT + '/logmelspectrum_3164_791_128_or_rand/model_10_1_1'
    fname_pred = 'test_weights_1_n3_100_99_0.51.hdf5.hdf5'

    PATH_SAVE = PATH_SAVE_ROOT + '/mfcc_3164_791_512_80_or_rand/model_10_1_1'
    fname_pred = 'test_weights_1_n3_100_95_0.49.hdf5.hdf5'

    see_metircs_int_data(PATH_SAVE, fname_pred, 'l_test', 'p_test', PATH_SAVE,
                         threshold=0.5, class_names=['RORO', 'motorboat', 'passenger'])

    print('finished')
