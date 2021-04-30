# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:23:07 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import itertools
import logging
import numpy as np
from sklearn.metrics import confusion_matrix

from loss_acc import round_y_pred_int_np
from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels
from recognition_mix_shipsear_s0tos3full3_preprocess import int_combinations, labels_int_short

class ConfusionMatrix(object):
    """Compute confusion Matrix of samples.

    Examples:
        >>> y_pred = np.array([[[0.1,-0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]],
                              [[-0.1,0.1]],[[1.2,1.8]],[[2.9,2.1]],[[3.5,3.6]]], dtype=np.float32)
        >>> y_true = np.array([[[0,0]],[[1,2]],[[3,2]],[[4,4]],
                              [[0,1]],[[1,1]],[[3,3]],[[3,4]]], dtype=np.int32)

        >>> source_confusion = ConfusionMatrix(y_true, y_pred)
        >>> matrix = source_confusion.compute_source_confusion_matrix(4)
        [array([[2, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 0, 1]], dtype=int64),
         array([[1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 2]], dtype=int64)]

        >>> source_confusion = ConfusionMatrix(y_true, y_pred)
        >>> matrix = source_confusion.compute_source_confusion_matrix(None)
        [array([[2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 1],
                [0, 0, 0, 1]], dtype=int64),
         array([[1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 2]], dtype=int64)]


        >>> y_pred = np.array([[0.1,0.1],[0.2,0.8],[0.9,0.1],[0.5,0.6],[0.1,0.1],[0.2,0.8],[0.9,0.1],[0.5,0.6]], dtype=np.float32)
        >>> y_true = np.array([[0,0],[0,1],[1,0],[1,1],[0,1],[0,0],[1,1],[0,1]], dtype=np.bool)

        >>> source_confusion = ConfusionMatrix(y_true, y_pred)
        >>> matrix = source_confusion.compute_source_confusion_matrix(4)
        [array([[4, 1, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]], dtype=int64),
         array([[2, 1, 0, 0, 0],
                [2, 3, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]], dtype=int64)]

        >>> source_confusion = ConfusionMatrix(y_true, y_pred)
        >>> matrix = source_confusion.compute_source_confusion_matrix(None)
        [array([[4, 1],
                [0, 3]], dtype=int64),
         array([[2, 1],
                [2, 3]], dtype=int64)]

        >>> standard_confusion = ConfusionMatrix(y_true, y_pred)
        >>> standard_confusion.standar_label(mode='nhot', n_src=3)
        >>> matrix = standard_confusion.compute_confusion_matrix()
        [[1 0 1 0]
         [0 1 0 0]
         [1 0 1 1]
         [0 1 0 1]]
    """
    def __init__(self, y_true, y_pred, normalize=None):
        super().__init__()
        self.y_true = y_true  # np.ndarray,shape==(n_samples, 1, n_src),dtype==int: true labels of samples.
        self.y_pred = y_pred  # np.ndarray,shape==(n_samples, 1, n_src),dtype==float: predit outputs.
        self.normalize = normalize  # str: {¡®true¡¯, ¡®pred¡¯, ¡®all¡¯}, default=None.
        self.y_true_standard = None  # list[tuple(int)], [n_samples]: true index labels of samples.
        self.y_pred_standard = None  # list[tuple(int)], [n_samples]: predict index labels of samples.
        self.dict_labels = None  # dict map of labels and labels index.
        self.labels = None  # list[list[int]], [n_classes][n_src]: original labels of samples.
        self.labels_standard = None  # list[int], [n_classes]: index of labels.
        self.matrix = None  # confusion matrix.

    def standar_label(self, mode='int', n_src=4, threshold=0.5, labels=None, max_src=3):
        """Standar nhot or multi-int labels to int.
        Args:
            mode (str, optional): Input label type. Defaults to 'int'.
            n_src (int, optional): Number of original sources. Defaults to 4.
            threshold (float, optional): Decimal >= threshold to 1. Defaults to 0.5.
            labels (list[[list[int]], optional): Only label in set of labels can be standarded. Defaults to None.
        """
        if labels is None:
            if mode == 'nhot':
                labels = n_hot_labels(n_src)
            elif mode == 'int':
                labels = labels_int_short(int_combinations(n_src), n_src)
            elif mode == 'int_all':
                labels = list(itertools.product(range(max_src+1), repeat=n_src-1))
        self.labels = labels

        labels_list = []
        for row in labels:
            labels_list.append(tuple(row))

        labels_standard = list(range(len(labels)))
        self.labels_standard = labels_standard
        dict_labels = dict(zip(labels_list, labels_standard))
        self.dict_labels = dict_labels

        y_true_standard = []
        y_pred_standard = []
        for y_true_i, y_pred_i in zip(self.y_true, self.y_pred):
            y_true_standard.append(dict_labels[tuple(y_true_i.reshape(-1,).tolist())])
            y_pred_i_standard = round_y_pred_int_np(y_pred_i.reshape(-1,))
            y_pred_i_standard = np.maximum(y_pred_i_standard, 0)
            y_pred_i_standard = np.minimum(y_pred_i_standard, max_src)            
            y_pred_standard.append(dict_labels[tuple(y_pred_i_standard.tolist())])
        self.y_true_standard = y_true_standard
        self.y_pred_standard = y_pred_standard
        logging.debug(y_true_standard)
        logging.debug(y_pred_standard)

    def compute_confusion_matrix(self):
        """Compute confusion matrix of y_true and y_pred, this works only when all y_pred in labels.
        Returns:
            matrix (np.ndarray,shape==(n_classes, n_classes),dtype==int): Confusion matrix.
        """
        matrix = confusion_matrix(self.y_true_standard, self.y_pred_standard,
                                  labels=self.labels_standard, normalize=self.normalize)
        self.matrix = matrix
        return matrix

    def compute_source_confusion_matrix(self, max_src=3):
        """Compute confusion matrix of each source.
        Args:
            max_src (int, optional): max number of each source. Defaults to 3.
        Returns:
            matrix (list[tuple(int),shape==(n_classes, n_classes)],shape==(n_source,)): Confusion matrix.
        """
        y_true_list = []
        y_pred_list = []
        for y_true_i, y_pred_i in zip(self.y_true, self.y_pred):
            y_true_list.append(tuple(y_true_i.reshape(-1,).tolist()))
            y_pred_list.append(tuple(round_y_pred_int_np(y_pred_i.reshape(-1,)).tolist()))

        n_source = len(y_true_list[0])
        if max_src is None:
            labels = []
            for j in range(n_source):
                labels_j = []
                for i in range(len(y_true_list)):
                    labels_j.append(y_true_list[i][j])
                labels_j = tuple(sorted(set(labels_j)))
                labels.append(labels_j)
        else:
            labels = [tuple(range(max_src+1))] * n_source

        matrix = []
        for j, labels_j in enumerate(labels):
            matrix_j = confusion_matrix(np.asarray(y_true_list)[:, j],
                                        np.asarray(y_pred_list)[:, j],
                                        labels=labels_j,
                                        normalize=self.normalize)
            matrix.append(matrix_j)
        return matrix

def walk_result_dirs(path_save_root, kw_model='.hdf5', num_models=None, **kwargs):
    """Walk all check models dirs under path_save_root.
    Args:
        path_save_root (str): path root of the saved models.
        kw_model (str, optional): file type of the saved models. Defaults to '.hdf5'.
        num_models (list[int], optional): numbers of models. Defaults to None.
    """
    import logging
    from file_operation import list_dirs, walk_dirs_start_str

    dir_save_models = walk_dirs_start_str(path_save_root, 'model_', full=False)
    path_result_dirs = []
    for path_dir_i in dir_save_models:
        path_result_dirs_i = []
        num_model_i = int(path_dir_i[len('model_'):].split('_')[0])
        if num_models is None or (num_models and num_model_i in num_models):
            path_dir_model = os.path.join(path_save_root, path_dir_i, 'model')
            path_dir_models = list_dirs(path_dir_model)
            logging.debug(f'path_dir_models {path_dir_models}')
            path_result_dirs_i.append(path_dir_models)
        if path_result_dirs_i:
            path_result_dirs.append(path_result_dirs_i)
    return path_result_dirs

def save_dict_to_hdf5(dict_s, h5py_fob):
    """Save dictionary datas to hdf5 file using h5py.
    Args:
        dict_s (dict): Dictionary of data.
        h5py_fob (h5py.File): H5py file object.
    """
    for key, value in dict_s.items():
        if isinstance(value, dict):
            fob_child = h5py_fob.create_group(str(key))
            save_dict_to_hdf5(value, fob_child)
        else:
            h5py_fob.create_dataset(str(key), data=value)

def extract_h5py_to_list(h5py_fob, kw, level=1, deep=1, kw2=None):
    """Extract keyword data from h5py file.
    Args:
        h5py_fob (h5py.File): H5py file object.
        kw (str): Data key word.
        kw2 (list[str], optional): Data sub-set key words. Defaults to None.
        level (int, optional): Keyword level in tree of file. Defaults to 1.
        deep (int, optional): Data dimension . Defaults to 1.
    Returns:
        data_list (list): Multi-level list of data, [num][subset].
    """
    data_list = []
    if level == 1:
        if deep == 1:
            for num in h5py_fob.keys():
                grp = h5py_fob[num]
                data_list.append(grp[kw])
        if deep == 2:
            for num in h5py_fob.keys():
                grp = h5py_fob[num]
                data_list.append([grp[kw][kw2_i] for kw2_i in kw2])
    return data_list

def extract_h5py_to_dict(h5py_fob, kw, level=1, deep=1, kw2=None):
    """Extract keyword data from h5py file.

    Args:
        h5py_fob (h5py.File): H5py file object.
        kw (str): Data key word.
        kw2 (list[str], optional): Data sub-set key words. Defaults to None.
        level (int, optional): Keyword level in tree of file. Defaults to 1.
        deep (int, optional): Data dimension. Defaults to 1.

    Returns:
        data_dict (dict): Multi-level dictionary of data.

    Examples:
        >>> PATH_SAVE_ROOT = '../result_recognition_mix_full3'
        >>> path_save_result = os.path.join(PATH_SAVE_ROOT, 'result')
        >>> file_result = h5py.File(os.path.join(path_save_result, 'result.hdf5'), 'r')

        >>> data_dict = extract_h5py_to_dict(file_result, 'file_name')
        >>> print('data_dict', data_dict)
        >>> print('data_frame', dict_to_df(data_dict))

        >>> data_dict = extract_h5py_to_dict(file_result, 'subset_acc', deep=2, kw2=['train', 'val', 'test'])
        >>> print('data_dict', data_dict)
        >>> print('data_frame', dict_to_df(data_dict, deep=2))

        >>> data_dict = extract_h5py_to_dict(file_result, 'macro_averaged_acc', deep=3, kw2=['train', 'val', 'test'])
        >>> print('data_dict', data_dict)
        >>> print('data_frame', dict_to_df(data_dict, deep=3))
    """

    if level == 1:
        if deep == 1:
            data_dict = {kw:[]}
            for num in h5py_fob.keys():
                grp = h5py_fob[num]
                data_dict[kw].append(np.asarray(grp[kw]))
        if deep == 2 or deep == 3:
            data_dict = {kw:dict()}
            for kw2_i in kw2:
                data_dict[kw].update({kw2_i:[]})
            for num in h5py_fob.keys():
                grp = h5py_fob[num]
                for kw2_i in kw2:
                    data_dict[kw][kw2_i].append(np.asarray(grp[kw][kw2_i]))
    return data_dict

def dict_to_df(data_dict, deep=1):
    """Convert dictionary of data to pandas.DataFrame.

    Args:
        data_dict (dict): Multi-level dictionary of data.
        deep (int, optional): Data dimension. Defaults to 1.

    Returns:
        (pd.DataFrame): Pandas DataFrame of data.
    """
    print('data_dict', data_dict)
    if deep == 1:
        return pd.DataFrame(data_dict)
    elif deep == 2:
        dict_series = dict()
        name = list(data_dict.keys())[0]
        for set_name, data in data_dict[name].items():
            dict_series.update({f'{name}_{set_name}':pd.Series(data)})
        logging.debug(f'dict_series {dict_series}')
        return pd.DataFrame(dict_series)
    elif deep == 3:
        dict_series = dict()
        name = list(data_dict.keys())[0]
        for set_name, data in data_dict[name].items():
            data_trans = np.transpose(np.asarray(data))
            for i, data_i in enumerate(data_trans):
                dict_series.update({f'{name}_{set_name}_{i}':pd.Series(data_i)})
        logging.debug(f'dict_series {dict_series}')
        return pd.DataFrame(dict_series)

import pandas as pd
def save_h5py_to_csv(h5py_fob, file_csv, data_names, paras):
    """Read data from .hdf5 file, save data to .csv file.

    Args:
        h5py_fob (h5py.File): H5py file object.
        file_csv (str): Name of .csv file to save.
        data_names (list[str]): Names of the datas to save.
        paras (list[dict]): Paras to describe data struct.

    Returns:
        result_df (pd.DataFrame): Pandas DataFrame of data.
    """

    result_df = []
    for name, para in zip(data_names, paras):
        level = para['level'] if 'level' in para.keys() else 1
        deep = para['deep'] if 'deep' in para.keys() else 1
        kw2 = para['kw2'] if 'kw2' in para.keys() else None
        data_dict = extract_h5py_to_dict(h5py_fob, name, level, deep, kw2)
        data_df = dict_to_df(data_dict, deep)
        result_df.append(data_df)
    logging.debug('result_df', result_df)
    result_df = pd.concat(result_df, axis=1)
    logging.debug('result_df', result_df)

    result_df.to_csv(file_csv)
    h5py_fob.close()
    return result_df

if __name__ == '__main__':
    import h5py
    import json
    import logging
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import os
    from file_operation import list_dirs, list_files_end_str, mkdir
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, save_datas

    def walk_result_files(path_feature, kw_dir='loss'):
        """Walk result files.

        Args:
            path_feature (str): Root path of the result of a feature.
            kw_dir (str, optional): Where sub dir result files saved. Defaults to 'loss'.

        Returns:
            path_result_files (list[(str)]): List of the names of the reult files.
        """
        path_result_files = []
        path_result_dirs = walk_result_dirs(path_feature)  # list[model_i][model_i_j][learn_rate]
        logging.debug(path_result_dirs)
        for path_result_dirs_i in path_result_dirs:
            for path_result_dirs_j in path_result_dirs_i:
                for path_result_dir_k in path_result_dirs_j:
                    path_result_files_i_j_k = list_files_end_str(
                        os.path.join(path_result_dir_k, kw_dir), '.hdf5')
                    logging.debug(path_result_files_i_j_k)
                    path_result_files += path_result_files_i_j_k
        return path_result_files

    class SeeResult(object):
        def __init__(self, path_result_files, mode='int', n_src=4):
            super().__init__()
            self.path_result_files = path_result_files
            self.mode = mode
            self.n_src = n_src

        def see_metrics(self, dict_metric, path_save=None):
            """See metrics of the results.

            Args:
                dict_metric (dict{str:bool}): Whether see the feature.
                path_save (str, optional): Path to save metrics. Defaults to None.

            Returns:
                dict_results (dict): Mutiple layer dictionary of the metrics.
            """
            path_result_files = self.path_result_files
            dict_results = dict()
            for i, path_result_file in enumerate(path_result_files):
                dict_result = {'file_name':path_result_file}
                if ('source_confusion' in dict_metric.keys() and dict_metric['source_confusion'] or
                    'standard_confusion' in dict_metric.keys() and dict_metric['standard_confusion']):
                    path, filename = os.path.split(path_result_file)
                    y_true_sets = [read_data(path, filename, 'hdf5', set_i) for set_i in ['l_train', 'l_val', 'l_test']]
                    y_pred_sets = [read_data(path, filename, 'hdf5', set_i) for set_i in ['p_train', 'p_val', 'p_test']]

                    if 'standard_confusion' in dict_metric.keys() and dict_metric['standard_confusion']:
                        matrix_sets = []  # [set](n_classes, n_classes)
                        for y_true, y_pred in zip(y_true_sets, y_pred_sets):
                            confusion = ConfusionMatrix(y_true, y_pred)
                            mode = 'int_all' if self.mode == 'int' else self.mode
                            confusion.standar_label(mode=mode, n_src=self.n_src, threshold=0.5)
                            matrix_sets.append(confusion.compute_confusion_matrix())
                        logging.debug('standard_confusion')
                        logging.debug(matrix_sets)
                        dict_result.update({'standard_confusion':matrix_sets})

                    if 'source_confusion' in dict_metric.keys() and dict_metric['source_confusion']:
                        matrix_sets = []  # [set][src](n_classes, n_classes)
                        for y_true, y_pred in zip(y_true_sets, y_pred_sets):
                            source_confusion = ConfusionMatrix(y_true, y_pred)
                            matrix_sets.append(source_confusion.compute_source_confusion_matrix(dict_metric['max_src']))
                        logging.debug('source_confusion')
                        logging.debug(matrix_sets)
                        dict_result.update({'source_confusion':matrix_sets})

                if 'subset_acc' in dict_metric.keys() and dict_metric['subset_acc']:
                    subset_acc_sets = dict()
                    for set_i in ['train', 'val', 'test']:
                        subset_acc_sets.update(
                                {set_i: read_data(path, filename, 'hdf5', f'subset_acc_{self.mode}_{set_i}')})
                    logging.debug(f'subset_acc_{self.mode}')
                    logging.debug(subset_acc_sets)
                    dict_result.update({'subset_acc':subset_acc_sets})

                if 'binary_acc' in dict_metric.keys() and dict_metric['binary_acc']:
                    subset_acc_sets = dict()
                    for set_i in ['train', 'val', 'test']:
                        subset_acc_sets.update(
                                {set_i: read_data(path, filename, 'hdf5', f'binary_acc_{set_i}')})
                    logging.debug(f'subset_acc_{self.mode}')
                    logging.debug(subset_acc_sets)
                    dict_result.update({'subset_acc':subset_acc_sets})

                if 'macro_averaged_acc' in dict_metric.keys() and dict_metric['macro_averaged_acc']:
                    macro_averaged_acc_sets = dict()
                    for set_i in ['train', 'val', 'test']:
                        macro_averaged_acc_sets.update(
                                {set_i: read_data(path, filename, 'hdf5', f'macro_averaged_acc_{self.mode}_{set_i}')})
                    logging.debug(f'macro_averaged_acc_{self.mode}')
                    logging.debug(macro_averaged_acc_sets)
                    dict_result.update({'macro_averaged_acc':macro_averaged_acc_sets})

                if 'acc' in dict_metric.keys() and dict_metric['acc']:
                    macro_averaged_acc_sets = dict()
                    for set_i in ['train', 'val', 'test']:
                        macro_averaged_acc_sets.update(
                                {set_i: read_data(path, filename, 'hdf5', f'acc_{set_i}')})
                    logging.debug(f'macro_averaged_acc_{self.mode}')
                    logging.debug(macro_averaged_acc_sets)
                    dict_result.update({'macro_averaged_acc':macro_averaged_acc_sets})

                dict_results.update({i:dict_result})
            logging.debug(dict_results)

            self.path_save = path_save
            f_a = h5py.File(os.path.join(path_save, 'result.hdf5'), 'a')
            save_dict_to_hdf5(dict_results, f_a)

            return dict_results

        def confusion_matrix_plot(self,
                                  file_result=None,
                                  confusion_name='standard_confusion',
                                  path_plot=None,
                                  file_type='.svg',
                                  label_standard=tuple(range(0, 8)),
                                  label_source=tuple(range(0, 4)),
                                  ):

            if file_result is None:
                file_result = os.path.join(os.path.join(self.path_save, 'result.hdf5'))

            if path_plot is None:
                path_plot = os.path.join(self.path_save, confusion_name)
                mkdir(path_plot)

            data_dict = extract_h5py_to_dict(h5py.File(file_result, 'r'), confusion_name)  # {key:list[np.ndarray]}
            if confusion_name == 'standard_confusion':
                for num, matrix_set in enumerate(data_dict[confusion_name]):  # [set](n_classes, n_classses)
                    for name_set, matrix in zip(['train', 'val', 'test'], matrix_set):
                        file_plot = os.path.join(path_plot, f'{num}_{name_set}{file_type}')
                        matrix = matrix / matrix.sum(axis=1, keepdims=True)
                        ConfusionMatrixDisplay(matrix, label_standard).plot()
                        plt.savefig(file_plot)
                        plt.close()

            elif confusion_name == 'source_confusion':
                for num, matrix_set in enumerate(data_dict[confusion_name]):  # [set][n_src](n_classes, n_classses)
                    for name_set, matrix_srcs in zip(['train', 'val', 'test'], matrix_set):
                        for i, matrix in enumerate(matrix_srcs):
                            logging.debug(f'matrix {matrix}')
                            file_plot = os.path.join(path_plot, f'{num}_{i}_{name_set}{file_type}')
                            matrix = matrix / matrix.sum(axis=1, keepdims=True)
                            ConfusionMatrixDisplay(matrix, label_source).plot()
                            plt.savefig(file_plot)
                            plt.close()

# =============================================================================
    PATH_SAVE_ROOT = '../result_recognition_mix_full3'
# -----------------------------------------------------------------------------
#    PATH_SAVE_ROOT = '../result_recognition'
# =============================================================================
    path_save_result = os.path.join(PATH_SAVE_ROOT, 'result')
    mkdir(path_save_result)
# =============================================================================
#    # only for extract part of the results
#    path_result_files = []
#    WIN_LIST = [3164]
#    HOP_LIST = [ 791]
#    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
#        path_feature = os.path.join(PATH_SAVE_ROOT, f'magspectrum_{win_i}_{hop_i}_or_rand')
#        path_result_files_feature = walk_result_files(path_feature)
#        path_result_files += path_result_files_feature
# -----------------------------------------------------------------------------
    path_result_files = []
    path_features = list_dirs(PATH_SAVE_ROOT)
    for path_feature in path_features:
        path_result_files_feature = walk_result_files(path_feature)
        path_result_files += path_result_files_feature
# =============================================================================
    see_result_full3 = SeeResult(path_result_files, 'int')
    dict_metrics = {'source_confusion':True,
                    'standard_confusion':True,
                    'subset_acc':True,
                    'macro_averaged_acc':True,
                    'max_src':3}
    see_result_full3.see_metrics(dict_metrics, path_save_result)
    see_result_full3.confusion_matrix_plot(confusion_name='source_confusion')
## -----------------------------------------------------------------------------
#    see_result = SeeResult(path_result_files, 'nhot')
##    dict_metrics = {'source_confusion':False,
##                    'standard_confusion':True,
##                    'subset_acc':True,
##                    'macro_averaged_acc':True,
##                    'max_src':1}
#    # only for being compatible with old version, which named 'binary_acc' instead of 'subset_acc_nhot'
#    dict_metrics = {'source_confusion':False,
#                    'standard_confusion':True,
#                    'binary_acc':True,
#                    'acc':True,
#                    'max_src':1}
#    see_result.see_metrics(dict_metrics, path_save_result)
#    see_result.confusion_matrix_plot(confusion_name='standard_confusion')
# =============================================================================
    file_result = h5py.File(os.path.join(path_save_result, 'result.hdf5'), 'r')
# =============================================================================
    data_names = ['file_name',
                  'subset_acc',
                  'macro_averaged_acc']
    paras = [dict(),
             {'deep':2, 'kw2':['train', 'val', 'test']},
             {'deep':3, 'kw2':['train', 'val', 'test']}]
# -----------------------------------------------------------------------------
#    data_names = ['file_name',
#                  'subset_acc',
#                  'macro_averaged_acc']
#    paras = [dict(),
#             {'deep':2, 'kw2':['train', 'val', 'test']},
#             {'deep':2, 'kw2':['train', 'val', 'test']}]
# -----------------------------------------------------------------------------
    file_csv = os.path.join(path_save_result, 'result.csv')
    save_h5py_to_csv(file_result, file_csv, data_names, paras)
# =============================================================================

    print('finished')

