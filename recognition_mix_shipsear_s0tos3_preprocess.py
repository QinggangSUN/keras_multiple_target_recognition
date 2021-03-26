# -*- coding: utf-8 -*-
"""
Created on Wed March 18 20:34:30 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

def n_hot_labels(nsrc):
    """Return an mixed sources n_hot labels matix with input number of nsrc.
    Args:
        nsrc (int): The number of sources.
    Returns:
        list[[int]]: a 2d list with shape 2**(nsrc-1) * (nsrc-1)
    Examples:
        Input 4, return 8*3 mix labels matix.
        >>> print(n_hot_labels(4))
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
        [0, 1, 1], [1, 1, 1]]

    """
    import numpy as np  # pylint: disable=redefined-outer-name
    from itertools import combinations
    labels = []     # n_hot labels
    labels.extend(np.zeros((1, nsrc-1), dtype=int).tolist())    # s0
    labels.extend(np.eye(nsrc-1, dtype=int).tolist())    # s1tos3
    for i in range(2, nsrc, 1):
        # e.g. [(1,2)(1,3)(2,3)(1,2,3)]
        index_ci = list(combinations(range(nsrc-1), i))
        n_ci = len(index_ci)
        labels_ci = np.zeros((n_ci, nsrc-1), dtype=int)
        for j in range(n_ci):   # e.g. (1,2)
            for index_ci_j_k in index_ci[j]:
                labels_ci[j, index_ci_j_k] = 1
        labels.extend(labels_ci.tolist())
    return labels

def subset_nums_create(path_source_root, sub_set_way, rates, n_samples, n_sources):
    """Save sub_set nums.
    Args:
        path_source_root (str): path root where data is.
        sub_set_way (str): ['order', 'rand'] way to subset data.
        rates (list[float]): The rates of each sub dataset, e.g. train val test.
        n_samples (int): Number of samples.
        n_sources (int): Number of mix sources.
    """
    import os
    import pickle  # pylint: disable=redefined-outer-name
    import json  # pylint: disable=redefined-outer-name
    from prepare_data import Subsets
    from prepare_data import shuffle_sets
    rss1 = Subsets(rates, n_samples)
    # nums: 3D list [sourcei][subseti][numi]
    if sub_set_way == 'rand':
        nums = [rss1.randsubsetsnums(n_samples) for i in range(n_sources)]
    elif sub_set_way == 'order':
        nums = [rss1.ordersubsetsnums(n_samples) for i in range(n_sources)]

    # return 2D list [subseti][(sourcei, numi)]
    nums_rand = shuffle_sets(nums)
    with open(os.path.join(path_source_root, f'nums_{sub_set_way}.pickle'), 'wb') as f_wb:
        pickle.dump(nums_rand, f_wb)
    with open(os.path.join(path_source_root, f'nums_{sub_set_way}.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'data': nums_rand}, f_w)

def subset_x(source_frames, nums_rand):
    """Sub_set feature datasets x.
    Args:
        source_frames (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources scalered.
        nums_rand (list[pair(int, int)]): [n_set](n_source, index), index of rand data.
    Returns:
        x_sets (list[np.ndarray,shape==(n_samples,)+feature_shape]): feature datasets.
    """
    import numpy as np  # pylint: disable=redefined-outer-name
    x_sets = []
    for nums_i in nums_rand:
        x_sets_i = []
        for pair_i in nums_i:
            x_sets_i.append(source_frames[pair_i[0]][pair_i[1]])
        x_sets.append(np.asarray(x_sets_i, dtype=np.float32))
    return x_sets

def y_sets_create(nums_rand, y_labels, n_src):
    """Create label data y_sets.
    Args:
        nums_rand (list[pair(int, int)]): [n_set](n_source, index), index of rand data.
        y_labels (list[list[int]): [n_sources][n_src] label of mix sources.
        n_src (int): number of original sources.
    """
    import numpy as np
    y_sets = []
    for pair_si in nums_rand:
        label_i = [y_labels[pair_i[0]] for pair_i in pair_si]
        y_sets.append(
            np.asarray(label_i, dtype=np.int32).reshape(-1, 1, n_src-1))
    return y_sets

from sklearn import preprocessing
import prepare_data_shipsear_recognition_mix_s0tos3 as m_pre_data_shipsear

class XsetSourceFrames(object):
    """Read and scaler data x_sets."""

    def __init__(self, path_source_root, dir_names, **kwargs):
        self._path_source_root = path_source_root
        self._dir_names = dir_names

        # Load data x_sets.
        if 'mode_read' in kwargs.keys():
            self._source_frames = np.asarray(
                m_pre_data_shipsear.read_datas(
                    os.path.join(self._path_source_root, 's_hdf5'),
                    self._dir_names, **{'mode':kwargs['mode_read']}), dtype=np.float32)
        else:
            self._source_frames = np.asarray(
                m_pre_data_shipsear.read_datas(
                    os.path.join(self._path_source_root, 's_hdf5'), self._dir_names), dtype=np.float32)

        self.n_sources = self._source_frames.shape[0]    # = nmixsources
        # number of samples per mixsource
        self.n_samples = self._source_frames.shape[1]
        self.feature_shape = self._source_frames.shape[2:]

    def get_source_frames(self):
        """Get the x_sets data _source_frames.
        Returns:
            self._source_frames (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources to scaler.
        """
        return self._source_frames

    def sourceframes_mm_create(self):
        """Scaler data feature.
        Args:
            self._source_frames (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources to scaler.
            self.n_sources (int): number of the sources.
            self.n_samples (int): number of samples per source.
        Returns:
            self._sourceframes_mm (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources scalered.
        """
        scaler_mm = preprocessing.MinMaxScaler()   # to [0,1]
        # return 2D np.array [num][feature]
        self._sourceframes_mm = scaler_mm.fit_transform(  # pylint: disable=attribute-defined-outside-init
            self._source_frames.reshape(
                self.n_sources*self.n_samples, -1))
        # return 3D np.array [n_sources][n_samples][feature]
        self._sourceframes_mm = self._sourceframes_mm.reshape(  # pylint: disable=attribute-defined-outside-init
            (self.n_sources, self.n_samples)+self.feature_shape)
        return self._sourceframes_mm


if __name__ == '__main__':
    import os
    import logging
    import numpy as np
    import pickle
    import json

    from file_operation import mkdir

    np.random.seed(1337)  # for reproducibility
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)


    def data_create(path_class, rates_set, **kwargs):  # pylint: disable=too-many-locals
        """Create X_train, X_val, X_test scalered data, and
            create Y_train, Y_val, Y_test data labels.
        Args:
            path_class (object class PathSourceRoot): object of class to compute path.
            rates_set (list[float]): rates of datasets.
        """
        path_source = path_class.path_source
        path_source_root = path_class.path_source_root
        scaler_data = path_class.get_scaler_data()
        sub_set_way = path_class.sub_set_way

        dir_names = json.load(
            open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname']

        x_source_frames_class = XsetSourceFrames(path_source_root, dir_names, **kwargs)
        if scaler_data == 'or':
            x_source_frames = x_source_frames_class.get_source_frames()
        elif scaler_data == 'mm':
            x_source_frames = x_source_frames_class.sourceframes_mm_create()
        logging.info('x_source_frames read and scaler finished')

        n_samples = x_source_frames_class.n_samples
        n_sources = x_source_frames_class.n_sources
        if not os.path.isfile(os.path.join(path_source_root, 'nums_'+sub_set_way+'.pickle')):
            subset_nums_create(
                path_source_root, sub_set_way, rates_set, n_samples, n_sources)
        with open(os.path.join(path_source_root, 'nums_'+sub_set_way+'.pickle'), 'rb') as f_rb:
            nums_rand = pickle.load(f_rb)

        x_sets = subset_x(x_source_frames, nums_rand)
        logging.info('x_sets created finished')

        # n_sources = len(dir_names) = N_SRC + 2^(N_SRC-1)-1-(N_SRC-1)
        n_src = int(np.log2(n_sources)+1)  # number of original source
        y_labels = n_hot_labels(n_src)
        y_sets = y_sets_create(nums_rand, y_labels, n_src)
        logging.info('y_sets created finished')

        mkdir(path_source)
        m_pre_data_shipsear.save_datas(
            dict(zip(['X_train', 'X_val', 'X_test'], x_sets)), path_source)
        m_pre_data_shipsear.save_datas(
            dict(zip(['Y_train', 'Y_val', 'Y_test'], y_sets)), path_source, dtype=np.int32)
# ==================================================================================================
    PATH_ROOT = '/home/sqg/data/shipsEar/mix_recognition'
    RATES_SET = [0.6, 0.2, 0.2]  # rates of train, val, test set
# ---------------------------------------------------------------------------------------------------
    # for feature original sample points
    PATH_CLASS = m_pre_data_shipsear.PathSourceRoot(
        PATH_ROOT, form_src='wav', scaler_data='or', sub_set_way='rand')
      # PATH_ROOT, form_src='wav', scaler_data='mm', sub_set_way='order')

    data_create(PATH_CLASS, RATES_SET)
# ---------------------------------------------------------------------------------------------------
    WIN_LIST = [264, 528, 1056, 1582, 2110, 2638, 3164, 10547]
    HOP_LIST = [ 66, 132,  264,  396,  527,  659,  791, 10547]

    N_MELS = [512, 256, 128]

    N_MFCC = [80, 40, 20]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        path_class = m_pre_data_shipsear.PathSourceRoot(
            PATH_ROOT, form_src='magspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data='or', sub_set_way='rand')
            # scaler_data='mm', sub_set_way='order')
        data_create(path_class, RATES_SET)  # , **{'mode_read':'pytables'}

        path_class = m_pre_data_shipsear.PathSourceRoot(
            PATH_ROOT, form_src='angspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data='or', sub_set_way='rand')
            # scaler_data='mm', sub_set_way='order')
        data_create(path_class, RATES_SET)

        path_class = m_pre_data_shipsear.PathSourceRoot(
            PATH_ROOT, form_src='realspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data='or', sub_set_way='rand')
            # scaler_data='mm', sub_set_way='order')
        data_create(path_class, RATES_SET)

        path_class = m_pre_data_shipsear.PathSourceRoot(
            PATH_ROOT, form_src='imgspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data='or', sub_set_way='rand')
            # scaler_data='mm', sub_set_way='order')
        data_create(path_class, RATES_SET)

        for n_mels_i in N_MELS:
            path_class = m_pre_data_shipsear.PathSourceRoot(
                PATH_ROOT, form_src='logmelspectrum', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i,
                scaler_data='or', sub_set_way='rand')
                # scaler_data='mm', sub_set_way='order')
            data_create(path_class, RATES_SET)

            for n_mfcc_i in N_MFCC:
                path_class = m_pre_data_shipsear.PathSourceRoot(
                    PATH_ROOT, form_src='mfcc', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i, n_mfcc=n_mfcc_i,
                    scaler_data='or', sub_set_way='rand')
                    # scaler_data='mm', sub_set_way='order')
                data_create(path_class, RATES_SET)
# ---------------------------------------------------------------------------------------------------
    # Create DEMON feature.
    HIGH_LIST = [7910.1]
    LOW_LIST = [5273.4]
    CUTOFF_LIST = [1000]

    for high_i, low_i in zip(HIGH_LIST, LOW_LIST):
        for cutoff_i in CUTOFF_LIST:
            path_class = m_pre_data_shipsear.PathSourceRoot(
                PATH_ROOT, form_src='demon',
                scaler_data='or', sub_set_way='rand',
                **{'high':high_i, 'low':low_i, 'cutoff':cutoff_i})
            data_create(path_class, RATES_SET)

    logging.info('data preprocessing finished')
