# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:56:26 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com
    First adjust numbers, split to train val test sets,
    secondly mix data s_1_1_2, s_1_1_3, s_2_2_1, s_2_2_3, s_3_3_1, s_3_3_2, s_1_1_1, s_2_2_2, s_3_3_3,
    then compute featues.
"""
import numpy as np
import os
import pickle
import itertools
import json
from prepare_data import Subsets
from prepare_data import shuffle_sets

class SubsetNums(object):
    """Class for split data sets."""

    def __init__(self, path_source_root, sub_set_way, rates, n_samples, n_sources):
        """__init__
        Args:
            path_source_root (str): path where save sources.
            sub_set_way (str) ['order', 'rand']: way to split datas.
            rates (list[double]): The rates of each sub dataset, e.g. train val test.
            n_samples (int): number of samples per source.
            n_sources (int): number of full classes.
        """
        self.path_source_root = path_source_root
        self.sub_set_way = sub_set_way
        self.rates = rates
        self.n_samples = n_samples
        self.n_sources = n_sources
        rss1 = Subsets(rates, n_samples)
        # nums: 3D list [sourcei][subseti][numi]
        if sub_set_way == 'rand':
            self.nums = [rss1.randsubsetsnums(n_samples) for i in range(n_sources)]
        elif sub_set_way == 'order':
            self.nums = [rss1.ordersubsetsnums(n_samples) for i in range(n_sources)]

    def standard_nums(self, int_n=6):
        """Standard the numbers of data in each dataset, for mix datas.
        Args:
            int_n (int): the number in each dataset % int_n is 0.
        """
        nums = self.nums
        for i, nums_srci in enumerate(nums):
            for j, nums_setj in enumerate(nums_srci):
                nums_standard = len(nums_setj) // int_n * int_n
                nums[i][j] = nums_setj[:nums_standard]
        self.nums = nums

    def save_nums_rand(self):
        """Save nums_rand."""
        nums = self.nums
        sub_set_way = self.sub_set_way
        path_source_root = self.path_source_root
        # return 2D list [subseti][(sourcei, numi)]
        self.nums_rand = shuffle_sets(nums)
        with open(os.path.join(path_source_root, f'nums_{sub_set_way}.pickle'), 'wb') as f_wb:
            pickle.dump(self.nums_rand, f_wb)
        with open(os.path.join(path_source_root, f'nums_{sub_set_way}.json'), 'w', encoding='utf-8') as f_w:
            json.dump({'data':self.nums_rand}, f_w)

def int_combinations(n_src):
    """Index of combinations of sources.
    Args:
        n_src (int): number of sources.
    Returns:
        int_src_list (list[tuple(int)]): list of combinations of sources.
    Examples:
        >>> int_combinations(4)
        [(0,), (1,), (2,), (3,),
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3),
        (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 3, 3),
        (2, 2, 2), (2, 2, 3), (2, 3, 3),
        (3, 3, 3)]
    """

    int_src = list(range(1, n_src))
    int_src_list = [(0,)]
    for i in range(1, n_src):
        int_src_list += [comb_j for comb_j in itertools.combinations_with_replacement(int_src, i)]
    return int_src_list

def labels_int(combinations_list, n_src):
    """Create labels_int.
    Args:
        combinations_list (list[tuple(int)]): list of combinations of sources.
        n_src (int): number of sources.
    Returns:
        labels_arr (np.ndarray,shape==(n_sources, n_src)): array of int labels.
    Examples:
        >>> labels_int(int_combinations(4), 4)
        [[1 0 0 0] [0 1 0 0] [0 0 1 0] [0 0 0 1]
        [0 2 0 0] [0 1 1 0] [0 1 0 1] [0 0 2 0] [0 0 1 1] [0 0 0 2]
        [0 3 0 0] [0 2 1 0] [0 2 0 1] [0 1 2 0] [0 1 1 1] [0 1 0 2]
        [0 0 3 0] [0 0 2 1] [0 0 1 2]
        [0 0 0 3]]
    """
    import logging
    labels_arr = np.zeros((len(combinations_list), n_src), dtype=int)
    logging.debug(f'labels_arr {labels_arr}')
    for i, comb_i in enumerate(combinations_list):
        for comb_ij in comb_i:
            labels_arr[i, comb_ij] += 1
    logging.debug(f'labels_arr {labels_arr}')
    return labels_arr

def labels_int_short(combinations_list, n_src):
    """Create labels_int.
    Args:
        combinations_list (list[tuple(int)]): list of combinations of sources.
        n_src (int): number of sources.
    Returns:
        labels_arr (np.ndarray,shape==(n_sources, n_src)): array of short int labels without s0.
    Examples:
        >>> labels_int_short(int_combinations(4), 4)
        [[0 0 0] [1 0 0] [0 1 0] [0 0 1]
        [2 0 0] [1 1 0] [1 0 1] [0 2 0] [0 1 1] [0 0 2]
        [3 0 0] [2 1 0] [2 0 1] [1 2 0] [1 1 1] [1 0 2]
        [0 3 0] [0 2 1] [0 1 2]
        [0 0 3]]
    """
    return labels_int(combinations_list, n_src)[:, 1:]

def chunks_n_size(lst, n):
    """Yield successive n-sized chunks from lst.
    Args:
        lst (list[type]): a list.
        n (int): split lst to n-sized chunks.
    Yields:
        (type): a chunk of lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunks_n_set(lst, n):
    """Yield successive n chunks from lst.
    Args:
        lst (list[type]): a list.
        n (int): split lst to n chunks.
    Yields:
        (type): a chunk of lst.
    """
    size_chunk = len(lst)//n
    for i in range(0, len(lst), size_chunk):
        yield lst[i:i + size_chunk]

def cycle_move_list(lst, k, right=True):
    """Cyclely move elemnets in list k steps.
    Args:
        lst (list[type]): a list.
        k (int): steps move.
        right (bool, optional): Move right direction. Defaults to True.
    Returns:
        list[type]: list after move.
    """
    if right:
        return lst[-k:]+lst[:-k]
    else:
        return lst[k:]+lst[:k]

def list_transpose(lst):
    """Transpose list first two dimensions.
    Args:
        lst (list[list[type]]): a list dimension >= 2.
    Returns:
        lst_t (list[list[type]]): transpose of the lst.
    """
    d_1 = len(lst[0])
    lst_t = [[] for _ in range(d_1)]
    for lst_i in lst:
        for j, lst_ij in enumerate(lst_i):
            lst_t[j].append(lst_ij)
    return lst_t

def list_reduce_dimension(lst):
    """Merge list first two dimensions to one dimension.
    Args:
        lst (list[list[type]]): : a list dimension >= 2.
    Returns:
        lst_sub (list[type]): lst after reduce dimension .
    """
    lst_reduce = []
    for lst_i in lst:
        lst_reduce += lst_i
    return lst_reduce

def list_split(lst, nums):
    """Split lst to len(nums) chunks, each longth in nums.
    Args:
        lst (list[type]): list to be split.
        nums (list[int]): list of size of each chunk.
    Returns:
        lst_s (list[type]): chunks of lst.
    """
    lst_s = []
    start = 0
    for num_i in nums:
        lst_s.append(lst[start:start+num_i])
        start += num_i
    return lst_s

if __name__ == '__main__':
    import logging
    from sklearn import preprocessing

    from error import ParameterError
    from feature_extract import feature_extract
    from file_operation import mkdir, mycopyfile
    from prepare_data import mixaddframes_np
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas, save_datas
    from prepare_data_shipsear_recognition_mix_s0tos3full3 import get_sr, PathSourceRootFull

    np.random.seed(1337)  # for reproducibility
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    def nums_sets_chunks(nums_subsets, n):
        """Generate nums_subsets_chunks from nums_subsets.
        Args:
            nums_subsets (list[list[list[int]]]): [set][src][nsams] index of samples for mix datas.
            n (int): split per source to n sets.
        Returns:
            nums_chunks (list[list[list[list[int]]]]): [set][src][chunks][nsams] chunks of index of samples for mix datas.
        """
        nums_chunks = []  # [set][src][chunks][samples]
        for nums_i in nums_subsets:  # set i
            nums_i_chunks = [list(chunks_n_set(nums_i_j, n)) for nums_i_j in nums_i] # src j
            nums_chunks.append(nums_i_chunks)
        return nums_chunks

    def nums_subsets_move(nums_subsets, n_src, k, m, right_move=False):
        """Split nums_subsets and cyclely move chunks.
        Args:
            nums_subsets (list[list[list[int]]]): [set][src][nsams] index of samples for mix datas.
            n_src (int): split datas to (n_src-1)*k chunks.
            k (int): split datas to (n_src-1)*k chunks.
            m (int): cyclely move chunks m steps.
        Returns:
            nums_subsets_m (list[list[list[int]]]): [set][src][nsams] index of samples for mix datas after move.
        """
        nums_chunks = nums_sets_chunks(nums_subsets, (n_src-1)*k)
        nums_chunks_m = []  # [set][src][chunks][nsams]
        for nums_i in nums_chunks:
            nums_chunks_m.append([cycle_move_list(nums_ij, m, right_move) for nums_ij in nums_i])
        nums_subsets_m = []  # [set][src][nsams]
        for nums_i in nums_chunks_m:
            nums_subsets_m.append([list_reduce_dimension(nums_ij) for nums_ij in nums_i])
        return nums_subsets_m

    def mix_nums_subset(source_frames, nums_src, comb_src):
        """Mix add one set datas.
        Args:
            source_frames (list[np.ndarray,shape==(n_sams,fl,1)]): [src] s0~s3 wav arrays.
            nums_src (list[list[int]]): [_src_][nsams] index of samples for mix datas, _src_ does not mean src.
            comb_src (tuple(int)): tuple of combinations of sources.
        Returns:
            mix_sources (list[np.ndarray,shape==feature_shape]): [nsams] list of mix datas.
        """
        mix_sources = []
        for k in range(len(nums_src[0])):  # nsams, number of samples per dataset.
            mix_frames_k = []
            for l, comb_l in enumerate(comb_src):
                mix_frames_k.append(source_frames[comb_l][nums_src[l][k]])
            mix_sources.append(mixaddframes_np(mix_frames_k))
        return mix_sources

    def mix_data(source_frames, nums_subsets, combinations_list):
        """Mix add datas.
        Args:
            source_frames (list[np.ndarray,shape==(n_sams,fl,1)]): [src] s0~s3 wav arrays.
            nums_subsets (list[list[list[int]]]): [src][set][nsams] index of samples for mix datas
            combinations_list (list[tuple(int)]): list of combinations of sources.
        Returns:
            x_mix_sources = [list[list[list[np.ndarray,shape==(fl,1)]]]]  # [n_set][src_][nsams] mix sources.
        """
        n_set = len(nums_subsets[0])
        n_src = len(source_frames)
        nums_subsets_t = list_transpose(nums_subsets)  # [set][src][nsams] transpose of nums_subsets
        nums_subsets_1_l_1 = nums_subsets_move(nums_subsets_t, n_src, 1, 1)

        x_mix_sources = [[] for _ in range(n_set)]  # [n_set][src_][nsams]
        for j, comb_source_j in enumerate(combinations_list):
            if len(set(comb_source_j)) == len(comb_source_j):  # 0, 1, 2, 3, 12, 13, 23, 123
                for i, nums_set_i in enumerate(nums_subsets_t):
                    nums_src_i = [nums_set_i[comb_l] for comb_l in comb_source_j]
                    x_mix_sources[i].append(mix_nums_subset(source_frames, nums_src_i, comb_source_j))
            elif len(set(comb_source_j)) == len(comb_source_j)-1:  # 11, 22, 33, 112, 113, 122, 133, 223, 233
                for i, nums_set_i in enumerate(nums_subsets_t):
                    nums_src_i = []
                    time_src = []
                    for comb_l in comb_source_j:
                        if comb_l in time_src:
                            nums_src_i.append(nums_subsets_1_l_1[i][comb_l])
                        else:
                            nums_src_i.append(nums_set_i[comb_l])
                        time_src.append(comb_l)
                    x_mix_sources[i].append(mix_nums_subset(source_frames, nums_src_i, comb_source_j))
            elif len(set(comb_source_j)) == len(comb_source_j)-2:  # 111, 222, 333
                for i, nums_set_i in enumerate(nums_subsets_t):
                    nums_src_i = []  # [src_][chunks][nsams]
                    time_src = dict()
                    for comb_l in comb_source_j:
                        nums_set_i_src_chunks = list(chunks_n_set(nums_set_i[comb_l], (n_src-1)*2))
                        if comb_l not in time_src.keys():
                            nums_src_i.append(nums_set_i_src_chunks)
                            time_src.update({comb_l:0})
                        else:
                            time_n = time_src[comb_l]+1
                            nums_src_i.append(cycle_move_list(nums_set_i_src_chunks, time_n*2, False))
                            time_src[comb_l] = time_n
                    for j, nums_src_i_j in enumerate(nums_src_i):
                        for k in range((n_src-1)-j, n_src-1):
                            nums_src_i[j][k*2], nums_src_i[j][k*2+1] = nums_src_i_j[k*2+1], nums_src_i_j[k*2]
                        nums_src_i[j] = list_reduce_dimension(nums_src_i[j])
                    x_mix_sources[i].append(mix_nums_subset(source_frames, nums_src_i, comb_source_j))
        return x_mix_sources

    def x_y_sets_create(x_mix_sources, combinations_list, n_src):
        """Create labels and shuffle samples and labels.
        Args:
            x_mix_sources = [list[list[list[np.ndarray,shape==(fl,1)]]]]: [n_set][src_][nsams] mix sources.
            combinations_list (list[tuple(int)]): list of combinations of sources.
            n_src (int): number of original sources.
        Returns:
            x_sets (list[np.ndarray,shape==(n_samples,fl,1)]): [n_set] data sets of inputs x.
            y_sets (list[np.ndarray],shape==(n_samples,n_src)): data sets of outputs labels y.
        """

        y_labels = labels_int_short(combinations_list, n_src)
        n_sources = len(y_labels)  # number of mix sources
        y_subsets = []  # [set][source][nsams]
        for i, x_set_i in enumerate(x_mix_sources):
            y_subset_i = []
            n_sams = len(x_set_i[0])
            for j in range(n_sources):
                y_subset_i.append([np.array(y_labels[j], dtype=np.int)]*n_sams)
            y_subsets.append(y_subset_i)

        x_sets = []
        y_sets = []
        for x_i, y_i in zip(x_mix_sources, y_subsets):
            x_set_i = list_reduce_dimension(x_i)
            y_set_i = list_reduce_dimension(y_i)
            n_set_i = len(y_set_i)
            randseq = list(range(n_set_i))
            np.random.shuffle(randseq)
            x_set_i_rand = [x_set_i[rand_j] for rand_j in randseq]
            y_set_i_rand = [y_set_i[rand_j] for rand_j in randseq]
            x_sets.append(np.asarray(x_set_i_rand, dtype=np.float32))
            y_sets.append(np.asarray(y_set_i_rand, dtype=np.int32).reshape(-1, 1, n_src-1))
        return x_sets, y_sets

    def x_sets_mm_create(x_sets):
        """Scaler data x_sets.
        Args:
            x_sets (list[np.ndarray],shape==(n_samples,fl,1)): [n_set] data sets of inputs x.
        Returns:
            x_sets_mm (list[np.ndarray,shape==(n_samples,fl,1)]): [n_set] data sets of inputs x after scaled.
        """
        x_sets_mm = []
        for x_set_i in x_sets:
            n_samples = x_set_i.shape[0]
            feature_shape = x_set_i.shape[1:]
            scaler_mm = preprocessing.MinMaxScaler()  # scaler to [0,1]
            x_set_i_mm = scaler_mm.fit_transform(x_set_i.reshape(n_samples, -1))
            x_set_i_mm = x_set_i_mm.reshape((n_samples,)+feature_shape)
            x_sets_mm.append(x_set_i_mm)
        return x_sets_mm

    def data_mixwav_create(path_class, rates_set, n_src=4, **kwargs):
        """"Create X_train, X_val, X_test scalered wav data, and Y_train, Y_val, Y_test labels.
        Args:
            path_class (object class PathSourceRootFull): object of class to compute path.
            rates_set (list[float]): rates of datasets to split.
            n_src (int, optional): number of original sources. Defaults to 4.
        """
        path_source = path_class.path_source
        path_source_root = path_class.path_source_root
        scaler_data = path_class.get_scaler_data()
        sub_set_way = path_class.sub_set_way

        dir_names = json.load(
            open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname'][:n_src]

        # Load data s_hdf5 datas.
        if 'mode_read' in kwargs.keys():
            source_frames = np.asarray(
                read_datas(
                    os.path.join(path_source_root, 's_hdf5'),
                    dir_names, **{'mode':kwargs['mode_read']}), dtype=np.float32)
        else:
            source_frames = np.asarray(
                read_datas(os.path.join(path_source_root, 's_hdf5'), dir_names), dtype=np.float32)

        if 'test_few' in kwargs.keys() and kwargs['test_few']:  # only for test few samples
            source_frames = source_frames[:, :30, :, :]

        # n_src = source_frames.shape[0]  # number of original sources
        n_samples = source_frames.shape[1]  # number of samples per mixsource

        subset_nums_class = SubsetNums(path_source_root, sub_set_way, rates_set, n_samples, n_src)
        subset_nums_class.standard_nums((n_src-1)*2)
        # subset_nums_class.save_nums_rand()
        nums_subsets = subset_nums_class.nums
        logging.debug(f'nums_subsets {nums_subsets}')

        combinations_list = int_combinations(n_src)
        x_mix_sources = mix_data(source_frames, nums_subsets, combinations_list)
        x_sets, y_sets = x_y_sets_create(x_mix_sources, combinations_list, n_src)

        if scaler_data == 'mm':
           x_sets = x_sets_mm_create(x_sets)

        mkdir(path_source)
        save_datas(dict(zip(['X_train', 'X_val', 'X_test'], x_sets)), path_source)
        save_datas(dict(zip(['Y_train', 'Y_val', 'Y_test'], y_sets)), path_source, dtype=np.int32)

    def data_feature_create(path_class_in, path_class_out, batch_save=0, **kwargs):
        """Create and save feature sources_frames.
        Args:
            path_class_in (object class PathSourceRootFull): object of class to compute path.
            path_class_out (object class PathSourceRootFull): object of class to compute path.
            batch_save (int, optional): each batch save batch_save samples. Defaults to 0 means save all samples.
        """
        path_source_in = path_class_in.path_source

        path_source_out = path_class_out.path_source
        mkdir(path_source_out)

        y_filenames = ['Y_train', 'Y_val', 'Y_test'] if 'y_filenames' not in kwargs.keys() else kwargs['y_filenames']
        y_filetype = '.hdf5' if 'y_filetype' not in kwargs.keys() else kwargs['y_filetype']
        for y_filename_i in y_filenames:
            mycopyfile(os.path.join(path_source_in, y_filename_i+y_filetype),
                        os.path.join(path_source_out, y_filename_i+y_filetype))

        x_filenames = ['X_train', 'X_val', 'X_test'] if 'x_filenames' not in kwargs.keys() else kwargs['x_filenames']
        if 'mode_read' in kwargs.keys():
            sources_wavmat = read_datas(path_source_in, x_filenames, **{'mode':kwargs['mode_read']})
        else:
            sources_wavmat = read_datas(path_source_in, x_filenames)

        def _spectrum_create(sources_wavmat, feature, win_length, hop_length, fix_length=False, window='hamming'):
            """Abstract method for create spectrum feature sources_frames."""
            import numpy as np
            source_frames = []
            for source_i in sources_wavmat:
                source_frames.append(feature_extract(
                    feature, **{
                        'source':source_i.reshape(-1, ),
                        'window':window,
                        'win_length':win_length, 'hop_length':hop_length,
                        'n_fft':win_length, 'center':False,
                        'dtype':np.complex64, 'fix_length':fix_length}))  # 2D to 3D
            return np.asarray(source_frames, dtype=np.float32)

        def magspectrum_create(sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
            """Create magnitude (amplitude) spectrum feature sources_frames."""

            return _spectrum_create(sources_wavmat, 'magspectrum', win_length, hop_length, fix_length, window)

        def angspectrum_create(sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
            """Create angle (phase) spectrum feature sources_frames."""

            return _spectrum_create(sources_wavmat, 'angspectrum', win_length, hop_length, fix_length, window)

        def realspectrum_create(sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
            """Create real part of spectrum feature sources_frames."""

            return _spectrum_create(sources_wavmat, 'realspectrum', win_length, hop_length, fix_length, window)

        def imgspectrum_create(sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
            """Create image part of spectrum feature sources_frames."""

            return _spectrum_create(sources_wavmat, 'imgspectrum', win_length, hop_length, fix_length, window)

        def logmelspectrum_create(sources, sr, n_mels, win_length=None, hop_length=None, window=None, mode=0):
            """Create Log-Mel Spectrogram feature sources_frames."""
            import numpy as np
            source_frames = []
            if mode == 0:  # input wavmat
                for source_i in sources:
                    source_frames.append(feature_extract(
                        'logmelspectrum', **{
                            'source':source_i.reshape(-1, ),
                            'sr':sr, 'n_mels':n_mels, 'window':window,
                            'win_length':win_length, 'hop_length':hop_length,
                            'n_fft':win_length, 'center':False, 'dtype':np.float32}))  # 2D to 3D
            elif mode == 1:  # input stft spectrum
                for source_i in sources:
                    source_frames.append(feature_extract(
                        'logmelspectrum', **{
                            'S':source_i.transpose()**2,
                            'sr':sr, 'n_mels':n_mels}))  # 2D to 3D
            return np.asarray(source_frames, dtype=np.float32)

        def mfcc_create(sources, sr, n_mfcc,
                        win_length=None, hop_length=None, window=None, n_mels=None, mode=0):
            """Create Log-Mel Spectrogram feature sources_frames."""
            import librosa
            import numpy as np
            source_frames = []
            if mode == 0:
                for source_i in sources:
                    source_frames.append(feature_extract(
                        'mfcc', **{
                            'source':source_i.reshape(-1, ), 'sr':sr, 'n_mfcc':n_mfcc,
                            'n_fft':win_length, 'hop_length':hop_length, 'win_length':win_length,
                            'window':window, 'center':False,
                            'n_mels':n_mels, 'dtype':np.float32
                            }))  # 2D to 3D
            elif mode == 1:  # input log-power Mel spectrogram
                for source_i in sources:
                    source_frames.append(feature_extract(
                        'mfcc', **{
                            'source':None, 'S':librosa.power_to_db(source_i.transpose()),
                            'sr':sr, 'n_mfcc':n_mfcc
                            }))  # 2D to 3D
            return np.asarray(source_frames, dtype=np.float32)

        def demon_create(sources, high=30000, low=20000, cutoff=1000.0, fs=200000, mode='square_law'):
            """Create Log-Mel Spectrogram feature sources_frames."""
            import numpy as np
            source_frames = []
            for source_i in sources:
                source_frames.append(feature_extract(
                    'demon', **{
                        'source':source_i, 'high':high, 'low':low, 'cutoff':cutoff, 'fs':fs, 'mode':mode
                        }))  # 2D to 3D
            return np.asarray(source_frames, dtype=np.float32)

        def feature_create(sources, path_class_out, form_src, **kwargs):
            if form_src == 'magspectrum':
                feature = magspectrum_create(
                    sources,
                    path_class_out.get_win_length(),
                    path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
            elif form_src == 'angspectrum':
                feature = angspectrum_create(
                    sources,
                    path_class_out.get_win_length(),
                    path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
            elif form_src == 'realspectrum':
                feature = realspectrum_create(
                    sources,
                    path_class_out.get_win_length(),
                    path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
            elif form_src == 'imgspectrum':
                feature = imgspectrum_create(
                    sources,
                    path_class_out.get_win_length(),
                    path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
            elif form_src == 'logmelspectrum':
                mode = 0 if 'mode' not in kwargs.keys() else kwargs['mode']
                if mode == 0:  # input wavmats
                    feature = logmelspectrum_create(
                        sources, kwargs['sr'], kwargs['n_mels'],
                        path_class_out.get_win_length(),
                        path_class_out.get_hop_length(), kwargs['window'], mode=mode)
                elif mode == 1:  # inpute stft spectrum
                    feature = logmelspectrum_create(
                        sources, kwargs['sr'], kwargs['n_mels'],
                        path_class_out.get_win_length(),
                        path_class_out.get_hop_length(), mode=mode)
            elif form_src == 'mfcc':
                mode = 0 if 'mode' not in kwargs.keys() else kwargs['mode']
                if mode == 0:  # input wavmats
                    feature = mfcc_create(
                        sources, kwargs['sr'], kwargs['n_mfcc'],
                        path_class_out.get_win_length(), path_class_out.get_hop_length(),
                        kwargs['window'], kwargs['n_mels'], mode=mode)
                elif mode == 1:  # input log-power Mel spectrogram
                    feature = mfcc_create(
                        sources, kwargs['sr'], kwargs['n_mfcc'], mode=mode)
            elif form_src == 'demon':
                feature = demon_create(sources, **kwargs)
            else:
                raise ParameterError('Invalid feature')
            return feature

        form_src = path_class_out.get_form_src()
        if batch_save == 0:
            # each set save a file
            for set_i, sources_wavmat_i in enumerate(sources_wavmat):
                source_frames_i = feature_create(sources_wavmat_i, path_class_out, form_src, **kwargs)
                save_datas(dict(zip([x_filenames[set_i]], [source_frames_i])), path_source_out)
        else:
            mode_batch = 'batch' if 'mode_save' not in kwargs.keys() else kwargs['mode_save']
            for set_i, sources_wavmat_i in enumerate(sources_wavmat):
                for j in range(0, sources_wavmat_i.shape[0], batch_save):
                    if j+batch_save > sources_wavmat_i.shape[0]:
                        sources_i_j = sources_wavmat_i[j:]
                    else:
                        sources_i_j = sources_wavmat_i[j:j+batch_save]

                    source_frames_i_j = feature_create(sources_i_j, path_class_out, form_src, **kwargs)
                    save_datas(dict(zip([x_filenames[set_i]], [source_frames_i_j])),
                                path_source_out, mode_batch=mode_batch)
# ==================================================================================================
    PATH_ROOT = '/home/sqg/data/shipsEar/mix_recognition'

    PATH_CLASS = PathSourceRootFull(
        PATH_ROOT, form_src='wav', scaler_data='or', sub_set_way='rand')
#        PATH_ROOT, form_src='wav', scaler_data='mm', sub_set_way='order')
# ---------------------------------------------------------------------------------------------------
    # for feature original sample points
    RATES_SET = [0.6, 0.2, 0.2]  # rates of train, val, test set
    data_mixwav_create(PATH_CLASS, RATES_SET, **{'test_few':True})
# ---------------------------------------------------------------------------------------------------
    SUB_SET_WAY = PATH_CLASS.sub_set_way
    SCALER_DATA = PATH_CLASS.get_scaler_data()
    SR = get_sr()
# ---------------------------------------------------------------------------------------------------
    WIN_LIST = [264, 528, 1056, 1582, 2110, 2638, 3164, 10547]
    HOP_LIST = [ 66, 132,  264,  396,  527,  659,  791, 10547]

    N_MELS = [512, 256, 128]

    N_MFCC = [80, 40, 20]

    for win_i, hop_i in zip(WIN_LIST, HOP_LIST):
        path_class_in = PATH_CLASS
        path_class_out = PathSourceRootFull(
            PATH_ROOT, form_src='magspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
        data_feature_create(path_class_in, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                            **{'fix_length':True, 'window':'hann'})  # , **{'mode_read':'pytables'}

        path_class_out = PathSourceRootFull(
            PATH_ROOT, form_src='angspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
        data_feature_create(path_class_in, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                            **{'fix_length':True, 'window':'hann'})  # , **{'mode_read':'pytables'}

        path_class_out = PathSourceRootFull(
            PATH_ROOT, form_src='realspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
        data_feature_create(path_class_in, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                            **{'fix_length':True, 'window':'hann'})  # , **{'mode_read':'pytables'}

        path_class_out = PathSourceRootFull(
            PATH_ROOT, form_src='imgspectrum', win_length=win_i, hop_length=hop_i,
            scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
        data_feature_create(path_class_in, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                            **{'fix_length':True, 'window':'hann'})  # , **{'mode_read':'pytables'}

        for n_mels_i in N_MELS:
            # Create logmelspectrum feature from wav.
            path_class_in = PATH_CLASS
            path_class_out = PathSourceRootFull(
                PATH_ROOT, form_src='logmelspectrum', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i,
                scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
            data_feature_create(path_class_in, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                                **{'sr':SR, 'n_mels':n_mels_i, 'window':'hann'})  # , **{'mode_read':'pytables'}

            # Create logmelspectrum feature from magspectrum.
            path_class_in = PathSourceRootFull(
                PATH_ROOT, form_src='magspectrum', win_length=win_i, hop_length=hop_i,
                scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
            path_class_out = PathSourceRootFull(
                PATH_ROOT, form_src='logmelspectrum', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i,
                scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
            data_feature_create(path_class_in, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                                **{'mode':1, 'sr':SR, 'n_mels':n_mels_i})  # , **{'mode_read':'pytables'}

            for n_mfcc_i in N_MFCC:
                # Create mfcc feature from wav.
                path_class_in = PATH_CLASS
                path_class_out = PathSourceRootFull(
                    PATH_ROOT, form_src='mfcc', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i, n_mfcc=n_mfcc_i,
                    scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
                data_feature_create(PATH_CLASS, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                                    **{'sr':SR, 'n_mfcc':n_mfcc_i,
                                        'n_mels':n_mels_i, 'window':'hann'})  # , **{'mode_read':'pytables'}

                # Create mfcc feature from logmelspectrum.
                path_class_in = PathSourceRootFull(
                    PATH_ROOT, form_src='logmelspectrum', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i,
                    scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
                path_class_out = PathSourceRootFull(
                    PATH_ROOT, form_src='mfcc', win_length=win_i, hop_length=hop_i, n_mels=n_mels_i, n_mfcc=n_mfcc_i,
                    scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
                data_feature_create(PATH_CLASS, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                                    **{'sr':SR, 'n_mfcc':n_mfcc_i,
                                        'n_mels':n_mels_i, 'window':'hann'})  # , **{'mode_read':'pytables'}
# ---------------------------------------------------------------------------------------------------
    # Create DEMON feature.
    HIGH_LIST = [7910.1]
    LOW_LIST = [5273.4]
    CUTOFF_LIST = [1000]

    for high_i, low_i in zip(HIGH_LIST, LOW_LIST):
        for cutoff_i in CUTOFF_LIST:
            path_class_out = PathSourceRootFull(
                PATH_ROOT, form_src='demon',
                scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY,
                **{'high':high_i, 'low':low_i, 'cutoff':cutoff_i})
            data_feature_create(PATH_CLASS, path_class_out, batch_save=0,  # batch_save=200, 0; 'mode_save':'batch_h5py'
                                **{'high':high_i, 'low':low_i, 'cutoff':cutoff_i,
                                    'fs':SR, 'mode':'square_law'})  # , **{'mode_read':'pytables'}

    logging.info('data preprocessing finished')
