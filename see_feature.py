# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:47:48 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""


def display_feature(data, feature='wav', display=True, file_save=None, **kwargs):
    """Display and save features.

    Args:
        data (np.array): Data fo feature.
        feature (str, optional): [description]. Defaults to 'wav'.
        display (bool, optional): [description]. Defaults to True.
        file_save ([type], optional): [description]. Defaults to None.
    """

    import librosa
    import librosa.display
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if feature == 'wav':
        plt.figure()
        librosa.display.waveplot(np.squeeze(data), sr=kwargs['sr'])
        plt.title('wav')

    elif feature == 'magspectrum':
        D = librosa.amplitude_to_db(data, ref=np.max)
        librosa.display.specshow(D, sr=kwargs['sr'], hop_length=kwargs['hop_length'],
                                 x_axis='ms', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('magnitude spectrogram')

    elif feature == 'realspectrum':
        cmap = mpl.cm.gist_ncar
        bounds = list(np.arange(-10, 0, 2))+list(np.arange(0, 2, 0.25))+list(np.arange(2, 10, 2))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        librosa.display.specshow(data, sr=kwargs['sr'], hop_length=kwargs['hop_length'],
                                 x_axis='ms', y_axis='linear', cmap=cmap, norm=norm)
        plt.colorbar(extend='both', boundaries=bounds)
        plt.title('amplitude spectrogram')

    elif feature == 'imgspectrum':
        cmap = mpl.cm.gist_stern
        bounds = list(np.arange(-np.pi, np.pi, 0.1))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        librosa.display.specshow(data, sr=kwargs['sr'], hop_length=kwargs['hop_length'],
                                 x_axis='ms', y_axis='linear', cmap=cmap, norm=norm)
        plt.colorbar(extend='both', boundaries=bounds)
        plt.title('imaginary spectrogram')

    elif feature == 'logmelspectrum':
        librosa.display.specshow(data, sr=kwargs['sr'], hop_length=kwargs['hop_length'],
                                 x_axis='ms',
                                 y_axis='mel',
                                 fmax=8000,
                                 cmap=mpl.cm.coolwarm)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log Mel-frequency spectrogram')

    elif feature == 'mfcc':
        cmap = mpl.cm.viridis
        # bounds = list(range(-150, 150, 10))
        bounds = list(np.arange(-10, 25, 1))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        librosa.display.specshow(data, sr=kwargs['sr'], hop_length=kwargs['hop_length'],
                                 x_axis='ms', cmap=cmap, norm=norm)
        plt.colorbar(extend='both', boundaries=bounds)
        plt.title('MFCC')

    elif feature == 'demon':
        plt.figure()
        librosa.display.waveplot(np.squeeze(data), sr=kwargs['sr'])
        plt.title('DEMON envelope')

    plt.tight_layout()

    if file_save:
        plt.savefig(f'{file_save}.eps')
        plt.savefig(f'{file_save}_dpi600.png', dpi=600)
        plt.savefig(f'{file_save}.svg')
    if display:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from file_operation import mkdir
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_data

    def load_datas(path_data, src_names, num_data=0, transpose=True):
        """Load a data from sources of feature data files.

        Args:
            path_data (str): Path where load datas.
            src_names (list[str]): Names of mix sources.
            num_data (int, optional): Index number of the data. Defaults to 0.
            transpose (bool, optional): Wether transpose data. Defaults to True.

        Returns:
            data_list (list[np.array]): List of sources of data.
        """

        data_list = []
        for src_name in src_names:
            if transpose:
                data_list.append(read_data(path_data, src_name)[num_data].transpose())
            else:
                data_list.append(read_data(path_data, src_name)[num_data])
        return data_list

    PATH_ROOT = 'C:/data/shipsEar/multiple_class/10547_10547/s0tos3/mix_1to3'
    NUM_DATA = 4999
    SR = 52734
    SRC_NAMES = read_data(os.path.join(PATH_ROOT, 'wavmat'), 'dirname', form_src='json', dict_key='dirname')['dirname']
    PATH_SAVE_ROOT = '../result_see_feature'
    mkdir(PATH_SAVE_ROOT)

    path_feature = os.path.join(PATH_ROOT, 'wavmat', 's_hdf5')
    data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA, transpose=False)
    path_save_feature = os.path.join(PATH_SAVE_ROOT, 'wav')
    mkdir(path_save_feature)
    for data, name in zip(data_feature, SRC_NAMES):
        display_feature(data, 'wav', file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
                        **{'sr': SR})

    path_feature = os.path.join(PATH_ROOT, 'magspectrum_264_66', 's_hdf5')
    data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA)
    path_save_feature = os.path.join(PATH_SAVE_ROOT, 'magspectrum_264_66')
    mkdir(path_save_feature)
    for data, name in zip(data_feature, SRC_NAMES):
        display_feature(data, 'magspectrum',
                        file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
                        **{'sr': SR, 'hop_length': 66})

    path_feature = os.path.join(PATH_ROOT, 'realspectrum_264_66', 's_hdf5')
    data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA)
    path_save_feature = os.path.join(PATH_SAVE_ROOT, 'realspectrum_264_66')
    mkdir(path_save_feature)
    for data, name in zip(data_feature, SRC_NAMES):
        display_feature(data, 'realspectrum',
                        file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
                        **{'sr': SR, 'hop_length': 66})

    path_feature = os.path.join(PATH_ROOT, 'imgspectrum_264_66', 's_hdf5')
    data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA)
    path_save_feature = os.path.join(PATH_SAVE_ROOT, 'imgspectrum_264_66')
    mkdir(path_save_feature)
    for data, name in zip(data_feature, SRC_NAMES):
        display_feature(data, 'imgspectrum',
                        file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
                        **{'sr': SR, 'hop_length': 66})

    path_feature = os.path.join(PATH_ROOT, 'logmelspectrum_3164_791_128', 's_hdf5')
    data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA)
    path_save_feature = os.path.join(PATH_SAVE_ROOT, 'logmelspectrum_3164_791_128')
    mkdir(path_save_feature)
    for data, name in zip(data_feature, SRC_NAMES):
        display_feature(data, 'logmelspectrum',
                        file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
                        **{'sr': SR, 'hop_length': 791})

    import seaborn as sns
    path_feature = os.path.join(PATH_ROOT, 'mfcc_3164_791_512_160', 's_hdf5')
    data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA)
    path_save_feature = os.path.join(PATH_SAVE_ROOT, 'mfcc_3164_791_512_160')
    mkdir(path_save_feature)
    for data, name in zip(data_feature, SRC_NAMES):
        # data = np.squeeze(data)
        # sns.histplot(data[data > -250], kde=True)  # [data > -250]
        # plt.show()
        # plt.close()

        display_feature(data, 'mfcc',
                        file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
                        **{'sr': SR, 'hop_length': 791})

    # demon_str = 'demon_5000_3001_1000'  # 'demon_7910_5273_1000'
    # path_feature = os.path.join(PATH_ROOT, demon_str, 's_hdf5')
    # data_feature = load_datas(path_feature, SRC_NAMES, NUM_DATA)
    # path_save_feature = os.path.join(PATH_SAVE_ROOT, demon_str)
    # mkdir(path_save_feature)
    # for data, name in zip(data_feature, SRC_NAMES):
    #     display_feature(data, 'demon',
    #                     file_save=os.path.join(path_save_feature, f'{name}_{NUM_DATA}'),
    #                     **{'sr':2000})

    print('finished')
