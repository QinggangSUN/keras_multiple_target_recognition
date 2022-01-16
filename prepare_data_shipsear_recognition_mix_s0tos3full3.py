# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:20:10 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=no-member
import logging
import os

from prepare_data_shipsear_recognition_mix_s0tos3 import PathSourceRoot

_SR = 52734
_IS_MONO = True
_FRAME_LENGTH = 10547    # ~200ms 10546.800 000 000 001
_FRAME_SHIFT = 10547
# _FRAME_SHIFT = 2636     # 10547/4 = 2636.7


def get_sr():
    """Return const global variable _SR.
    Returns:
        _SR (int): sample rate.
    """
    return _SR


# output mix features
# _WIN_LENGTH = _FRAME_LENGTH
# _HOP_LENGTH = _FRAME_LENGTH
# _HOP_LENGTH = 2636
_WIN_LENGTH = 1582     # 52734*0.03 = 1582.02
_HOP_LENGTH = 396      # 52734*0.03/4 = 395.505
# _WIN_LENGTH = 1055     # 52734*0.02 = 1054.68
# _HOP_LENGTH = 264      # 52734*0.02/4 = 263.67


class PathSourceRootFull(PathSourceRoot):
    """Path to find sources."""

    def __init__(self, path_root, **kwargs):
        super().__init__(path_root, **kwargs)

    def _set_path_mix_root(self, value):
        """Calculate the 'path_mix_root' property.
        Args:
            value (str): set the path_mix_root.
        """
        if value:
            self._path_mix_root = value
            return
        self._path_mix_root = os.path.join(self._get_path_seg_root(), 's0tos3', 'mix_1to3full3')


if __name__ == '__main__':
    from prepare_data_shipsear_recognition_mix_s0tos3 import data_seg_create, data_mixwav_create
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    PATH_ROOT = '/home/sqg/data/shipsEar/mix_recognition'

    # Create segment datas.
    PATH_CLASS = PathSourceRootFull(PATH_ROOT)
    data_seg_create(PATH_CLASS)

    # Create original wavmat mixed sources.
    PATH_CLASS = PathSourceRootFull(PATH_ROOT, form_src='wav')
    data_mixwav_create(PATH_CLASS)

    logging.info('data_mixwav_create finished')
