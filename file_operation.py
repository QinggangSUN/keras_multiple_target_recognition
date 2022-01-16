# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:27:09 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import os
import logging
import shutil
import scipy.io as sio
import numpy as np

from error import Error, ParameterError


def list_files(path, full=True):
    """Return list[str] names of the files in the path.
    Example:
        path = './testfunction'
        full_names = list_files(path)
        print(full_names)
    """

    file_names = os.listdir(path)

    full_names = []
    for file_name in file_names:
        if os.path.isfile(os.path.join(path, file_name)):
            full_names.append(os.path.join(path, file_name) if full else file_name)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort file_names changed.')
    return full_names_sorted


def list_files_filter(path, full=True, filter_func=None):
    """Return list[str] names of the files in the path."""

    if filter_func is None:
        def filter_func(x): return True

    file_names = os.listdir(path)

    full_names = []
    for file_name in file_names:
        if os.path.isfile(os.path.join(path, file_name)):
            if filter_func(file_name):
                full_names.append(os.path.join(path, file_name) if full else file_name)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort file_names changed.')
    return full_names_sorted


def list_files_end_str(path, str_end, full=True):
    """Return list[str] names of the files, end with str_end, in the path."""

    return list_files_filter(path, full, lambda file_name: file_name.endswith(str_end))


def list_dirs(path, full=True):
    """Return list[str] names of the dirs in the path.
    Example:
        path = './testfunction'
        full_names = list_dirs(path)
        print(full_names)
    """

    dir_names = os.listdir(path)

    full_names = []
    for dir_name in dir_names:
        if os.path.isdir(os.path.join(path, dir_name)):
            full_names.append(os.path.join(path, dir_name) if full else dir_name)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort dir_names changed.')
    return full_names_sorted


def list_dirs_filter(path, full=True, filter_func=None):
    """Return list[str] names of the dirs in the path."""

    if filter_func is None:
        def filter_func(x): return True

    dir_names = os.listdir(path)

    full_names = []
    for dir_name in dir_names:
        if os.path.isdir(os.path.join(path, dir_name)):
            if filter_func(dir_name):
                full_names.append(os.path.join(path, dir_name) if full else dir_name)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort dir_names changed.')
    return full_names_sorted


def list_dirs_start_str(path, str_start, full=True):
    """Return list[str] names of the dirs, start with str_start, in the path."""

    return list_dirs_filter(path, full, lambda file_name: file_name.startswith(str_start))


def walk_dirs(path, full=True):
    """Return list[str] names of the dirs multi-under path without path itself."""

    full_names = []
    for dirpath, dirnames, filenames in os.walk(path):  # pylint: disable=unused-variable
        for dirname in dirnames:
            full_names.append(os.path.join(dirpath, dirname) if full else dirname)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort dir_names changed.')
    return full_names_sorted


def walk_dirs_filter(path, full=True, filter_func=None):
    """Return list[str] names of the dirs multi-under path without path itself."""

    if filter_func is None:
        def filter_func(x): return True

    full_names = []
    for dirpath, dirnames, filenames in os.walk(path):  # pylint: disable=unused-variable
        for dirname in dirnames:
            if filter_func(dirname):
                full_names.append(os.path.join(dirpath, dirname) if full else dirname)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort dir_names changed.')
    return full_names_sorted


def walk_dirs_start_str(path, str_start, full=True):
    """Return list[str] names of the dirs, start with str_start,
    multi-under path without path itself."""

    return walk_dirs_filter(path, full, lambda dirname: dirname.startswith(str_start))


def walk_files(path, full=True):
    """Return list[str] dir and file names multi-under path without path itself."""

    full_names = []
    for dirpath, dirnames, filenames in os.walk(path):  # pylint: disable=unused-variable
        for filename in filenames:
            full_names.append(os.path.join(dirpath, filename) if full else filename)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort file_names changed.')
    return full_names_sorted


def walk_files_filter(path, full=True, filter_func=None):
    """Return list[str] dir and file names multi-under path without path itself."""

    if filter_func is None:
        def filter_func(x): return True

    full_names = []
    for dirpath, dirnames, filenames in os.walk(path):  # pylint: disable=unused-variable
        for filename in filenames:
            if filter_func(filename):
                full_names.append(os.path.join(dirpath, filename) if full else filename)

    full_names_sorted = sorted(full_names)
    if not full_names == full_names_sorted:
        logging.warning('After sort file_names changed.')
    return full_names_sorted


def walk_files_end_str(path, str_end, full=True):
    """Return list[str] dir and file names, end with str_end,
    multi-under path without path itself."""

    return walk_files_filter(path, full, lambda filename: filename.endswith(str_end))


def mkdir(path):
    """Create a directory if it does not exist.
        If father path not exist, it will be created automaticlly.
    Returns:
        True : created succeed,
        False : already exists."""

    path = path.strip()  # Remove the first space
    path = path.rstrip("\\")  # Remove the end \ symbol
    isExists = os.path.exists(path)  # Determine if the path exists
    if not isExists:  # Create a directory if it does not exist
        # if father path not exist, it will be created automaticlly
        os.makedirs(path)
        logging.debug(f'{path} created succeed')
        return True
    else:  # Do not create a directory if it exists, and prompt that it already exists
        logging.debug(f'{path} already exists')
        return False


def mycopyfile(srcfile, dstfile):
    """Copy file from srcfile to dstfile."""
    if not os.path.isfile(srcfile):
        raise Exception('Source file '+srcfile+' not exist.')
    else:
        fpath, fname = os.path.split(dstfile)  # split path and filename
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # make dir
        shutil.copyfile(srcfile, dstfile)  # copy file
        logging.debug(''.join(['copy ', srcfile, ' -> ', dstfile]))


def copy_files(srcpath, dstpath, srcnames, dstnames):
    """Copy srcnames from srcpath to dstpath and rename to dstnames."""
    for srcname, dstname in zip(srcnames, dstnames):
        if not os.path.isfile(os.path.join(srcpath, srcname)):
            raise ParameterError('Source file '+srcname+' not exist.')
        else:
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)
            shutil.copyfile(os.path.join(srcpath, srcname),
                            os.path.join(dstpath, dstname))


def read_wavs_to_np(path, sr, mono):  # pylint: disable=invalid-name
    import librosa
    """Read .wav files, return list [np.ndarray]."""
    filenames = walk_files_end_str(path, '.wav')
    sources = [librosa.load(fi, sr, mono)[0] for fi in filenames]
    return sources


def read_wavs_to_list(path, sr, mono):  # pylint: disable=invalid-name
    """Read .wav files, return 2d-list[[float]], using librosa."""
    import librosa
    filenames = walk_files_end_str(path, '.wav')
    sources = [librosa.load(fi, sr, mono)[0].tolist() for fi in filenames]
    return sources  # return 1D list [frame][fl]


def read_1d_mats_to_list(path):
    """Read .mat files, return 2d-list[[float]]."""
    filenames = walk_files_end_str(path, '.mat')
    sources = [np.array(sio.loadmat(fi)['sij'], dtype=np.float32).tolist()[0] for fi in filenames]
    return sources  # return 1D list [frame][fl]


def read_mats_to_list(path):
    """Read .wav files, return 2d-list[[float]]."""
    filenames = walk_files_end_str(path, '.mat')
    sources = [np.array(sio.loadmat(fi)['sij'], dtype=np.float32) for fi in filenames]
    return sources  # return 1D list [frame][fl]


def str_add_0(num_k, length_n):
    """Add str number num_k to length length_n by head padding 0.
    Examples:
        >>> str_add_0(123, 5)
        00123
    """
    strk = str(num_k)
    if not strk:
        raise ParameterError('Not len(strk) > 0')
    if not len(strk) < length_n+1:
        raise ParameterError('Not len(strk) < length_n+1')
    strk = '0'*(length_n-len(strk))+strk
    return strk
