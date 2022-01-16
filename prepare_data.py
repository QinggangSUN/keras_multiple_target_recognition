# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:14:53 2018

@author: SUN Qinggang
E-mail: sun10qinggang@163.com

"""
from error import Error, ParameterError
import logging
import numpy as np
np.random.seed(1337)


def balancesets(sets):
    """"Cut to the min number of the element of the list.
    Args:
        sets (list[type]): a list.
    Returns:
        sets (list[type]): a list after cut.
    """
    lis = [len(si) for si in sets]
    num = min(lis)

    for i in range(len(sets)):  # pylint: disable=consider-using-enumerate
        sets[i] = sets[i][0:num]
    return sets  # dimension not changed


class Subsets(object):
    """Split datas in to train, val, test subsets."""

    def __init__(self, rates, ndata):
        self.rates = rates
        self.ndata = ndata
        self.randseq = list(range(ndata))
        np.random.shuffle(self.randseq)

    def randsubsets(self, data):
        """Input one source list of data,
        output 2D list of data[subseti][datai]"""
        if not self.ndata == len(data):
            raise Exception("self.ndata != len(data)")
        randseq = self.randseq
        subs = []
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*self.ndata)  # num of subsets[i]
            subs.append([data[randseq[j]]for j in range(numi)])
            randseq = randseq[numi:]
        subs.append([data[randseq_j] for randseq_j in randseq])
        return subs

    def ordersubsets(self, data):
        """Input one source list of data,
        output 2D list of data[subseti][datai]"""
        if not self.ndata == len(data):
            raise Exception("self.ndata != len(data)")
        subs = []
        istart = 0
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*self.ndata)  # num of subsets[i]
            subs.append(data[istart:istart+numi])
            istart += numi
        subs.append(data[istart:])
        return subs

    def randsubsetsnums(self, ndata):
        """Input num of one source,
        output 2D list of [subseti][numi]"""
        randseq = list(range(ndata))
        np.random.shuffle(randseq)
        subs = []
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*ndata)  # num of subsets[i]
            subs.append(randseq[:numi])
            randseq = randseq[numi:]
        subs.append(randseq)
        return subs

    def ordersubsetsnums(self, ndata):
        """Input num of one source, output 2D list of [subseti][numi]"""
        subs = []
        istart = 0
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*ndata)  # num of subsets[i]
            subs.append(list(range(istart, istart+numi)))
            istart += numi
        subs.append(list(range(istart, ndata)))
        return subs


def shuffle_sets(ni_3d):
    """Input 3D nums [source][subset][numi], return shuffled sets."""
    nsrc = len(ni_3d)
    nsub = len(ni_3d[0])
    nsets = []
    for subj in range(nsub):
        ni_setj = []
        for si in range(nsrc):  # pylint: disable=invalid-name
            for nk in ni_3d[si][subj]:  # pylint: disable=invalid-name
                ni_setj.append((si, nk))
        nsamples = len(ni_setj)
        randseq = list(range(nsamples))
        np.random.shuffle(randseq)
        rand_ni_setj = [ni_setj[randseq[sami]]for sami in range(nsamples)]
        nsets.append(rand_ni_setj)
    return nsets  # return 2D list [subset][(source, numi)]


def mixaddframes_np(frames):
    """Input 2D list frames [source][frames][fl],
    output 1D list of mix using average add"""
    nsrc = len(frames)  # number of sources
    if nsrc == 1:
        return np.asarray(frames[0], dtype=np.float32)

    # mix = np.sum(frames, axis=0)/np.float32(nsrc)
    mix = np.average(frames, axis=0)
    return np.asarray(mix, dtype=np.float32)


def ld3_to_ld2(ld3):
    """3D list to 2D list.
    Input: list[m][n_1...n_m][p], output: [n_1+...+n_m][p]
    """
    ld2 = []
    for ld3i in ld3:
        ld2 += ld3i
    return ld2


def nhot_3to4(nhot_3):
    """Input: np.ndarray shape (1,3)
    Return: np.ndarray shape (1, 4)."""
    nhot_4 = np.full((1, 4), 0)
    nhot_4[0, 1:] = nhot_3
    nhot_4[0, 0] = 0 if np.any(nhot_3) else 1
    return nhot_4


def filter_data(x, y, condiction='one'):
    """Filter specific_data."""
    if condiction == 'one':
        index = np.where(y == 1)[0]
        logging.debug(''.join(['filter_data index: ', str(index)]))
        x_filter = x[index]
    return x_filter
