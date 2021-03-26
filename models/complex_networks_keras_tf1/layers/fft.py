# !/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Olexa Bilaniuk
#

import keras.backend                        as KB
import keras.engine                         as KE
import keras.layers                         as KL
import keras.optimizers                     as KO
import tensorflow                           as tf
import numpy                                as np

if tf.__version__ <= '1.12.0':
    from tensorflow.spectral import fft, rfft, ifft
else:
    from tensorflow.signal import fft, rfft, ifft

#
# FFT functions:
#
#  fft():   Batched 1-D FFT  (Input: (Batch, TimeSamples))
#  ifft():  Batched 1-D IFFT (Input: (Batch, FreqSamples))
#  fft2():  Batched 2-D FFT  (Input: (Batch, TimeSamplesH, TimeSamplesW))
#  ifft2(): Batched 2-D IFFT (Input: (Batch, FreqSamplesH, FreqSamplesW))
#

def fft_func(z):
    B      = z.shape[0]//2
    L      = z.shape[1]
    C      = tf.Variable(np.asarray([[[1,-1]]], dtype=np.float32))
    Zr, Zi = rfft(z[:B]), rfft(z[B:])
    isOdd  = tf.equal(L%2, 1)
    Zr     = tf.cond(isOdd, tf.concat([Zr, C*Zr[:,1:  ][:,::-1]], axis=1),
                     tf.concat([Zr, C*Zr[:,1:-1][:,::-1]], axis=1))
    Zi     = tf.cond(isOdd, tf.concat([Zi, C*Zi[:,1:  ][:,::-1]], axis=1),
                     tf.concat([Zi, C*Zi[:,1:-1][:,::-1]], axis=1))
    Zi     = (C*Zi)[:,:,::-1]  # Zi * i
    Z      = Zr+Zi
    return tf.concat([Z[:,:,0], Z[:,:,1]], axis=0)

def ifft_func(z):
    B      = z.shape[0]//2
    real, imag = z[:B], z[B:]
    ifft_complex = ifft(tf.complex(real, imag))
    ifft_real = tf.math.real(ifft_complex)
    ifft_imag = tf.math.imag(ifft_complex)
    return tf.concat([ifft_real, ifft_imag], axis=0)

def fft2_func(x):
    tt = x
    tt = KB.reshape(tt, (x.shape[0] *x.shape[1], x.shape[2]))
    tf = fft_func(tt)
    tf = KB.reshape(tf, (x.shape[0], x.shape[1], x.shape[2]))
    tf = KB.permute_dimensions(tf, (0, 2, 1))
    tf = KB.reshape(tf, (x.shape[0] *x.shape[2], x.shape[1]))
    ff = fft_func(tf)
    ff = KB.reshape(ff, (x.shape[0], x.shape[2], x.shape[1]))
    ff = KB.permute_dimensions(ff, (0, 2, 1))
    return ff

def ifft2_func(x):
    ff = x
    ff = KB.permute_dimensions(ff, (0, 2, 1))
    ff = KB.reshape(ff, (x.shape[0] *x.shape[2], x.shape[1]))
    tf = ifft_func(ff)
    tf = KB.reshape(tf, (x.shape[0], x.shape[2], x.shape[1]))
    tf = KB.permute_dimensions(tf, (0, 2, 1))
    tf = KB.reshape(tf, (x.shape[0] *x.shape[1], x.shape[2]))
    tt = ifft_func(tf)
    tt = KB.reshape(tt, (x.shape[0], x.shape[1], x.shape[2]))
    return tt

#
# FFT Layers:
#
#  FFT:   Batched 1-D FFT  (Input: (Batch, FeatureMaps, TimeSamples))
#  IFFT:  Batched 1-D IFFT (Input: (Batch, FeatureMaps, FreqSamples))
#  FFT2:  Batched 2-D FFT  (Input: (Batch, FeatureMaps, TimeSamplesH, TimeSamplesW))
#  IFFT2: Batched 2-D IFFT (Input: (Batch, FeatureMaps, FreqSamplesH, FreqSamplesW))
#

class FFT(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1,0,2))
        a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2]))
        a = fft_func(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2]))
        return KB.permute_dimensions(a, (1,0,2))
class IFFT(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1,0,2))
        a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2]))
        a = ifft_func(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2]))
        return KB.permute_dimensions(a, (1,0,2))
class FFT2(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1,0,2,3))
        a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2], x.shape[3]))
        a = fft2_func(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
        return KB.permute_dimensions(a, (1,0,2,3))
class IFFT2(KL.Layer):
    def call(self, x, mask=None):
        a = KB.permute_dimensions(x, (1,0,2,3))
        a = KB.reshape(a, (x.shape[1] *x.shape[0], x.shape[2], x.shape[3]))
        a = ifft2_func(a)
        a = KB.reshape(a, (x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
        return KB.permute_dimensions(a, (1,0,2,3))
