import copy

import numpy as np

from pyspatools import AudioSignal
from pyspatools import Diff
from pyspatools import siggen
from pyspatools.plots import spectrogram
from pyspatools.plots.base import ABplot
from pyspatools.plots.base import plot


sig= AudioSignal('./data/Test51.wav')
sig2 = AudioSignal('./data/Test51.wav')

sig2.sig = sig2.sig + np.random.rand(sig2.shape[0], sig2.shape[1]).astype(sig2.dtype)

fig = plot(x=sig.sig[:, :5], wrap=2) # fig.show() to display
fig.show()

# STFT
freqs, times, stfs = sig.stft()

# Spectrogram
spectro_fig = spectrogram(data=stfs, freqs=freqs, times=times)


ab_plot = ABplot(sig.sig, sig2.sig, single_channel=True, downsample=6)

loudness = sig.lkfs()
fft_spectrum = sig.spectrum(256)

# Descriptive Diff class
diffs = Diff(sig, sig2, tolerance=4)

# Generate a cosine signal with dropouts
cos = siggen.cos(freq=100, sr=44100)
cos.zero_padding(front=1000, back=1000)
cos_clean = copy.copy(cos)
cos.sig[10000:10900, 0] = 0
