from pyspatools.fileio import read_file
from pyspatools.plots.base import plot, ABplot
from pyspatools.process import stft, lkfs, spectrum, Diff
from pyspatools.plots import spectrogram
from pyspatools.helpers import cosine, zero_padding
import numpy as np
import copy

sig, sr = read_file('./data/Test51.wav')
sig2, _ = read_file('./data/Test51.wav')

sig2 = sig2 + np.random.randint(low=-10, high=10, size=sig2.shape, dtype=sig2.dtype)

fig = plot(data=sig[:5], wrap=2) # fig.show() to display

# STFT
freqs, times, stfs = stft(sig, sr)

# Spectrogram
spectro_fig = spectrogram(data=stfs, freqs=freqs, times=times)


ab_plot = ABplot(sig, sig2, single_channel=True, downsample=6)

loudness = lkfs(sig, sr, bitdepth='PCM24', per_channel=False)
freqs, fft_spectrum = spectrum(sig, 0.5)

# Descriptive Diff class
diffs = Diff(sig, sig2, sr, tolerance=4)

# Generate a cosine signal with dropouts
cos = cosine(freq=100, channels=6, sr=44100, dtype='PCM24')
cos = zero_padding(cos, front=1000, back=1000)
cos_clean = copy.copy(cos)
cos[3, 10000:10900] = 0
cos[4, 10000:10900] = 0
cos[4, 100000:10300] = 0
