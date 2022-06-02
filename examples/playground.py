from pyspatools.plots.base import plot, ABplot
from pyspatools.fileio.reader import read_file
from pyspatools.process import stft, lkfs, spectrum
from pyspatools.plots import spectrogram
import numpy as np


sig, sr = read_file('./data/Test51.wav')
sig2, sr = read_file('./data/Test51.wav')

# fig = plot(data=sig[:5], wrap=2).show()


freqs, times, stfs = stft(sig, sr)


# fig = spectrogram(data=stfs, freqs=freqs, times=times)
# fig.show()



# fig = ABplot(sig, sig2, single_channel=True, downsample=6)
# fig.show()

# loudness = lkfs(sig, sr, bitdepth='PCM24', per_channel=False)

freqs, fft_spectrum = spectrum(sig, 0.5)

#TODO Add spectrum to base plot and it should work
print('stop')





