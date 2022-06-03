import numpy as np
import soundfile


__all__ = ['read_file']


def read_file(path: str, bitrate: int = 24, transpose: bool = True):
    """
    Read audio source. Return a list of all channels
    TODO: Add support for other bitrate audio

    Parameters
    ----------
    path : str
        Path to audio source
    bitrate : int
        Bit rate
    transpose : bool
        Whether to apply transpose to the read file signal.

    Returns
    -------
    numpy.ndarray
        A list of audio signal channels
    sr : int
        The sampling rate of the file
    """
    if bitrate==24:
        _dtype = 'int32'
    else:
        raise AttributeError('Unsupported bitrate, currently: 24')
    source, sr = soundfile.read(file=path, dtype=_dtype, always_2d=True)
    if transpose:
        source = np.transpose(np.right_shift(source, 8))
    return source, sr


def write_file(x, path, sr, subtype='PCM_24'):
    """
    Write array to file based on soundfile.write()

    Parameters
    ----------
    x : numpy.ndarray
        Signal array
    path : str
        File path with file name
    sr : int
        Sampling rate
    subtype : str
        Subtype options: FLOAT, PCM_16, PCM_24, PCM_32

    Returns
    -------
    None

    """
    soundfile.write(path, x, sr, subtype=subtype)
