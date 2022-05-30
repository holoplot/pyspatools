import numpy as np
import soundfile


def read_file(path: str, bitrate: int = 24, transpose: bool = True):
    """
    Read audio source. Return a list of all channels
    TODO: Add support for other bitrate audio
    """
    if bitrate==24:
        _dtype = 'int32'
    else:
        raise AttributeError('Unsupported bitrate, currently: 24')
    source = soundfile.SoundFile(path).read(dtype=_dtype, always_2d=True)
    if transpose:
        source = np.transpose(np.right_shift(source, 8))

    return [ch for ch in source]

