# This file contains plotting functions for all AB comparison
from ..helpers.const import *
from math import ceil
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def set_ylim(bitdepth):
    if bitdepth.upper() == 'PCM24':
        ymin = PCM24_SIGNED_MIN
        ymax = PCM24_SIGNED_MAX
    elif bitdepth.upper() == 'PCM16':
        ymin = PCM16_SIGNED_MIN
        ymax = PCM16_SIGNED_MAX
    elif bitdepth.upper() == 'PCM32':
        ymin = PCM32_SIGNED_MIN
        ymax = PCM32_SIGNED_MAX
    elif bitdepth.upper() == 'PCM8':
        ymin = PCM8_SIGNED_MIN
        ymax = PCM8_SIGNED_MAX
    else:
        raise AttributeError('Unsupported bitdepth')
    return ymin, ymax


def plot(data: np.ndarray, wrap : int = 1, bitdepth : str = 'PCM24', **kwargs):
    """
    A single figure of 1 audio source with a subplot for each channel.
    TODO Add time scale option, add track title

    Parameters
    ----------
    data : numpy.ndarray
        The signal array should be in [channels, data] format
    wrap : int
        The amount of subplot per column before it will be wrap to a new row.
    bitdepth : str
        Bitdepth support PCM8, PCM16, PCM24 (default), PCM32

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object

    """
    ch = len(data)
    cols = wrap if wrap <= ch else ch
    rows = ceil(ch / cols)
    titles = [f'Ch{i+1}' for i in range(ch)]
    fig = make_subplots(rows=rows, cols=wrap, shared_xaxes=False,
                        subplot_titles=titles, **kwargs)

    for i in range(rows):
        for j in range(cols):
            ch_idx = i * cols + j
            if ch_idx < ch:
                fig.add_trace(go.Scatter(y=data[ch_idx]), row=i+1, col=j+1)

    if data.dtype == np.float32 or data.dtype == np.float64:
        ymin = -1.0
        ymax = 1.0
    else:
        ymin, ymax = set_ylim(bitdepth)
    fig.update_yaxes(range=[ymin, ymax])
    fig.update_layout(showlegend=False)

    return fig


def ABplot(a, b, speaker_setup=None, allow_mismatch=True):
    """
    A straight forward A to B sample wide and channel wide comparison plot

    Parameters
    ----------
    a
    b
    speaker_setup
    allow_mismatch

    Returns
    -------

    """
    pass
