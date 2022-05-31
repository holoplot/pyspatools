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


def plot(data: np.ndarray, wrap: int = 1, bitdepth: str = 'PCM24', **kwargs):
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
    titles = [f'Ch{i + 1}' for i in range(ch)]
    fig = make_subplots(rows=rows, cols=wrap, shared_xaxes=False,
                        subplot_titles=titles, **kwargs)

    for i in range(rows):
        for j in range(cols):
            ch_idx = i * cols + j
            if ch_idx < ch:
                fig.add_trace(go.Scatter(y=data[ch_idx]), row=i + 1, col=j + 1)

    if data.dtype == np.float32 or data.dtype == np.float64:
        ymin = -1.0
        ymax = 1.0
    else:
        ymin, ymax = set_ylim(bitdepth)
    fig.update_yaxes(range=[ymin, ymax])
    fig.update_layout(showlegend=False)

    return fig


def ABplot(a, b, a_name='A', b_name='B', bitdepth='PCM24', **kwargs):
    """
    A straight forward A to B sample wide and channel wide comparison plot

    Parameters
    ----------
    a : numpy.ndarray
        Signal A
    b : numpy.ndarray
        Signal B
    a_name : str
        Name of signal A, default is A
    b_name : str
        Name of signal B, default is B
    bitdepth : str
        Bitdepth support PCM8, PCM16, PCM24 (default), PCM32

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object

    """
    if a.dtype != b.dtype:
        raise TypeError("a and b are in different data type")
    else:
        if a.dtype == np.float32 or a.dtype == np.float64:
            ymin = -1.0
            ymax = 1.0
        else:
            ymin, ymax = set_ylim(bitdepth)
    ch = max((len(a), len(b)))
    rows = ch
    cols = 3

    diff = a - b

    row_titles = [f'Ch{i + 1}' for i in range(ch)]
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                        column_titles=(a_name, b_name, 'Diff'), row_titles=row_titles, **kwargs)
    for i in range(rows):
        fig.add_trace(go.Scatter(y=a[i]), row=i + 1, col=1)
        fig.add_trace(go.Scatter(y=b[i]), row=i + 1, col=2)
        fig.add_trace(go.Scatter(y=diff[i]), row=i + 1, col=3)

    fig.update_yaxes(range=[ymin, ymax])
    fig.update_layout(showlegend=False)
    return fig
