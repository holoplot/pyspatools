# This file contains plotting functions for all AB comparison
from typing import Optional
from ..helpers.const import *
from math import ceil
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

__all__ = ['plot', 'ABplot']


def _set_ylim(bitdepth):
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


def plot(data: np.ndarray, wrap : int = 1, bitdepth : str = None,
         ylim=None, xlim=None, downsample=1,
         logx : bool = False, logy : bool = False, **kwargs):
    """
    A single figure of 1 audio source with a subplot for each channel.

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
    # TODO Add time scale option, add track title
    if downsample > 1:
        # This is to reduce data points for more faster plotting
        data = data[:, ::downsample]
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

    if bitdepth and not ylim:
        ymin, ymax = _set_ylim(bitdepth)
        fig.update_yaxes(range=[ymin, ymax])
    elif ylim:
        fig.update_yaxes(range=ylim)
    if xlim:
        fig.update_xaxes(range=xlim)
    if logy:
        fig.update_yaxes(type='log')
    if logx:
        fig.update_xaxes(type='log')
    fig.update_layout(showlegend=False)

    return fig


def ABplot(a, b, a_name='A', b_name='B', bitdepth='PCM24', downsample=1, single_channel=False, **kwargs):
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
    downsample : int
        If > 1, perform a simple downsample that skip amount equals to downsample. [::downsample]. This is
        to improve plotting performance.
    single_channel : bool
        Default is False, if True, instead of making a subplot for all channels, it display only a single channel
        with A and B overlaying. A dropdown menu is available to select the channel to view.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object

    """
    def _visibility_mask(i, ch):
        column_count = 3  # A, B, Diff
        result = [False] * ch * column_count
        start = i * column_count
        end = start + column_count
        result[start:end] = [True, True, True]
        return result

    if a.dtype != b.dtype:
        raise TypeError("a and b are in different data type")
    else:
        if a.dtype == np.float32 or a.dtype == np.float64:
            ymin = -1.0
            ymax = 1.0
        else:
            ymin, ymax = _set_ylim(bitdepth)

    if downsample > 1:
        # This is to reduce data points for more faster plotting
        a = a[:, ::downsample]
        b = b[:, ::downsample]

    ch = max((len(a), len(b)))
    diff = a - b

    if not single_channel:
        rows = ch
        cols = 3  # A, B, Diff
        row_titles = [f'Ch{i + 1}' for i in range(ch)]
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                            column_titles=(a_name, b_name, 'Diff'), row_titles=row_titles, **kwargs)
        for i in range(rows):
            fig.add_trace(go.Scatter(y=a[i]), row=i + 1, col=1)
            fig.add_trace(go.Scatter(y=b[i]), row=i + 1, col=2)
            fig.add_trace(go.Scatter(y=diff[i]), row=i + 1, col=3)

        fig.update_yaxes(range=[ymin, ymax])
        fig.update_layout(showlegend=False)
    else:

        # TODO Need to add diff to a separate subplot or it will not be visible if diffs are small
        # TODO Enable default display to be on the first channel
        fig = go.Figure()
        for i in range(ch):
            fig.add_trace(go.Scatter(y=a[i], visible=False, name=f'A_Ch{i+1}'))
            fig.add_trace(go.Scatter(y=b[i], visible=False, name=f'B_Ch{i+1}'))
            fig.add_trace(go.Scatter(y=diff[i], visible=False, name=f'Diff_Ch{i+1}'))

        menu_content_list = [
            dict(
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0,
                xanchor='left',
                y=1.2,
                yanchor='top'
            ),
        ]
        listener = []
        for ch_idx in range(ch):
            listener.append(dict(
                args=[{'visible': _visibility_mask(ch_idx, ch)}],
                label=f'Ch{ch_idx+1}',
                method='restyle'
            ))
        menu_content_list[0]['buttons'] = listener
        fig.update_layout(yaxis=dict(range=[ymin, ymax]), updatemenus=menu_content_list, showlegend=True)
    return fig


