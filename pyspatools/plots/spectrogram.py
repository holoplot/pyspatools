from math import ceil
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = ['spectrogram', 'ABspectrogram']


def spectrogram(data: list, freqs: list, times: list, wrap: int = 1, filename="", **kwargs):
    """
    Multichannel spectrogram as a form of heatmap
    Parameters
    ----------
    data : list
        stft z data array for all channels, output from pyspatool.process.stft()
    freqs : list
        frequency data array for all channels, output from pyspatool.process.stft()
    times : list
        time data array for all channels, output from pyspatool.process.stft()
    wrap : int, optional
        The amount of subplot per column before it will be wrap to a new row.
    filename : str, optional
        Name to be printed in the plot title


    Returns
    -------

    """
    ch = len(data)
    cols = wrap if wrap <= ch else ch
    rows = ceil(ch / cols)
    titles = [f'Ch{i + 1}' for i in range(ch)]
    fig = make_subplots(rows=rows, cols=wrap, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=titles, **kwargs)

    for i in range(rows):
        for j in range(cols):
            ch_idx = i * cols + j
            if ch_idx < ch:
                z = np.abs(data[ch_idx])
                x = times[ch_idx]
                y = freqs[ch_idx]
                fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale='Viridis'),
                              row=i + 1, col=j + 1)

    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Frequency', type='log')
    fig.update_layout(title=f'Spectrogram {filename}')
    return fig


def ABspectrogram(stft_a, stft_b, freqs=None, times=None):
    """
    Not implemented yet

    Parameters
    ----------
    stft_a
    stft_b
    freqs
    times

    Returns
    -------

    """
    pass
