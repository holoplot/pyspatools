from pyspatools.plots.base import plot, ABplot
from pyspatools.fileio import read_file
from pyspatools.process import stft, lkfs, spectrum, Diff, latency
from pyspatools.plots import spectrogram
from pyspatools.helpers import cosine, zero_padding
import numpy as np
import copy
from dash import Dash, Input, Output, dcc, html

sig, sr = read_file('./data/Test51.wav')
sig2, _ = read_file('./data/Test51.wav')
# Generate a cosine signal with dropouts
cos = cosine(freq=100, channels=8, sr=sr, dtype='PCM24')
cos = zero_padding(cos, front=sr//2)

cos_reference = copy.copy(cos)
cos[3, 40000:40900] = 0
cos[4, 40000:40900] = 0
cos[4, 400000:40300] = 0

freqs_speech, times_speech, stfs_speech = stft(sig, sr)
freqs_cos, times_cos, stfs_cos = stft(cos, sr)
speech_plots = {
    'a': plot(sig, wrap=2, downsample=6),
    'ab': ABplot(sig, sig2, downsample=6),
    'ab_single': ABplot(sig, sig2, downsample=6, single_channel=True),
    'spectro': spectrogram(data=stfs_speech, freqs=freqs_speech, times=times_speech)
}

cos_plots = {
    'a': plot(cos, wrap=2),
    'ab': ABplot(cos, cos_reference, downsample=6),
    'ab_single': ABplot(cos, cos_reference, downsample=6, single_channel=True),
    'spectro': spectrogram(data=stfs_cos, freqs=freqs_cos, times=times_cos)
}

plots = (speech_plots, cos_plots)

dashboard = Dash(__name__)

colors = {
    'text': '#7FDBFF'
}

dashboard.layout = html.Div([
    html.H1("5.1 Audio Signals Comparison.", style={'text-align': 'center', 'color': colors['text']}),
    dcc.Dropdown(id='signal_selector', options=[
        {'label': '5.1 Test Audio', 'value': '5.1wav'},
        {'label': 'Cosine Wave', 'value': 'cosine'}
    ], multi=False, value='cosine', style={'width': '50%'}),
    dcc.Dropdown(id='plot_selector', options=[
        {'label': 'Waveforms', 'value': 'waveform'},
        {'label': 'Waveforms AB', 'value': 'waveform_ab'},
        {'label': 'Waveforms AB Single Channel', 'value': 'waveform_ab_1ch'},
        {'label': 'Spectrogram', 'value': 'spectro'},
    ], multi=False, value='waveform', style={'width': '50%'}),
    html.Br(),
    html.Div(id='filename_container', children=[]),
    html.Br(),
    dcc.Graph(id='signal_plot', figure={'layout': {'height': 700}}),
    html.Div(id='delta_raw', children=[]),
    html.Div(id='delta_where', children=[]),
    html.Div(id='delta_max', children=[]),
    html.Div(id='delta_min', children=[]),
    html.Br(),
    html.Div(id='latency', children=[]),
    html.Div(id='lkfs', children=[]),
])


@dashboard.callback(
    [Output(component_id='filename_container', component_property='children')],
    [Output(component_id='signal_plot', component_property='figure')],
    [Output(component_id='delta_raw', component_property='children')],
    [Output(component_id='delta_where', component_property='children')],
    [Output(component_id='delta_max', component_property='children')],
    [Output(component_id='delta_min', component_property='children')],
    [Output(component_id='latency', component_property='children')],
    [Output(component_id='lkfs', component_property='children')],
    [Input(component_id='signal_selector', component_property='value')],
    [Input(component_id='plot_selector', component_property='value')]
)
def update_layout(sig_selc_val, plt_selc_val):
    if sig_selc_val == '5.1wav':
        a = sig
        b = sig2
        fn_container_txt = "The signal selected is a 5.1 speech"
        plt_idx = 0
        latency_a = latency(a, threshold=8000, offset=0)
    else:
        a = cos
        b = cos_reference
        fn_container_txt = "The signal selected is a 7.1 cosine wave"
        plt_idx = 1
        latency_a = latency(a, threshold=0, offset=0)

    if plt_selc_val == 'waveform':
        fig = plots[plt_idx]['a']
    elif plt_selc_val == 'waveform_ab':
        fig = plots[plt_idx]['ab']
    elif plt_selc_val == 'waveform_ab_1ch':
        fig = plots[plt_idx]['ab_single']
    elif plt_selc_val == 'spectro':
        fig = plots[plt_idx]['spectro']

    diff = Diff(a, b, sr)
    _ = diff.channel_max

    loudness_a = lkfs(a, sr=sr, bitdepth='PCM24')

    diff_delta_txt = f"Raw diff {diff.delta}"
    diff_where_txt = f"Indices with diff {diff.delta}"
    diff_channel_max_txt = f"Max Diff {diff.channel_max}"
    diff_channel_min_txt = f"Min Diff {diff.channel_min}"

    latency_a_txt = f"Observed Signal latency {latency_a}"
    loudness_a_txt = f"Observed Signal LKFS {loudness_a}"


    return fn_container_txt, fig, diff_delta_txt, diff_where_txt, diff_channel_max_txt, diff_channel_min_txt, latency_a_txt, loudness_a_txt


if __name__ == '__main__':
    dashboard.run_server(debug=True)
