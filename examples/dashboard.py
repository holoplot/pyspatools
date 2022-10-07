from dash import Dash
from dash import dcc
from dash import html
from dash import Input
from dash import Output

from pyspatools import AudioSignal
from pyspatools import Diff
from pyspatools.plots import spectrogram
from pyspatools.plots.base import ABplot
from pyspatools.plots.base import plot

sig = AudioSignal('./data/Test51.wav')

sig2 = AudioSignal('./data/Test51.wav')
sr = sig.sr

freqs_speech, times_speech, stfs_speech = sig.stft()

speech_plots = {
    'a': plot(sig, wrap=2, downsample=6),
    'ab': ABplot(sig.sig, sig2.sig, downsample=6),
    'ab_single': ABplot(sig.sig, sig2.sig, downsample=6, single_channel=True),
    'spectro': spectrogram(data=stfs_speech, freqs=freqs_speech, times=times_speech)
}

dashboard = Dash(__name__)

colors = {
    'text': '#7FDBFF'
}

dashboard.layout = html.Div([
    html.H1("PySpaTools Demo", style={'text-align': 'center', 'color': colors['text']}),
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
    [Output(component_id='lkfs', component_property='children')],
    [Input(component_id='plot_selector', component_property='value')]
)
def update_layout(plt_selc_val):
    a = sig
    b = sig2
    fn_container_txt = "The signal selected is a 5.1 speech"

    fig = speech_plots['a']
    if plt_selc_val == 'waveform_ab':
        fig = speech_plots['ab']
    elif plt_selc_val == 'waveform_ab_1ch':
        fig = speech_plots['ab_single']
    elif plt_selc_val == 'spectro':
        fig = speech_plots['spectro']

    tolerance = 0
    diff = Diff(a, b, tolerance=tolerance)
    _ = diff.channel_max

    loudness_a = a.lkfs()

    diff_delta_txt = f"Raw diff {diff.delta[diff.delta > tolerance]}"
    diff_where_txt = f"Indices with diff {diff.where}"
    diff_channel_max_txt = f"Max Diff {diff.channel_max}"
    diff_channel_min_txt = f"Min Diff {diff.channel_min}"

    # latency_a_txt = f"Observed Signal latency {latency_a}"
    loudness_a_txt = f"Observed Signal LKFS {loudness_a}"

    return fn_container_txt, fig, diff_delta_txt, diff_where_txt, diff_channel_max_txt, diff_channel_min_txt, loudness_a_txt


if __name__ == '__main__':
    dashboard.run_server(debug=True)
