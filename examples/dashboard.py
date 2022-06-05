from pyspatools.plots.base import plot, ABplot
from pyspatools.fileio import read_file
from pyspatools.process import stft, lkfs, spectrum, Diff
from pyspatools.plots import spectrogram
from pyspatools.helpers import cosine
import numpy as np
import copy
from dash import Dash, Input, Output, dcc, html

sig, sr = read_file('./data/Test51.wav')
sig2, _ = read_file('./data/Test51.wav')
# Generate a cosine signal with dropouts
cos = cosine(freq=100, channels=8, sr=44100, dtype='PCM24')
cos_reference = copy.copy(cos)
cos[3, 10000:10900] = 0
cos[4, 10000:10900] = 0
cos[4, 100000:10300] = 0

dashboard = Dash(__name__)

colors = {
    'text': '#7FDBFF'
}

dashboard.layout = html.Div([
    html.H1("5.1 Audio Signals Comparison.", style={'text-align': 'center', 'color': colors['text']}),
    dcc.Dropdown(id='signal_selector', options=[
        {'label': '5.1 Test Audio', 'value': '5.1wav'},
        {'label': 'Cosine Wave', 'value': 'cosine'}
    ], multi=False, value='5.1wav', style={'width': '50%'}),
    html.Div(id='filename_container', children=[]),
    html.Br(),
    dcc.Graph(id='signal_plot', figure={}),
    html.Div(id='positive_diff', children=[]),
    html.Div(id='positive_diff_location', children=[]),
    html.Div(id='negative_diff', children=[]),
    html.Div(id='negative_diff_location', children=[]),
])


@dashboard.callback(
    [Output(component_id='filename_container', component_property='children')],
    [Output(component_id='signal_plot', component_property='figure')],
    [Output(component_id='positive_diff', component_property='children')],
    [Output(component_id='positive_diff_location', component_property='children')],
    [Output(component_id='negative_diff', component_property='children')],
    [Output(component_id='negative_diff_location', component_property='children')],
    [Input(component_id='signal_selector', component_property='value')]
)
def update_layout(opt):
    if opt == '5.1wav':
        a = sig
        b = sig2
        fn_container_txt = "The signal selected is a 5.1 speech"
    elif opt == 'cosine':
        a = cos
        b = cos_reference
        fn_container_txt = "The signal selected is a 7.1 cosine wave"
    else:
        raise AttributeError("Unrecognized dropdown option")

    diff = Diff(a, b, sr)
    fig = ABplot(a, b, downsample=6)

    pd_txt = f"Positive Diff {diff.channel_max}"
    pdl_txt = f"Negative Diff {diff.channel_min}"
    nd_txt = f"Positive Diff Indices {diff.channel_max_indices}"
    ndl_txt = f"Negative Diff Indices {diff.channel_min_indices}"

    return fn_container_txt, fig, pd_txt, pdl_txt, nd_txt, ndl_txt


if __name__ == '__main__':
    dashboard.run_server(debug=True)
