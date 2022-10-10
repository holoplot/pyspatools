# PySpaTools - A Spatial Audio Analysis toolset in Python

On Linux, please make sure `libsndfile` is installed, e.g. `sudo apt-get install libsndfile-dev`

## Optional Feature: Playback on device

PySpaTools can also be used to play audio to audio device. To enable that we need to have pya and PyAudio installed.
Please follow this links to install the dependencies:
https://github.com/interactive-sonification/pya

Usage:

```Python
from pyspatools.playback import play, device_info

devices = device_info() # Here you can find a dict of all devices available. Pick the index
idx = 0 # Replace with the index you want, or do a name match from devices

inputs = [as1, as2, as3] # A list of AudioSignal
chs = [1, 2, 1] # A list channels that each input is supposed to be played on. For stereo, use [as1, as1], [1, 2] as inputs and chs
play(inputs, chs, device_index=idx)
```
