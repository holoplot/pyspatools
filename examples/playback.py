from pyspatools import AudioSignal
from pyspatools.playback import device_info
from pyspatools.playback import play

devices = device_info()

a = AudioSignal("./data/Test51.wav")


play([a], [1])
