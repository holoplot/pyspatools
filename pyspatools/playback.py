from typing import Optional
from warnings import warn


def pya_warning():
    ms1 = "pya is not installed. Can't play audio."
    ms2 = "Please follow instruction in https://github.com/interactive-sonification/pya to install "
    warn(ms1 + ms2)


def device_info():
    """
    Return a dictionary of device info detected by PyAudio
    """
    try:
        from pya import find_device

        return find_device()
    except ImportError:
        pya_warning()


def play(
    inputs: list,
    channels: list,
    sr: int = 48000,
    bs: int = 256,
    block: bool = True,
    num_channels: Optional[int] = None,
    device_index: Optional[int] = None,
):
    """
    Take a list of AudioSignal and a list of integer for the channel (1 base) of each AudioSignal is supposed to be played on,
    play it through device using pya's Aserver context manager.
    """
    try:
        from pya import Aserver
        from pya import Asig

        max_channel = num_channels or max(channels)
        with Aserver(
            sr=sr, bs=bs, device=device_index, channels=max_channel
        ) as input_audio_server:
            for item, chan in zip(inputs, channels):
                asig = Asig(item.sig, item.sr)
                asig.play(server=input_audio_server, channel=chan - 1, block=block)

    except ImportError:
        pya_warning()
