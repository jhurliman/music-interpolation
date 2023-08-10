from typing import cast

import librosa
import numpy as np
import numpy.typing as npt

from music_interpolation.encodec_interpolation import EncodecInterpolation

AUDIO_A_PATH = "tests/data/house-equanimity-10s.mp3"
AUDIO_B_PATH = "tests/data/they-know-me-10s.mp3"

NDFloat = npt.NDArray[np.float_]


def test_interpolate():
    interp = EncodecInterpolation(device="cpu")
    assert interp.sampling_rate == 48000
    audio_a = cast(NDFloat, librosa.load(AUDIO_A_PATH, sr=interp.sampling_rate, mono=False)[0])
    audio_b = cast(NDFloat, librosa.load(AUDIO_B_PATH, sr=interp.sampling_rate, mono=False)[0])

    # Trim to 2 seconds to speed up the test
    audio_a = audio_a[:, :96000]
    audio_b = audio_b[:, :96000]
    assert audio_a.shape == (2, 96000)
    assert audio_b.shape == (2, 96000)

    audio_interp = interp.interpolate(audio_a, audio_b)
    assert audio_interp.shape == (2, 96000)

    # Write to file for manual inspection
    # import soundfile as sf
    # sf.write("interpolated.wav", audio_interp.T, samplerate=interp.sampling_rate)
