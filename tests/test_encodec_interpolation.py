from music_interpolation.audio import load_audio
from music_interpolation.encodec_interpolation import EncodecInterpolation

AUDIO_A_PATH = "tests/data/house-equanimity.mp3"
AUDIO_B_PATH = "tests/data/they-know-me.mp3"


def test_interpolate():
    interp = EncodecInterpolation(device="cpu")
    assert interp.sampling_rate == 48000

    audio_a, _ = load_audio(AUDIO_A_PATH, interp.sampling_rate)
    audio_b, _ = load_audio(AUDIO_B_PATH, interp.sampling_rate)

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
