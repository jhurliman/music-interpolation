import typing as typ

import librosa
import numpy as np
import numpy.typing as npt

NDFloat = npt.NDArray[np.float32]


def load_audio(path: str, sr: typ.Optional[float] = None) -> typ.Tuple[NDFloat, float]:
    return typ.cast(typ.Tuple[NDFloat, float], librosa.load(path, sr=sr, mono=False))


def resample(audio: NDFloat, orig_sr: float, target_sr: float) -> NDFloat:
    return typ.cast(
        NDFloat, librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_vhq")
    )


def time_stretch(audio: NDFloat, rate: float) -> NDFloat:
    return typ.cast(NDFloat, librosa.effects.time_stretch(audio, rate=rate))


def to_mono_resampled(audio: NDFloat, orig_sr: float, target_sr: float) -> NDFloat:
    """Convert a stereo audio array to mono and resample from the original sample rate to the
    target sample rate."""
    audio = librosa.to_mono(audio)
    if target_sr != orig_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_vhq")
    return audio


def trim_samples_to_match(audio_a: NDFloat, audio_b: NDFloat) -> tuple[NDFloat, NDFloat]:
    """Trim the longer of two audio arrays with shapes (channels, samples) to match the length of
    the shorter."""
    assert len(audio_a.shape) == 2
    assert len(audio_b.shape) == 2

    samples_a = audio_a.shape[1]
    samples_b = audio_b.shape[1]
    if samples_a > samples_b:
        audio_a = audio_a[:, :samples_b]
    elif samples_b > samples_a:
        audio_b = audio_b[:, :samples_a]
    return audio_a, audio_b
