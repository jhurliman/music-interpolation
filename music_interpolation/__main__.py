import argparse
import typing as typ

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from music_interpolation.encodec_interpolation import EncodecInterpolation

NDFloat = npt.NDArray[np.float32]
LoadedAudio = tuple[npt.NDArray[np.float32], float]


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Generate audio transitions between two songs"
    parser.add_argument("input_a", help="Path to first input audio file")
    parser.add_argument("input_b", help="Path to second input audio file")
    parser.add_argument("output", help="Path to output .wav audio file")
    args = parser.parse_args()

    interp = EncodecInterpolation(device="cpu")

    # Load the audio files into raw waveform numpy arrays
    audio_a, orig_sr_a = typ.cast(LoadedAudio, librosa.load(args.input_a, sr=None, mono=False))
    audio_b, orig_sr_b = typ.cast(LoadedAudio, librosa.load(args.input_b, sr=None, mono=False))

    # Manually resample (if needed) instead of at load time to enable the highest
    # quality resampler
    if orig_sr_a != interp.sampling_rate:
        audio_a = typ.cast(
            NDFloat,
            librosa.resample(
                audio_a, orig_sr=orig_sr_a, target_sr=interp.sampling_rate, res_type="soxr_vhq"
            ),
        )
    if orig_sr_b != interp.sampling_rate:
        audio_b = typ.cast(
            NDFloat,
            librosa.resample(
                audio_b, orig_sr=orig_sr_b, target_sr=interp.sampling_rate, res_type="soxr_vhq"
            ),
        )

    # Trim to the shorter of the two audio files
    duration = min(audio_a.shape[1], audio_b.shape[1])
    if audio_a.shape[1] > duration:
        print(f"Trimming audio_a from {audio_a.shape[1]} to {duration}")
        audio_a = audio_a[:, :duration]
    elif audio_b.shape[1] > duration:
        print(f"Trimming audio_b from {audio_b.shape[1]} to {duration}")
        audio_b = audio_b[:, :duration]

    # Interpolate the audio
    audio_c = interp.interpolate(audio_a, audio_b)

    # Write the interpolated audio to the output .wav file
    sf.write(args.output, audio_c.T, samplerate=interp.sampling_rate)


if __name__ == "__main__":
    main()
