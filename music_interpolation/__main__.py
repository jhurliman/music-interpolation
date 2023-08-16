import argparse

import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from audiocraft.models.musicgen import MusicGen

from music_interpolation.audio import (
    load_audio,
    resample,
    time_stretch,
    to_mono_resampled,
    trim_samples_to_match,
)
from music_interpolation.beats import tempo_beats_downbeats
from music_interpolation.encodec_interpolation import EncodecInterpolation
from music_interpolation.musicgen import generate_continuation_with_chroma

NDFloat = npt.NDArray[np.float32]
LoadedAudio = tuple[npt.NDArray[np.float32], float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.description = "Generate audio transitions between two songs"
    parser.add_argument("input_a", help="Path to first input audio file")
    parser.add_argument("input_b", help="Path to second input audio file")
    parser.add_argument("output", help="Path to output .wav audio file")
    args = parser.parse_args()

    interp = EncodecInterpolation(device="cpu")

    # Load the audio files into raw waveform numpy arrays
    print(f"Loading {args.input_a} and {args.input_b}")
    audio_a, orig_sr_a = load_audio(args.input_a, None)
    audio_b, orig_sr_b = load_audio(args.input_b, None)

    print(f"Converting {args.input_a} and {args.input_b} to mono 44.1kHz for downbeat detection")
    audio_a_mono = to_mono_resampled(audio_a, orig_sr_a, 44100)
    audio_b_mono = to_mono_resampled(audio_b, orig_sr_b, 44100)

    print(f"Performing downbeat detection on {args.input_a}")
    tempo_a, tempo_a_confidence, beats_downbeats_a = tempo_beats_downbeats(audio_a_mono)
    print(f"Performing downbeat detection on {args.input_b}")
    tempo_b, tempo_b_confidence, beats_downbeats_b = tempo_beats_downbeats(audio_b_mono)

    # Create new arrays of bar positions for each audio file. beats_downbeats_a has shape
    # (num_beats, 2) where the first column is beat positions in seconds and the second
    # column is the bar index (1-4). We want to create a new array of shape (num_bars,)
    # where each element is the beat position of the first beat in the bar
    bars_a = beats_downbeats_a[beats_downbeats_a[:, 1] == 1, 0]
    bars_b = beats_downbeats_b[beats_downbeats_b[:, 1] == 1, 0]

    print(
        f"Tempo of audio_a: {tempo_a} bpm (confidence: {tempo_a_confidence * 100:.1f}%), "
        f"{beats_downbeats_a.shape[0]} beats, {bars_a.shape[0]} bars"
    )
    print(
        f"Tempo of audio_b: {tempo_b} bpm (confidence: {tempo_b_confidence * 100:.1f}%), "
        f"{beats_downbeats_b.shape[0]} beats, {bars_b.shape[0]} bars"
    )

    # Manually resample (if needed) instead of at load time to enable the highest
    # quality resampler
    if orig_sr_a != interp.sampling_rate:
        print(f"Resampling {args.input_a} from {orig_sr_a} to {interp.sampling_rate}")
        audio_a = resample(audio_a, orig_sr_a, interp.sampling_rate)
    if orig_sr_b != interp.sampling_rate:
        print(f"Resampling {args.input_b} from {orig_sr_b} to {interp.sampling_rate}")
        audio_b = resample(audio_b, orig_sr_b, interp.sampling_rate)

    # Time stretch track_b to match the tempo of track_a
    tempo_ratio = tempo_a / tempo_b
    print(f"Time stretching audio_b by {tempo_ratio:.3f}x")
    audio_b = time_stretch(audio_b, tempo_ratio)
    bars_b = bars_b / tempo_ratio

    # Define the start and end of the interpolation in bars (1 bar = 4 beats)
    bar_start_a = 36
    bar_start_b = 12
    bar_count = 8

    # Calculate the start and end sample positions
    start_a = int(bars_a[bar_start_a] * interp.sampling_rate)
    start_b = int(bars_b[bar_start_b] * interp.sampling_rate)
    end_a = int(bars_a[bar_start_a + bar_count] * interp.sampling_rate)
    end_b = int(bars_b[bar_start_b + bar_count] * interp.sampling_rate)

    # Extract the audio for the interpolation
    audio_overlap_a = audio_a[:, start_a:end_a]
    audio_overlap_b = audio_b[:, start_b:end_b]
    audio_overlap_a, audio_overlap_b = trim_samples_to_match(audio_overlap_a, audio_overlap_b)

    # Extract up to four bars of audio from audio_a leading up to the start of the
    # interpolation
    leadup_bars = 4
    leadup_samples_a = int(bars_a[bar_start_a - leadup_bars] * interp.sampling_rate)
    leadup_a = audio_a[:, leadup_samples_a:start_a]

    # Write the extracted audio to .wav files for manual inspection
    sf.write("audio_overlap_a.wav", audio_overlap_a.T, samplerate=interp.sampling_rate)
    sf.write("audio_overlap_b.wav", audio_overlap_b.T, samplerate=interp.sampling_rate)
    sf.write("audio_leadup_a.wav", leadup_a.T, samplerate=interp.sampling_rate)

    # Interpolate the audio
    print("Interpolating audio")
    audio_c = interp.interpolate(audio_overlap_a, audio_overlap_b)
    sf.write("audio_interpolated.wav", audio_c.T, samplerate=interp.sampling_rate)

    leadup_duration = leadup_a.shape[1] / interp.sampling_rate
    overlap_duration = audio_c.shape[1] / interp.sampling_rate
    total_duration = leadup_duration + overlap_duration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MusicGen model ({device})")
    model = MusicGen.get_pretrained("melody", device)

    print(
        f"Generating {leadup_duration:.1f} + {overlap_duration:.1f} = "
        f"{total_duration:.1f} seconds of audio"
    )
    model.set_generation_params(duration=total_duration, cfg_coef=6)
    prompt = torch.Tensor(leadup_a)
    melody = torch.Tensor(audio_c)
    melody_wavs = melody[None]
    wav = generate_continuation_with_chroma(
        model, prompt, interp.sampling_rate, None, melody_wavs, interp.sampling_rate, progress=True
    )

    # Remove the leadup audio from the start of the generated audio
    # leadup_samples = int(leadup_duration * interp.sampling_rate)
    # wav = wav[:, leadup_samples:]

    generated_sec = wav.shape[-1] / model.sample_rate
    print(f"Generated {generated_sec:.1f} seconds ({wav.shape} samples). Writing {args.output}")
    sf.write(args.output, wav[0].T, samplerate=model.sample_rate)


if __name__ == "__main__":
    main()
