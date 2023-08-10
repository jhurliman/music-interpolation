# music-interpolation

> Mix between music tracks using machine learning

## Introduction

This repository is a sandbox to experiment with machine learning techniques for smoothly transitioning between music tracks.

## Development

1. `git clone https://github.com/jhurliman/music-interpolation.git`
2. `cd music-interpolation`
3. Install pipenv if you have not already: `pip install pipenv`
4. Install dependencies: `pipenv sync --python 3.10 --categories="packages cpu" --dev`

> Note: Replace `cpu` with `gpu` if you have a CUDA-enabled GPU and want to use it

5. Run tests: `pipenv run pytest`

## Usage

```python
import librosa
from IPython.display import Audio
from music_interpolation import EncodecInterpolation

# Instantiate the interpolation class, fetching and loading the pre-trained
# Encodec model
interp = EncodecInterpolation()

# Load two audio tracks, resampling to the sampling rate of the Encodec model
# (defaults to 48kHz)
audio_a, _ = librosa.load('audio_a.mp3', sr=interp.sampling_rate, mono=False)
audio_b, _ = librosa.load('audio_b.mp3', sr=interp.sampling_rate, mono=False)

# Transition between two audio tracks using linear interpolation between
# embedding vectors generated by a pre-trained Encodec model
audio_c = interp.interpolate(audio_a, audio_b)
Audio(data=audio_c, rate=interp.sampling_rate)
```

## Contributing

Contributions are welcome! Open a pull request to fix a bug, or open an issue to discuss a new feature or change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
