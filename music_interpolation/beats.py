import typing as typ

import numpy as np
import numpy.typing as npt

from music_interpolation.madmom import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

NDFloat = npt.NDArray[np.float_]

downbeat_processor = RNNDownBeatProcessor()
downbeat_tracking_processor = DBNDownBeatTrackingProcessor(
    beats_per_bar=4,
    fps=100,
    min_bpm=100,
    max_bpm=200,
    correct=True,
)

CONFIDENCE_RANGE = 0.05


def tempo_beats_downbeats(audio: NDFloat) -> typ.Tuple[float, float, NDFloat]:
    """
    Compute the tempo, confidence, beat positions, and bar indices (1-4) of a
    single-channel (mono) audio clip with a 44100 Hz sampling rate.
    """

    activations = typ.cast(NDFloat, downbeat_processor.process(audio))
    # ndarray of shape (n, 2) holding beat positions and bar indices (1-4)
    beats_and_indices = typ.cast(NDFloat, downbeat_tracking_processor.process(activations))

    beats = beats_and_indices[:, 0]

    # Calculate the tempo as the mean of the deltas between beat positions,
    # rounded to one decimal place. This avoids the aliasing effect of using a
    # histogram
    beat_intervals = np.diff(beats)
    tempo = float(round(np.mean(60 / beat_intervals), 1))

    # Compute a confidence score based on the percentage of estimates that are
    # within a range of the mean of the beat intervals
    intervals_mean = np.round(np.mean(beat_intervals), 1)
    within_range = np.sum(
        (intervals_mean - CONFIDENCE_RANGE <= beat_intervals)
        & (beat_intervals <= intervals_mean + CONFIDENCE_RANGE)
    )
    tempo_confidence = float(within_range) / beat_intervals.size

    return tempo, tempo_confidence, beats_and_indices
