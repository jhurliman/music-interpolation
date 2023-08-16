"""
A wrapper module for madmom 0.16.1 that includes Python 3.10 compatibility fixes.
"""

import collections
import collections.abc
import typing as typ

import numpy as np
import numpy.typing as npt

# Python < 3.10 compatibility hacks for madmom
collections.MutableMapping = collections.abc.MutableMapping  # type: ignore
collections.MutableSequence = collections.abc.MutableSequence  # type: ignore
np.int = np.int64  # type: ignore
np.float = np.float64  # type: ignore

# flake8: noqa
from madmom.features.downbeats import RNNDownBeatProcessor  # type: ignore
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

NDFloat = npt.NDArray[np.float_]


def _process_fixed(self: typ.Any, activations: NDFloat) -> NDFloat:
    """
    Detect the (down-)beats in the given activation function.

    Parameters
    ----------
    activations : numpy array, shape (num_frames, 2)
        Activation function with probabilities corresponding to beats
        and downbeats given in the first and second column, respectively.

    Returns
    -------
    beats : numpy array, shape (num_beats, 2)
        Detected (down-)beat positions [seconds] and beat numbers.

    """
    # pylint: disable=arguments-differ
    import itertools as it

    # use only the activations > threshold (init offset to be added later)
    first = 0
    if self.threshold:
        idx = np.nonzero(activations >= self.threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))  # type: ignore
            last = min(len(activations), np.max(idx) + 1)  # type: ignore
        else:
            last = first
        activations = activations[first:last]
    # return no beats if no activations given / remain after thresholding
    if not activations.any():
        return np.empty((0, 2))
    # (parallel) decoding of the activations with HMM
    results = list(self.map(_process_dbn, zip(self.hmms, it.repeat(activations))))
    # choose the best HMM (highest log probability)
    # PATCH(jhurliman): make argmax work with our list of tuples
    # best = np.argmax(np.asarray(results)[:, 1])
    best = np.argmax([x[1] for x in results])
    # the best path through the state space
    path, _ = results[best]
    # the state space and observation model of the best HMM
    st = self.hmms[best].transition_model.state_space
    om = self.hmms[best].observation_model
    # the positions inside the pattern (0..num_beats)
    positions = st.state_positions[path]
    # corresponding beats (add 1 for natural counting)
    beat_numbers = positions.astype(int) + 1
    if self.correct:
        beats = np.empty(0, dtype=np.int64)
        # for each detection determine the "beat range", i.e. states where
        # the pointers of the observation model are >= 1
        beat_range = om.pointers[path] >= 1
        # get all change points between True and False (cast to int before)
        idx = np.nonzero(np.diff(beat_range.astype(np.int64)))[0] + 1
        # if the first frame is in the beat range, add a change at frame 0
        if beat_range[0]:
            idx = np.r_[0, idx]
        # if the last frame is in the beat range, append the length of the
        # array
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        # iterate over all regions
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                # pick the frame with the highest activations value
                # Note: we look for both beats and down-beat activations;
                #       since np.argmax works on the flattened array, we
                #       need to divide by 2
                peak = np.argmax(activations[left:right]) // 2 + left
                beats = np.hstack((beats, peak))
    else:
        # transitions are the points where the beat numbers change
        # FIXME: we might miss the first or last beat!
        #        we could calculate the interval towards the beginning/end
        #        to decide whether to include these points
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    # return the beat positions (converted to seconds) and beat numbers
    return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T


def _process_dbn(process_tuple: typ.Any):
    """
    Extract the best path through the state space in an observation sequence.

    This proxy function is necessary to process different sequences in parallel
    using the multiprocessing module.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (HMM, observations).

    Returns
    -------
    path : numpy array
        Best path through the state space.
    log_prob : float
        Log probability of the path.

    """
    # pylint: disable=no-name-in-module
    return process_tuple[0].viterbi(process_tuple[1])


# Patch the madmom DBNDownBeatTrackingProcessor.process method to fix a bug with np.argmax()
DBNDownBeatTrackingProcessor.process = _process_fixed  # type: ignore
