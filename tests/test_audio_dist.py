import numpy as np
import os

from audio_collage.audio_dist import AudioDist
from audio_collage.audio_segment import AudioSegment

def test_mfcc_dist():
    """
    Tests that the mfcc_dist function returns the correct distance.
    """
    a1 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    a2 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    assert AudioDist.mfcc_dist(a1, a2) == 0

    b1 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    b2 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.mfcc_dist(b1, b2) == 1

def test_fast_mfcc_dist():
    """
    Tests that the fast_mfcc_dist function returns the correct distance.
    """
    a1 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    a2 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    assert AudioDist.fast_mfcc_dist(a1, a2) == 0

    b1 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    b2 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.fast_mfcc_dist(b1, b2) == 1
    
    c1 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5]]))
    c2 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 5, 0]]))
    assert AudioDist.fast_mfcc_dist(a1, a2) == 0

    d1 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4]]))
    d2 = AudioSegment([], 1, mfcc=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.fast_mfcc_dist(b1, b2) == 1

def test_mean_mfcc_dist():
    """
    Tests that the mean_mfcc_dist function returns the correct distance.
    """
    a1 = AudioSegment([], 1, mfcc_mean=np.array([1, 2, 3, 4, 5]))
    a2 = AudioSegment([], 1, mfcc_mean=np.array([1, 2, 3, 4, 5]))
    assert AudioDist.mean_mfcc_dist(a1, a2) == 0

    b1 = AudioSegment([], 1, mfcc_mean=np.array([1, 2, 3, 4, 5]))
    b2 = AudioSegment([], 1, mfcc_mean=np.array([1, 2, 3, 4, 6]))
    assert AudioDist.mean_mfcc_dist(b1, b2) == 1

def test_chroma_dist():
    """
    Tests that the chroma_dist function returns the correct distance.
    """
    a1 = AudioSegment([], 1, chroma_stft=np.array([[1, 2, 3, 4, 5]]))
    a2 = AudioSegment([], 1, chroma_stft=np.array([[1, 2, 3, 4, 5]]))
    assert AudioDist.chroma_dist(a1, a2) == 0

    b1 = AudioSegment([], 1, chroma_stft=np.array([[1, 2, 3, 4, 5]]))
    b2 = AudioSegment([], 1, chroma_stft=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.chroma_dist(b1, b2) == 1

def test_audio_dist():
    """
    Tests that the audio_dist function returns the correct distance.
    """
    # same mfcc, same chroma
    a1 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    a2 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    assert AudioDist.audio_dist(a1, a2) == 0

    # different mfcc, same chroma
    b1 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 6]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    b2 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    assert AudioDist.audio_dist(b1, b2) == 1

    # same mfcc, different chroma
    c1 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    c2 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 7]])
    )
    assert AudioDist.audio_dist(c1, c2) == 2

    # different mfcc, different chroma
    d1 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    d2 = AudioSegment(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 6]]),
        chroma_stft=np.array([[1, 2, 3, 4, 7]])
    )
    assert AudioDist.audio_dist(d1, d2) == 3

def test_dist():
    """
    Tests that the dist function returns the correct distance.
    """
    assert AudioDist.dist(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 5]])) == 0
    assert AudioDist.dist(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 6]])) == 1
    assert AudioDist.dist(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 7]])) == 2
    assert AudioDist.dist(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 8]])) == 3
