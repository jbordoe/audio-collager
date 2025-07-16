import numpy as np
from unittest.mock import MagicMock

from audio_collage.audio_dist import AudioDist

def test_mfcc_dist():
    """
    Tests that the mfcc_dist function returns the correct distance.
    """
    a1 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    a2 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    assert AudioDist.mfcc_dist(a1, a2) == 0

    b1 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    b2 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.mfcc_dist(b1, b2) == 1

def test_fast_mfcc_dist():
    """
    Tests that the fast_mfcc_dist function returns the correct distance.
    """
    a1 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    a2 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    assert AudioDist.fast_mfcc_dist(a1, a2) == 0

    b1 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    b2 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.fast_mfcc_dist(b1, b2) == 1
    
    c1 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    c2 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5, 0]]))
    assert AudioDist.fast_mfcc_dist(c1, c2) == 0

    d1 = MagicMock(mfcc=np.array([[1, 2, 3, 4, 5]]))
    d2 = MagicMock(mfcc=np.array([[1, 2, 3, 4]]))
    assert AudioDist.fast_mfcc_dist(d1, d2) == 5

def test_mean_mfcc_dist():
    """
    Tests that the mean_mfcc_dist function returns the correct distance.
    """
    a1 = MagicMock(mfcc_mean=np.array([1, 2, 3, 4, 5]))
    a2 = MagicMock(mfcc_mean=np.array([1, 2, 3, 4, 5]))
    assert AudioDist.mean_mfcc_dist(a1, a2) == 0

    b1 = MagicMock(mfcc_mean=np.array([1, 2, 3, 4, 5]))
    b2 = MagicMock(mfcc_mean=np.array([1, 2, 3, 4, 6]))
    assert AudioDist.mean_mfcc_dist(b1, b2) == 1

def test_chroma_dist():
    """
    Tests that the chroma_dist function returns the correct distance.
    """
    a1 = MagicMock(chroma_stft=np.array([[1, 2, 3, 4, 5]]))
    a2 = MagicMock(chroma_stft=np.array([[1, 2, 3, 4, 5]]))
    assert AudioDist.chroma_dist(a1, a2) == 0

    b1 = MagicMock(chroma_stft=np.array([[1, 2, 3, 4, 5]]))
    b2 = MagicMock(chroma_stft=np.array([[1, 2, 3, 4, 6]]))
    assert AudioDist.chroma_dist(b1, b2) == 1

def test_audio_dist():
    """
    Tests that the audio_dist function returns the correct distance.
    """
    # same mfcc, same chroma
    a1 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    a2 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=a1.mfcc,
        chroma_stft=a1.chroma_stft
    )
    assert AudioDist.audio_dist(a1, a2) == 0

    # different mfcc, same chroma
    b1 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 6]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    b2 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=b1.chroma_stft
    )
    assert AudioDist.audio_dist(b1, b2) == 1

    # same mfcc, different chroma
    c1 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    c2 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=c1.mfcc,
        chroma_stft=np.array([[1, 2, 3, 4, 7]])
    )
    assert AudioDist.audio_dist(c1, c2) == 2

    # different mfcc, different chroma
    d1 = MagicMock(
        timeseries=np.array([]),
        sample_rate=1,
        mfcc=np.array([[1, 2, 3, 4, 5]]),
        chroma_stft=np.array([[1, 2, 3, 4, 5]])
    )
    d2 = MagicMock(
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
