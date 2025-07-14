import numpy as np
from audio_collage.audio_file import AudioFile

def test_audio_file_creation():
    """
    Tests that an AudioFile object is created with the correct attributes.
    """
    timeseries = np.array([1, 2, 3])
    sample_rate = 44100
    path = "/path/to/file.wav"

    audio_file = AudioFile(
        timeseries=timeseries,
        sample_rate=sample_rate,
        path=path
    )

    assert np.array_equal(audio_file.timeseries, timeseries)
    assert audio_file.sample_rate == sample_rate
    assert audio_file.path == path
    assert audio_file.offset_frames is None
    assert audio_file.mfcc is None
    assert audio_file.chroma_stft is None

def test_n_samples():
    """
    Tests that the n_samples method returns the correct number of samples.
    """
    timeseries = np.array([1, 2, 3, 4, 5])
    sample_rate = 44100

    audio_file = AudioFile(
        timeseries=timeseries,
        sample_rate=sample_rate
    )

    assert audio_file.n_samples() == 5
