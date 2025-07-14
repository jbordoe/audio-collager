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

def test_hash():
    """
    Tests that the hash method returns the correct hash.
    """
    audio_file_a = AudioFile(
        timeseries=np.array([1, 2, 3, 4, 5]),
        sample_rate=44100
    )
    audio_file_b = AudioFile(
        timeseries=audio_file_a.timeseries,
        sample_rate=12345
    )
    audio_file_c = AudioFile(
        timeseries=np.array([100,200,300]),
        sample_rate=audio_file_a.sample_rate
    )
    assert isinstance(audio_file_a.hash(), str)
    assert audio_file_a.hash() != audio_file_b.hash()
    assert audio_file_a.hash() != audio_file_c.hash()
    assert audio_file_b.hash() != audio_file_c.hash()
