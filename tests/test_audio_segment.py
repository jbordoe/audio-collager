import numpy as np
import os

from audio_collage.audio_segment import AudioSegment

def test_audio_segment_creation():
    """
    Tests that an AudioSegment object is created with the correct attributes.
    """
    timeseries = np.array([1, 2, 3])
    sample_rate = 44100
    path = "/path/to/file.wav"

    audio_segment = AudioSegment(
        timeseries=timeseries,
        sample_rate=sample_rate,
        path=path
    )

    assert np.array_equal(audio_segment.timeseries, timeseries)
    assert audio_segment.sample_rate == sample_rate
    assert audio_segment.path == path
    assert audio_segment.offset_frames is None
    assert audio_segment.mfcc is None
    assert audio_segment.chroma_stft is None

def test_from_file():
    """
    Tests that an AudioSegment object is created with the correct attributes.
    """
    path = "./tests/data/test.wav"
    audio_segment = AudioSegment.from_file(path)
    assert audio_segment.path == path

def test_to_file():
    """
    Tests writing an AudioSegment to a file.
    """
    path = "./tests/data/test.wav"
    out_path = "./tests/data/test_out.wav"
    audio_segment = AudioSegment.from_file(path)
    audio_segment.to_file(out_path)
    assert os.path.exists(out_path)
    os.remove(out_path)

def test_n_samples():
    """
    Tests that the n_samples method returns the correct number of samples.
    """
    timeseries = np.array([1, 2, 3, 4, 5])
    sample_rate = 44100

    audio_segment = AudioSegment(
        timeseries=timeseries,
        sample_rate=sample_rate
    )

    assert audio_segment.n_samples() == 5

def test_hash():
    """
    Tests that the hash method returns the correct hash.
    """
    audio_segment_a = AudioSegment(
        timeseries=np.array([1, 2, 3, 4, 5]),
        sample_rate=44100
    )
    audio_segment_b = AudioSegment(
        timeseries=audio_segment_a.timeseries,
        sample_rate=12345
    )
    audio_segment_c = AudioSegment(
        timeseries=np.array([100,200,300]),
        sample_rate=audio_segment_a.sample_rate
    )
    assert isinstance(audio_segment_a.hash(), str)
    assert audio_segment_a.hash() != audio_segment_b.hash()
    assert audio_segment_a.hash() != audio_segment_c.hash()
    assert audio_segment_b.hash() != audio_segment_c.hash()
