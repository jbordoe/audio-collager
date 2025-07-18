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

def test_trim():
    """
    Tests that the trim method returns the correct trimmed AudioSegment.
    """
    timeseries = np.array([1, 2, 3, 4, 5])
    sample_rate = 44100

    audio_segment = AudioSegment(
        timeseries=timeseries,
        sample_rate=sample_rate,
    )

    trimmed_audio_segment = audio_segment.trim(3)

    assert trimmed_audio_segment.n_samples() == 3
    assert np.array_equal(trimmed_audio_segment.timeseries, np.array([1, 2, 3]))
    # Test that the original AudioSegment is not modified
    assert np.array_equal(audio_segment.timeseries, np.array([1, 2, 3, 4, 5]))

def test_trim_inplace():
    """
    Tests that the trim_inplace method modifies the original AudioSegment.
    """
    timeseries = np.array([1, 2, 3, 4, 5])
    sample_rate = 44100

    audio_segment = AudioSegment(
        timeseries=timeseries,
        sample_rate=sample_rate,
    )

    audio_segment.trim(3, inplace=True)

    assert audio_segment.n_samples() == 3
    assert np.array_equal(audio_segment.timeseries, np.array([1, 2, 3]))

def test_pad():
    """
    Tests that the pad method returns the correct padded AudioSegment.
    """
    timeseries = np.array([1, 2, 3, 4, 5])
    sample_rate = 44100

    audio_segment = AudioSegment(
        timeseries=timeseries,
        sample_rate=sample_rate,
    )

    padded_audio_segment = audio_segment.pad(10)

    assert padded_audio_segment.n_samples() == 10
    assert np.array_equal(padded_audio_segment.timeseries, np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0]))
    # Test that the original AudioSegment is not modified
    assert np.array_equal(audio_segment.timeseries, np.array([1, 2, 3, 4, 5]))

def test_pad_inplace():
    """
    Tests that the pad_inplace method modifies the original AudioSegment.
    """
    timeseries = np.array([1, 2, 3, 4, 5])
    sample_rate = 44100

    audio_segment = AudioSegment(
        timeseries=timeseries,
        sample_rate=sample_rate,
    )

    audio_segment.pad(10, inplace=True)

    assert audio_segment.n_samples() == 10
    assert np.array_equal(audio_segment.timeseries, np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0]))

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

def test_mfcc_lazy_loading(mocker):
    """
    Tests that the MFCC features are computed lazily and cached.
    """
    mock_mfcc_computation = mocker.patch('librosa.feature.mfcc', return_value=np.array([[1,2],[3,4]]))
    segment = AudioSegment(timeseries=np.random.rand(1000), sample_rate=44100)

    assert segment._mfcc is None

    mfcc1 = segment.mfcc
    mfcc2 = segment.mfcc

    assert mfcc1 is not None
    assert mfcc2 is not None
    mock_mfcc_computation.assert_called_once()

def test_mfcc_mean_lazy_loading(mocker):
    """
    Tests that the mean of MFCCs is computed lazily and cached.
    """
    mocker.patch(
        'audio_collage.audio_segment.AudioSegment.mfcc',
        new_callable=mocker.PropertyMock,
        return_value=np.array([[1., 2., 3.], [4., 5., 6.]])
    )
    segment = AudioSegment(timeseries=np.random.rand(1000), sample_rate=44100)

    assert segment._mfcc_mean is None

    mean1 = segment.mfcc_mean
    mean2 = segment.mfcc_mean

    assert np.array_equal(mean1, np.array([2., 5.]))
    assert mean1 is mean2

def test_chroma_stft_lazy_loading(mocker):
    """
    Tests that the chroma_stft features are computed lazily and cached.
    """
    mock_chroma_stft_computation = mocker.patch('librosa.feature.chroma_stft', return_value=np.array([[1,2],[3,4]]))
    segment = AudioSegment(timeseries=np.random.rand(1000), sample_rate=44100)

    assert segment._chroma_stft is None

    chroma_stft1 = segment.chroma_stft
    chroma_stft2 = segment.chroma_stft

    assert chroma_stft1 is not None
    assert chroma_stft2 is not None
    mock_chroma_stft_computation.assert_called_once()
