from dataclasses import dataclass
import hashlib
import librosa
import numpy as np
import soundfile as sf

@dataclass
class AudioSegment:
    timeseries: np.ndarray
    sample_rate: int
    offset_frames: int = None
    mfcc: np.ndarray = None
    mfcc_mean: np.ndarray = None
    chroma_stft: np.ndarray = None
    path: str = None

    @staticmethod
    def from_file(path):
        timeseries, sample_rate = librosa.load(path)
        return AudioSegment(timeseries, sample_rate, path=path)

    def to_file(self, path):
        sf.write(path, self.timeseries, self.sample_rate, format='wav')

    def n_samples(self) -> int:
        return len(self.timeseries)

    def hash(self) -> str:
        """
        Returns a hash of the audio data, using timeseries and sample rate.
        """
        return hashlib.sha256(
            self.timeseries.tobytes() + str(self.sample_rate).encode()
        ).hexdigest()
        
