from dataclasses import dataclass, field
import hashlib
import librosa
import numpy as np
import soundfile as sf
from typing import Optional

@dataclass
class AudioSegment:
    timeseries: np.ndarray
    sample_rate: int
    path: Optional[str] = None
    offset_frames: Optional[int] = None
    _mfcc: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _mfcc_mean: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _chroma_stft: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @staticmethod
    def from_file(path: str) -> "AudioSegment":
        timeseries, sample_rate = librosa.load(path)
        return AudioSegment(timeseries, sample_rate, path=path)

    def to_file(self, path: str):
        sf.write(path, self.timeseries, self.sample_rate, format='wav')

    @property
    def mfcc(self) -> np.ndarray:
        if self._mfcc is None:
            self._mfcc = librosa.feature.mfcc(
                y=self.timeseries,
                sr=self.sample_rate,
                n_fft=min(2048, len(self.timeseries))
            )
        return self._mfcc

    @property
    def mfcc_mean(self) -> np.ndarray:
        if self._mfcc_mean is None:
            self._mfcc_mean = np.mean(self.mfcc, axis=1)
        return self._mfcc_mean

    @property
    def chroma_stft(self) -> np.ndarray:
        if self._chroma_stft is None:
            self._chroma_stft = librosa.feature.chroma_stft(
                y=self.timeseries,
                sr=self.sample_rate,
                hop_length=self.sample_rate // 100,
                n_fft=self.sample_rate // 2
            )
        return self._chroma_stft
        
    def n_samples(self) -> int:
        return len(self.timeseries)

    def hash(self) -> str:
        """
        Returns a hash of the audio data, using timeseries and sample rate.
        """
        return hashlib.sha256(
            self.timeseries.tobytes() + str(self.sample_rate).encode()
        ).hexdigest()
