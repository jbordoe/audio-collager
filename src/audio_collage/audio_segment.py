from dataclasses import dataclass, field
import hashlib
import librosa
import numpy as np
import soundfile as sf
from typing import List, Optional

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

    def trim(
        self,
        n_samples: int,
        inplace: bool = False
    ) -> Optional["AudioSegment"]:
        """
        Trims the AudioSegment to the specified number of samples.

        Args:
            n_samples (int): The number of samples to trim to.
            inplace (bool, optional): Whether to modify the original AudioSegment or create a new one. Defaults to False.

        Returns:
            Optional[AudioSegment]: The trimmed AudioSegment, or None if inplace is True.
        """
        if inplace:
            self.timeseries = self.timeseries[:n_samples]
            return None
        else:
            return AudioSegment(
                timeseries=self.timeseries[:n_samples],
                sample_rate=self.sample_rate
            )

    def pad(
        self,
        n_samples: int,
        inplace: bool = False
    ) -> Optional["AudioSegment"]:
        """
        Pads the AudioSegment to the specified number of samples.

        Args:
            n_samples (int): The number of samples to pad to.
            inplace (bool, optional): Whether to modify the original AudioSegment or create a new one. Defaults to False.

        Returns:
            Optional[AudioSegment]: The padded AudioSegment, or None if inplace is True.
        """
        if inplace:
            self.timeseries = np.pad(self.timeseries, (0, n_samples - self.n_samples()), 'constant')
            return None
        else:
            return AudioSegment(
                timeseries=np.pad(self.timeseries, (0, n_samples - self.n_samples()), 'constant'),
                sample_rate=self.sample_rate
            )

    def split(self, n_chunks: int) -> List["AudioSegment"]:
        """
        Splits the AudioSegment into the specified number of chunks.

        Args:
            n_chunks (int): The number of chunks to split the AudioSegment into.

        Returns:
            List[AudioSegment]: A list of AudioSegments, each containing a chunk of the original AudioSegment.
        """
        if n_chunks == 1:
            return [self]
        if n_chunks > self.timeseries.size:
            raise ValueError("Cannot split AudioSegment into more chunks than samples.")

        chunk_size = int(self.n_samples() / n_chunks)
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            if i == n_chunks - 1:
                end = self.timeseries.size
            else:
                end = start + chunk_size
            ts = self.timeseries[start:end]
            chunks.append(AudioSegment(ts, self.sample_rate))
        return chunks
