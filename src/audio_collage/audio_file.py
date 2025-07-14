from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass
class AudioFile:
    timeseries: np.ndarray
    sample_rate: int
    offset_frames: int = None
    mfcc: np.ndarray = None
    mfcc_mean: np.ndarray = None
    chroma_stft: np.ndarray = None
    path: str = None

    def n_samples(self) -> int:
        return len(self.timeseries)

    def hash(self) -> str:
        """
        Returns a hash of the audio data, using timeseries and sample rate.
        """
        return hashlib.sha256(
            self.timeseries.tobytes() + str(self.sample_rate).encode()
        ).hexdigest()
        
