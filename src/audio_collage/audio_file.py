from dataclasses import dataclass
import numpy as np

@dataclass
class AudioFile:
    timeseries: np.ndarray
    sample_rate: int
    offset_frames: int = None
    mfcc: np.ndarray = None
    chroma_stft: np.ndarray = None
    path: str = None

    def n_samples(self):
        return len(self.timeseries)
