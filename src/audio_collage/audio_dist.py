import numpy as np
from numpy.linalg import norm
from dtw import accelerated_dtw as dtw

from .audio_segment import AudioSegment

class AudioDist:
    @staticmethod
    def mfcc_dist(a1: AudioSegment, a2: AudioSegment) -> float:
        return AudioDist.dist(a1.mfcc, a2.mfcc)

    @staticmethod
    def fast_mfcc_dist(a1: AudioSegment, a2: AudioSegment) -> float:
        mfcc1: np.ndarray = a1.mfcc
        mfcc2: np.ndarray = a2.mfcc

        # Pad the mfccs to the same length
        len1, len2 = mfcc1.shape[1], mfcc2.shape[1]
        if len1 > len2:
            padding = np.zeros((mfcc1.shape[0], len1 - len2))
            mfcc2 = np.hstack((mfcc2, padding))
        elif len2 > len1:
            padding = np.zeros((mfcc2.shape[0], len2 - len1))
            mfcc1 = np.hstack((mfcc1, padding))

        return norm(mfcc1 - mfcc2)

    @staticmethod
    def mean_mfcc_dist(a1: AudioSegment, a2: AudioSegment) -> float:
        return norm(a1.mfcc_mean - a2.mfcc_mean)

    @staticmethod
    def mfcc_cosine_dist(a1: AudioSegment, a2: AudioSegment) -> float:
        v1 = a1.mfcc.flatten()
        v2 = a2.mfcc.flatten()

        if len(v1) < len(v2):
            v1 = np.hstack((v1, np.zeros(len(v2) - len(v1))))
        elif len(v2) < len(v1):
            v2 = np.hstack((v2, np.zeros(len(v1) - len(v2))))

        # The cosine function from scipy computes the distance, not the similarity.
        # The distance is 1 - similarity.
        return 1 - np.dot(v1, v2) / (norm(v1) * norm(v2))

    @staticmethod
    def chroma_dist(a1: AudioSegment, a2: AudioSegment) -> float:
        return AudioDist.dist(a1.chroma_stft, a2.chroma_stft)

    # TODO: rework combination of chroma and mfcc. Also, rename the function!
    @staticmethod
    def audio_dist(a1: AudioSegment, a2: AudioSegment) -> float:
        return AudioDist.dist(a1.chroma_stft, a2.chroma_stft) + AudioDist.dist(a1.mfcc, a2.mfcc)

    @staticmethod
    def dist(mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
        distance, _cost, _acc_cost, _path = dtw(
            mfcc1.T,
            mfcc2.T,
            dist=lambda x, y: norm(x - y, ord=1)
        )
        return distance
