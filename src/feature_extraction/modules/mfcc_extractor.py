import numpy as np
from scipy.stats import skew, kurtosis
import librosa

class MfccExtractor:
    def __init__(self, signal, sample_rate=16000, n_mfcc=20, frame_size=2048, hop_size=512):
        self.signal = signal
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.frame_size = frame_size
        self.hop_size = hop_size
    
    def compute_mfccs(self):
        # Compute MFCCs using Discrete Cosine Transform (DCT)
        mfccs = librosa.feature.mfcc(y=self.signal, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return mfccs
    
    def compute_mfccs_statistics(self):
        mfccs = self.compute_mfccs()
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_variance = np.var(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_median = np.median(mfccs, axis=1)
        
        # Additional features
        mfcc_25th_percentile = np.percentile(mfccs, 25, axis=1)
        mfcc_75th_percentile = np.percentile(mfccs, 75, axis=1)
        mfcc_range = mfcc_max - mfcc_min
        mfcc_skewness = skew(mfccs, axis=1)
        mfcc_kurtosis = kurtosis(mfccs, axis=1)
        mfcc_energy = np.sum(np.square(mfccs), axis=1)
        
        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_variance': mfcc_variance,
            'mfcc_max': mfcc_max,
            'mfcc_min': mfcc_min,
            'mfcc_median': mfcc_median,
            'mfcc_25th_percentile': mfcc_25th_percentile,
            'mfcc_75th_percentile': mfcc_75th_percentile,
            'mfcc_range': mfcc_range,
            'mfcc_skewness': mfcc_skewness,
            'mfcc_kurtosis': mfcc_kurtosis,
            'mfcc_energy': mfcc_energy
        }