from loguru import logger
import numpy as np
import librosa

class RmsExtractor:
    def __init__(self, signal, frame_size = 2048, hop_size = 512):
        self.signal = signal
        self.frame_size = frame_size
        self.hop_size = hop_size

    def compute_rms(self):
        """
        Tính RMS cho tín hiệu âm thanh.

        Returns:
        - rms: Mảng chứa giá trị RMS cho mỗi khung.
        """
        rms = librosa.feature.rms(y=self.signal, frame_length=self.frame_size, hop_length=self.hop_size)
        return np.squeeze(rms)

    def compute_rms_statistics(self):
        """
        Tính các đặc trưng thống kê của RMS cho tín hiệu âm thanh.

        Returns:
        - Các đặc trưng thống kê của RMS bao gồm: mean, variance, max, min, median,
          25th percentile, 50th percentile, và 75th percentile.
        """
        rms = self.compute_rms()
        
        rms_mean = np.mean(rms)
        rms_variance = np.var(rms)
        rms_max = np.max(rms)
        rms_min = np.min(rms)
        rms_median = np.median(rms)
        rms_25th_percentile = np.percentile(rms, 25)
        rms_50th_percentile = np.percentile(rms, 50)
        rms_75th_percentile = np.percentile(rms, 75)

        return {
            'mean': rms_mean,
            'variance': rms_variance,
            'max': rms_max,
            'min': rms_min,
            'median': rms_median,
            '25th_percentile': rms_25th_percentile,
            '50th_percentile': rms_50th_percentile,
            '75th_percentile': rms_75th_percentile
        }
