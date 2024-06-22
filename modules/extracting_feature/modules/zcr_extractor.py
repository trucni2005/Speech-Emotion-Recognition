from loguru import logger
import os
import pandas as pd
import numpy as np
import librosa

class ZcrExtractor:
    def __init__(self, signal, frame_size = 2048, hop_size = 512):
        self.signal = signal
        self.frame_size = frame_size
        self.hop_size = hop_size

    def compute_zcr(self):
        """
        Tính Zero-Crossing Rate (ZCR) cho tín hiệu âm thanh.
        
        Args:
        - signal: Tín hiệu âm thanh đầu vào.
        - frame_size: Kích thước khung (số lượng mẫu mỗi khung).
        - hop_size: Khoảng cách giữa các khung liên tiếp (số lượng mẫu).

        Returns:
        - zcr: Mảng chứa giá trị ZCR cho mỗi khung.
        """
        zcr = librosa.feature.zero_crossing_rate(self.signal, frame_length=self.frame_size, hop_length=self.hop_size)
        return np.squeeze(zcr)

    def compute_zcr_statistics(self):
        """
        Tính các đặc trưng thống kê của Zero-Crossing Rate (ZCR) cho tín hiệu âm thanh.

        Returns:
        - Các đặc trưng thống kê của ZCR bao gồm: mean, variance, max, min, median,
          25th percentile, 75th percentile, range, skewness, kurtosis, và tổng bình phương (sum of squares).
        """
        zcr = self.compute_zcr()
        zcr_mean = np.mean(zcr)
        zcr_variance = np.var(zcr)
        zcr_max = np.max(zcr)
        zcr_min = np.min(zcr)
        zcr_median = np.median(zcr)
        zcr_25th_percentile = np.percentile(zcr, 25)
        zcr_75th_percentile = np.percentile(zcr, 75)
        return {
            'mean': zcr_mean,
            'variance': zcr_variance,
            'max': zcr_max,
            'min': zcr_min,
            'median': zcr_median,
            '25th_percentile': zcr_25th_percentile,
            '75th_percentile': zcr_75th_percentile
        }