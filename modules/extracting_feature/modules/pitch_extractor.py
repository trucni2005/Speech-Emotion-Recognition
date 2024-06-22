from loguru import logger
import numpy as np
import librosa

class PitchExtractor:
    def __init__(self, signal, sr, frame_size = 2048, hop_size = 512, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')):
        self.signal = signal
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.frame_size = frame_size
        self.hop_size = hop_size

    def compute_pitch(self):
        """
        Tính pitch cho tín hiệu âm thanh sử dụng phương pháp librosa.pyin.

        Returns:
        - pitches: Mảng chứa giá trị pitch cho mỗi khung.
        """
        pitches, _, _ = librosa.pyin(self.signal, fmin=self.fmin, fmax=self.fmax, sr=self.sr, hop_length=self.hop_size, frame_length=self.frame_size)
        return pitches

    def compute_pitch_statistics(self):
        """
        Tính các đặc trưng thống kê của pitch cho tín hiệu âm thanh.

        Returns:
        - Các đặc trưng thống kê của pitch bao gồm: mean, variance, max, min, median,
          25th percentile, 50th percentile, và 75th percentile.
        """
        pitches = self.compute_pitch()
        pitches = pitches[~np.isnan(pitches)]  # Loại bỏ các giá trị NaN (do không có giọng)

        if len(pitches) == 0:
            return None, None, None, None, None, None, None

        pitch_mean = np.mean(pitches)
        pitch_variance = np.var(pitches)
        pitch_max = np.max(pitches)
        pitch_min = np.min(pitches)
        pitch_median = np.median(pitches)
        pitch_25th_percentile = np.percentile(pitches, 25)
        pitch_75th_percentile = np.percentile(pitches, 75)

        return {
            'mean': pitch_mean,
            'variance': pitch_variance,
            'max': pitch_max,
            'min': pitch_min,
            'median': pitch_median,
            '25th_percentile': pitch_25th_percentile,
            '75th_percentile': pitch_75th_percentile
        }
