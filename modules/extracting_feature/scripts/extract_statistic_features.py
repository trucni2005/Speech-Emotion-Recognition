from loguru import logger
import librosa
import pandas as pd
from modules.extracting_feature.modules.mfcc_extractor import MfccExtractor
from modules.extracting_feature.modules.pitch_extractor import PitchExtractor
from modules.extracting_feature.modules.rms_extractor import RmsExtractor
from modules.extracting_feature.modules.zcr_extractor import ZcrExtractor

class StatisticFeatureExtraction:
    """
    Lớp trích xuất các đặc trưng thống kê từ tín hiệu âm thanh.

    Attributes:
        n_mfcc (int): Số lượng hệ số MFCC cần trích xuất.
    """

    def __init__(self, n_mfcc=20):
        """
        Khởi tạo đối tượng StatisticFeatureExtraction.

        Args:
            n_mfcc (int): Số lượng hệ số MFCC cần trích xuất. Mặc định là 20.
        """
        self.n_mfcc = n_mfcc
    
    def _extract_mfcc_statistics(self, signal, sample_rate):
        """
        Trích xuất các đặc trưng thống kê từ hệ số MFCC.

        Args:
            signal (ndarray): Tín hiệu âm thanh.
            sample_rate (int): Tần số mẫu của tín hiệu âm thanh.

        Returns:
            dict: Từ điển chứa các đặc trưng thống kê của MFCC.
        """
        mfcc_extractor = MfccExtractor(signal, sample_rate, self.n_mfcc)
        features = mfcc_extractor.compute_mfccs_statistics()
        feature_data = {}
        for i, (mean, var, max_val, min_val, median, p25, p75, rng, skewness, kurt, energy) in enumerate(zip(
                features['mfcc_mean'],
                features['mfcc_variance'],
                features['mfcc_max'],
                features['mfcc_min'],
                features['mfcc_median'],
                features['mfcc_25th_percentile'],
                features['mfcc_75th_percentile'],
                features['mfcc_range'],
                features['mfcc_skewness'],
                features['mfcc_kurtosis'],
                features['mfcc_energy'])):
            feature_data[f'mfcc_mean_{i+1}'] = mean
            feature_data[f'mfcc_variance_{i+1}'] = var
            feature_data[f'mfcc_max_{i+1}'] = max_val
            feature_data[f'mfcc_min_{i+1}'] = min_val
            feature_data[f'mfcc_median_{i+1}'] = median
            feature_data[f'mfcc_25th_percentile_{i+1}'] = p25
            feature_data[f'mfcc_75th_percentile_{i+1}'] = p75
            feature_data[f'mfcc_range_{i+1}'] = rng
            feature_data[f'mfcc_skewness_{i+1}'] = skewness
            feature_data[f'mfcc_kurtosis_{i+1}'] = kurt
            feature_data[f'mfcc_energy_{i+1}'] = energy

        return feature_data
    
    def _extract_pitch_statistics(self, signal, sample_rate):
        """
        Trích xuất các đặc trưng thống kê từ tín hiệu pitch.

        Args:
            signal (ndarray): Tín hiệu âm thanh.
            sample_rate (int): Tần số mẫu của tín hiệu âm thanh.

        Returns:
            dict: Từ điển chứa các đặc trưng thống kê của pitch.
        """
        pitch_extractor = PitchExtractor(signal, sample_rate)
        return pitch_extractor.compute_pitch_statistics()

    def _extract_zcr_statistics(self, signal, frame_size = 2048, hop_size = 512):
        """
        Trích xuất các đặc trưng thống kê từ tín hiệu Zero Crossing Rate (ZCR).

        Args:
            signal (ndarray): Tín hiệu âm thanh.
            frame_size (int): Kích thước khung.
            hop_size (int): Kích thước bước nhảy.

        Returns:
            dict: Từ điển chứa các đặc trưng thống kê của ZCR.
        """
        zcr_extractor = ZcrExtractor(signal, frame_size, hop_size)
        return zcr_extractor.compute_zcr_statistics()

    def _extract_rms_statistics(self, signal, frame_size = 2048, hop_size = 512):
        """
        Trích xuất các đặc trưng thống kê từ tín hiệu Root Mean Square (RMS).

        Args:
            signal (ndarray): Tín hiệu âm thanh.
            frame_size (int): Kích thước khung.
            hop_size (int): Kích thước bước nhảy.

        Returns:
            dict: Từ điển chứa các đặc trưng thống kê của RMS.
        """
        rms_extractor = RmsExtractor(signal, frame_size, hop_size)
        return rms_extractor.compute_rms_statistics()
    
    def _feature_engineering_for_file(self, audio_file):
        """
        Trích xuất các đặc trưng từ một tệp âm thanh.

        Args:
            audio_file (str): Đường dẫn tới tệp âm thanh.
            target_seconds (int): Thời lượng mục tiêu cho tín hiệu âm thanh.

        Returns:
            dict: Từ điển chứa các đặc trưng trích xuất từ tệp âm thanh.
        """
        try:
            signal, sample_rate = librosa.load(audio_file, sr=None)
            if len(signal) >= 2048:
                features = {}
                
                # Đặc trưng MFCC
                mfcc_stats = self._extract_mfcc_statistics(signal, sample_rate)
                features.update({f'{key}': value for key, value in mfcc_stats.items()})

                # Đặc trưng ZCR
                zcr_stats = self._extract_zcr_statistics(signal, frame_size=2048, hop_size=512)
                features.update({f'zcr_{key}': value for key, value in zcr_stats.items()})

                # Đặc trưng Pitch
                pitch_stats = self._extract_pitch_statistics(signal, sample_rate)
                features.update({f'pitch_{key}': value for key, value in pitch_stats.items()})

                # Đặc trưng RMS
                rms_stats = self._extract_rms_statistics(signal, frame_size=2048, hop_size=512)
                features.update({f'rms_{key}': value for key, value in rms_stats.items()})

                return features
            
        except Exception as e:
            pass
        return None
    
    def process_folder(self, input_csv, output_csv):
        """
        Xử lý và trích xuất đặc trưng từ các tệp âm thanh trong một thư mục.

        Args:
            input_csv (str): Đường dẫn tới tệp CSV chứa đường dẫn tệp âm thanh và nhãn.
            output_csv (str): Đường dẫn tới tệp CSV để lưu trữ các đặc trưng trích xuất.
        """
        file_and_label_df = pd.read_csv(input_csv)
        feature_dataframes = pd.DataFrame()

        for index, row in file_and_label_df.iterrows():
            file_path = row['cleaned_file_path']
            label = row['label']
            features = self._feature_engineering_for_file(file_path)

            if index % 100 == 0:
                logger.info(f'Processed {index} file.')
                feature_dataframes.to_csv(output_csv, index=False)

            if features is not None:
                features['file_path'] = file_path
                features['label'] = label
                feature_dataframe = pd.DataFrame([features])
                feature_dataframes = pd.concat([feature_dataframes, feature_dataframe], ignore_index=True)

        feature_dataframes.to_csv(output_csv, index=False)
