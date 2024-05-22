from loguru import logger
import os
import pandas as pd
import numpy as np
import librosa

# Configure loguru logger
logger.add("feature_extraction.log", rotation="10 MB", retention="10 days", level="INFO")

class FeatureExtractor:
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc

    def compute_mfccs(self, signal, sample_rate):
        """
        Tính toán Mel-frequency cepstral coefficients (MFCCs) của tín hiệu âm thanh.
        """
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=self.n_mfcc)
        return np.mean(mfccs, axis=1)
    
    def compute_mfccs_statistics(self, signal, sample_rate):
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_variance = np.var(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_median = np.median(mfccs, axis=1)
        return mfcc_mean, mfcc_variance, mfcc_max, mfcc_min, mfcc_median

    def compute_zcr(self, signal):
        """
        Tính toán Zero Crossing Rate (ZCR) của tín hiệu âm thanh.
        """
        zcr = librosa.feature.zero_crossing_rate(signal)[0]
        zcr_mean = np.mean(zcr)
        zcr_variance = np.var(zcr)
        zcr_max = np.max(zcr)
        zcr_min = np.min(zcr)
        zcr_median = np.median(zcr)
        return zcr_mean, zcr_variance, zcr_max, zcr_min, zcr_median

    def extract_pitch(self, signal, sample_rate):
        pitch, _ = librosa.core.piptrack(y=signal, sr=sample_rate)
        mean_pitch = pitch.mean()
        return mean_pitch

    def extract_rms_features(self, y, frame_length=512, hop_length=256):
        """Trích xuất RMS energy và tính các giá trị thống kê."""
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_mean = np.mean(rms)
        rms_variance = np.var(rms)
        rms_max = np.max(rms)
        rms_min = np.min(rms)
        rms_median = np.median(rms)
        return rms_mean, rms_variance, rms_max, rms_min, rms_median

    def feature_engineering_for_file(self, audio_file, selected_features):
        """
        Trích xuất các đặc trưng từ một tệp âm thanh dựa trên các đặc trưng đã chọn.
        """
        try:
            signal, sample_rate = librosa.load(audio_file, sr=None)
            if len(signal) >= 2048:
                features = {}
                if 'mfcc' in selected_features:
                    features['mfcc'] = self.compute_mfccs(signal, sample_rate)
                if 'mfcc_statistics' in selected_features:
                    mfcc_mean, mfcc_variance, mfcc_max, mfcc_min, mfcc_median = self.compute_mfccs_statistics(signal, sample_rate)
                    features['mfcc_mean'] = mfcc_mean
                    features['mfcc_variance'] = mfcc_variance
                    features['mfcc_max'] = mfcc_max
                    features['mfcc_min'] = mfcc_min
                    features['mfcc_median'] = mfcc_median
                if 'zcr' in selected_features:
                    zcr_mean, zcr_variance, zcr_max, zcr_min, zcr_median = self.compute_zcr(signal)
                    features['zcr_mean'] = zcr_mean
                    features['zcr_variance'] = zcr_variance
                    features['zcr_max'] = zcr_max
                    features['zcr_min'] = zcr_min
                    features['zcr_median'] = zcr_median
                if 'pitch' in selected_features:
                    features['pitch'] = self.extract_pitch(signal, sample_rate)
                if 'rms' in selected_features:
                    rms_mean, rms_variance, rms_max, rms_min, rms_median = self.extract_rms_features(signal)
                    features['rms_mean'] = rms_mean
                    features['rms_variance'] = rms_variance
                    features['rms_max'] = rms_max
                    features['rms_min'] = rms_min
                    features['rms_median'] = rms_median
                return features
        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {e}")
        return None

    def process_file(self, label, input_folder_path, selected_features):
        """
        Xử lý tất cả các tệp WAV trong thư mục để trích xuất các đặc trưng dựa trên các đặc trưng đã chọn.
        """
        mfcc_data = pd.DataFrame()

        for file_name in os.listdir(input_folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(input_folder_path, file_name)

                features = self.feature_engineering_for_file(file_path, selected_features)
                if features is not None:
                    feature_data = {'label': label, 'file': file_name}
                    if 'mfcc' in selected_features and 'mfcc' in features:
                        mfcc_array = features['mfcc']
                        for i, mfcc in enumerate(mfcc_array):
                            feature_data[f'mfcc_{i+1}'] = mfcc
                    if 'mfcc_statistics' in selected_features and 'mfcc_mean' in features:
                        for i, (mean, var, max_val, min_val, median) in enumerate(zip(
                            features['mfcc_mean'], 
                            features['mfcc_variance'], 
                            features['mfcc_max'], 
                            features['mfcc_min'], 
                            features['mfcc_median'])):
                            feature_data[f'mfcc_mean_{i+1}'] = mean
                            feature_data[f'mfcc_variance_{i+1}'] = var
                            feature_data[f'mfcc_max_{i+1}'] = max_val
                            feature_data[f'mfcc_min_{i+1}'] = min_val
                            feature_data[f'mfcc_median_{i+1}'] = median
                    if 'zcr' in selected_features and 'zcr_mean' in features:
                        feature_data['zcr_mean'] = features['zcr_mean']
                        feature_data['zcr_variance'] = features['zcr_variance']
                        feature_data['zcr_max'] = features['zcr_max']
                        feature_data['zcr_min'] = features['zcr_min']
                        feature_data['zcr_median'] = features['zcr_median']
                    if 'pitch' in selected_features and 'pitch' in features:
                        feature_data['pitch'] = features['pitch']
                    if 'rms' in selected_features and 'rms_mean' in features:
                        feature_data['rms_mean'] = features['rms_mean']
                        feature_data['rms_variance'] = features['rms_variance']
                        feature_data['rms_max'] = features['rms_max']
                        feature_data['rms_min'] = features['rms_min']
                        feature_data['rms_median'] = features['rms_median']

                    mfcc_df = pd.DataFrame([feature_data])
                    mfcc_data = pd.concat([mfcc_data, mfcc_df], ignore_index=True)
        
        return mfcc_data

    def process_folder(self, input_folder, output_csv_path, selected_features):
        """
        Xử lý tất cả các thư mục con trong thư mục đầu vào để trích xuất các đặc trưng và lưu chúng vào một tệp CSV.
        """
        logger.info(f"Processing input folder: {input_folder}")
        mfcc_data = pd.DataFrame()

        for root, dirs, files in os.walk(input_folder):
            for folder in dirs:
                subdirectory_input = os.path.join(root, folder)
                mfcc_df = self.process_file(folder, subdirectory_input, selected_features)
                mfcc_data = pd.concat([mfcc_data, mfcc_df], ignore_index=True)

        mfcc_data.to_csv(output_csv_path, index=False)
        logger.info(f"Feature extraction completed. Data saved to: {output_csv_path}")