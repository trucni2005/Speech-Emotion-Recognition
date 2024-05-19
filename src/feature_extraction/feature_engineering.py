import os
import pandas as pd
import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc

    def compute_mfccs(self, signal, sample_rate):
        """
        Tính toán Mel-frequency cepstral coefficients (MFCCs) của tín hiệu âm thanh.
        """
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=self.n_mfcc)
        return np.mean(mfccs.T, axis=0)

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
        # Tính toán pitch sử dụng hàm estimate_tuning của Librosa
        pitch, _ = librosa.core.piptrack(y=signal, sr=sample_rate)
        
        # Lấy giá trị pitch trung bình
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

    def feature_engineering_for_file(self, audio_file):
        """
        Trích xuất các đặc trưng MFCC, ZCR và Pitch từ một tệp âm thanh.
        """
        signal, sample_rate = librosa.load(audio_file, sr=None)
        if len(signal) >= 2048:
            mfcc_array = self.compute_mfccs(signal, sample_rate)
            zcr_mean, zcr_variance, zcr_max, zcr_min, zcr_median = self.compute_zcr(signal)
            pitch = self.extract_pitch(signal, sample_rate)
            rms_mean, rms_variance, rms_max, rms_min, rms_median = self.extract_rms_features(signal)
            return mfcc_array, zcr_mean, zcr_variance, zcr_max, zcr_min, zcr_median, pitch, rms_mean, rms_variance, rms_max, rms_min, rms_median
        return None

    def process_file(self, label, input_folder_path):
        """
        Xử lý tất cả các tệp WAV trong thư mục để trích xuất các đặc trưng.
        """
        # Tạo một DataFrame để lưu trữ đặc trưng MFCC
        mfcc_data = pd.DataFrame()

        for file_name in os.listdir(input_folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(input_folder_path, file_name)

                features = self.feature_engineering_for_file(file_path)
                if features is not None:
                    mfcc_array, zcr_mean, zcr_variance, zcr_max, zcr_min, zcr_median, pitch, rms_mean, rms_variance, rms_max, rms_min, rms_median = features
                    
                    mfcc_df = pd.DataFrame(mfcc_array.reshape(1, -1), columns=[f'mfcc_{i+1}' for i in range(len(mfcc_array))])

                    mfcc_df['label'] = label
                    mfcc_df['file'] = file_name
                    mfcc_df['zcr_mean'] = zcr_mean
                    mfcc_df['zcr_variance'] = zcr_variance
                    mfcc_df['zcr_max'] = zcr_max
                    mfcc_df['zcr_min'] = zcr_min
                    mfcc_df['zcr_median'] = zcr_median
                    mfcc_df['Pitch'] = pitch
                    mfcc_df['rms_mean'] = rms_mean
                    mfcc_df['rms_variance'] = rms_variance
                    mfcc_df['rms_max'] = rms_max
                    mfcc_df['rms_min'] = rms_min
                    mfcc_df['rms_median'] = rms_median
                    
                    mfcc_data = pd.concat([mfcc_data, mfcc_df], ignore_index=True)
            
        return mfcc_data

    def process_folder(self, input_folder, output_csv_path):
        """
        Xử lý tất cả các thư mục con trong thư mục đầu vào để trích xuất các đặc trưng và lưu chúng vào một tệp CSV.
        """
        # Tạo một DataFrame để lưu trữ dữ liệu MFCC
        mfcc_data = pd.DataFrame()

        # Duyệt qua tất cả các thư mục con trong thư mục đầu vào
        for root, dirs, files in os.walk(input_folder):
            for folder in dirs:
                # Xác định đường dẫn của thư mục con trong thư mục đầu vào
                subdirectory_input = os.path.join(root, folder)
                # Xử lý các tệp trong thư mục con và thêm dữ liệu MFCC vào DataFrame
                mfcc_df = self.process_file(folder, subdirectory_input)
                mfcc_data = pd.concat([mfcc_data, mfcc_df], ignore_index=True)

        # Lưu dữ liệu MFCC vào tệp CSV
        mfcc_data.to_csv(output_csv_path, index=False)