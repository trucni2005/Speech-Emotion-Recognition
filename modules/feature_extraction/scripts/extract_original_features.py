from loguru import logger
import os
import librosa
import csv
import pandas as pd
import numpy as np
from feature_extraction.modules.mfcc_extractor import MfccExtractor
from feature_extraction.modules.pitch_extractor import PitchExtractor
from feature_extraction.modules.rms_extractor import RmsExtractor
from feature_extraction.modules.zcr_extractor import ZcrExtractor

class FeatureExtractor:
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc

    def extract_original_mfcc(self, signal, sample_rate, frame_size, hop_size):
        mfcc_extractor = MfccExtractor(signal, sample_rate, self.n_mfcc, frame_size, hop_size)
        mfcc_matrix = mfcc_extractor.compute_mfccs()
        mfcc_vectors = []
        for row in range(mfcc_matrix.shape[0]):
            mfcc_vector = mfcc_matrix[row, :]
            mfcc_vectors.append(mfcc_vector)

        return mfcc_vectors
    
    def extract_pitch(self, signal, sample_rate, frame_size, hop_size):
        pitch_extractor = PitchExtractor(signal, sample_rate, frame_size, hop_size)
        return pitch_extractor.compute_pitch()
    
    def extract_original_zcr(self, signal, frame_size, hop_size):
        zcr_extractor = ZcrExtractor(signal, frame_size, hop_size)
        zcr = zcr_extractor.compute_zcr()
        return zcr
    
    def extract_original_rms(self, signal, frame_size, hop_size):
        rms_extractor = RmsExtractor(signal, frame_size, hop_size)
        return rms_extractor.compute_rms()
    
    def pad_or_trim(self, zcr, target_length):
      if len(zcr) < target_length:
          return np.pad(zcr, (0, target_length - len(zcr)), 'constant')
      else:
          return zcr[:target_length]
    
    def calculate_number_of_frames(self, audio_length, sample_rate, frame_length, hop_length):
      """
      Tính số lượng khung dựa vào độ dài tín hiệu âm thanh, tần số lấy mẫu, frame length, và hop length.
      """
      # Tính độ dài tín hiệu âm thanh bằng số mẫu
      signal_length = int(audio_length * sample_rate)
      
      # Tính số lượng khung
      num_frames = 1 + (signal_length - frame_length) // hop_length
      
      return num_frames

    def feature_engineering_for_file(self, audio_file, frame_size=2048, hop_size=512):
        try:
            signal, sample_rate = librosa.load(audio_file, sr=None)
            if len(signal) >= frame_size:
                zcr = self.extract_original_zcr(signal, frame_size, hop_size)
                rms = self.extract_original_rms(signal, frame_size, hop_size)
                pitch = self.extract_pitch(signal, sample_rate, frame_size, hop_size)
                mfccs = self.extract_original_mfcc(signal, sample_rate, frame_size, hop_size)
                
                stacked_mfcc = np.hstack(mfccs)

                combined_features = np.hstack((zcr, rms, pitch, stacked_mfcc))
                return combined_features
        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {e}")
        return None

    def process_folder(self, csv_file, output_csv_file):
        file_and_label_df = pd.read_csv(csv_file)
        feature_dataframes = pd.DataFrame()

        for index, row in file_and_label_df.iterrows():
            file_path = row['cleaned_file_path']
            label = row['label']
            features = self.feature_engineering_for_file(file_path)

            if index % 100 == 0:
                logger.info(f'Processed {index} file')
                feature_dataframes.to_csv(output_csv_file, index=False)

            if features is not None:
                features_dict = {
                    'file_path': file_path,
                    'label': label
                }
                for i, feature in enumerate(features):
                    features_dict[f'feature_{i+1}'] = feature
                feature_dataframe = pd.DataFrame([features_dict]) 
                feature_dataframes = pd.concat([feature_dataframes, feature_dataframe], ignore_index=True)

        feature_dataframes.to_csv(output_csv_file, index=False)