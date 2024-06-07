import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
import pandas as pd

# Configure loguru logger
logger.add("mel_spectrogram_extraction.log", rotation="10 MB", retention="10 days", level="INFO")

class MelSpectrogramImageExtractor:
    def __init__(self, n_mels=512, figsize=(128, 128), fmax=8000):
        self.n_mels = n_mels
        self.figsize = figsize
        self.fmax = fmax

    def extract_and_save(self, audio_path, output_path):
        """
        Trích xuất và lưu Mel spectrogram từ tệp âm thanh.

        Parameters:
        - audio_path: Đường dẫn tới tệp âm thanh.
        - output_path: Đường dẫn tới tệp đầu ra.
        """
        try:
            y, sr = librosa.load(audio_path)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=self.figsize)
            ax = plt.axes([0, 0, 1, 1], frameon=False)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=self.fmax, ax=ax, cmap='cool')
            ax.set_axis_off()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")

    def extract_from_file(self, input_file, output_folder):
        """
        Trích xuất Mel spectrogram từ một tệp âm thanh và lưu vào thư mục đầu ra.
        """
        file_name = os.path.basename(input_file)
        file_stem, _ = os.path.splitext(file_name)
        output_path = os.path.join(output_folder, f"{file_stem}.png")
        self.extract_and_save(input_file, output_path)
        return output_path

    def process_folder(self, csv_file, output_folder, output_csv_file):
        file_and_label_df = pd.read_csv(csv_file)
        new_data = []

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for index, row in file_and_label_df.iterrows():
            logger.info(f'Processed {index} files!')
            file_path = row['cleaned_file_path']
            label = row['label']
            new_file_path = self.extract_from_file(file_path, output_folder)
            new_data.append({'file_path': new_file_path, 'label': label})

            if index % 100 == 0:
                new_df = pd.DataFrame(new_data)
                new_df.to_csv(output_csv_file, index=False)

        new_df = pd.DataFrame(new_data)
        new_df.to_csv(output_csv_file, index=False)
        logger.info(f'Saved new CSV file to {output_csv_file}')