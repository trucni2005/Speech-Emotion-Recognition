import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils.file_utils import OutputFileManager  # Import OutputFileManager từ thư viện utils
from loguru import logger

# Configure loguru logger
logger.add("mel_spectrogram_extraction.log", rotation="10 MB", retention="10 days", level="INFO")

class MelSpectrogramImageExtractor:
    def __init__(self, n_mels=512, figsize=(10, 4), fmax = 8000):
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
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=self.fmax)
            plt.colorbar(format='%+2.0f dB')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")

    def extract_from_file(self, input_file, output_folder):
        """
        Trích xuất Mel spectrogram từ một tệp âm thanh và lưu vào thư mục đầu ra.
        """
        OutputFileManager.create_output_folder(output_folder)
        file_name = os.path.basename(input_file)
        file_stem, _ = os.path.splitext(file_name)
        output_path = os.path.join(output_folder, f"{file_stem}.png")
        self.extract_and_save(input_file, output_path)

    def extract_from_folder(self, input_folder, output_folder):
        """
        Trích xuất Mel spectrogram từ tất cả các tệp WAV trong một thư mục và lưu vào thư mục đầu ra.
        """
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(input_folder, file_name)
                self.extract_from_file(file_path, output_folder)

    def extract_from_folders(self, input_folder, output_folder):
        """
        Trích xuất Mel spectrogram từ tất cả các thư mục con trong một thư mục và lưu vào thư mục đầu ra.
        """
        logger.info(f"Processing root folder: {input_folder}")
        folder_count = 0
        for root, dirs, files in os.walk(input_folder):
            for folder in dirs:
                subdirectory_input = os.path.join(root, folder)
                relative_path = os.path.relpath(subdirectory_input, input_folder)
                subdirectory_output = os.path.join(output_folder, relative_path)
                if not os.path.exists(subdirectory_output):
                    os.makedirs(subdirectory_output)
                self.extract_from_folder(subdirectory_input, subdirectory_output)
                folder_count += 1
                logger.info(f"Processed {folder_count} subfolders in root folder: {input_folder}")

        print(f"Completed processing root folder: {input_folder}, total subfolders: {folder_count}")
