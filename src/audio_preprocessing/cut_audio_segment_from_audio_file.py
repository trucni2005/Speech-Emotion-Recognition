import os
import librosa
from audio_preprocessing.utils.audio_utils import AudioUtils
from utils.file_utils import OutputFileManager

class AudioProcessor:
    def __init__(self, frame_size=256, hop_size=128, best_threshold=0.001):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.best_threshold = best_threshold
        self.min_silence_duration = 0.1

    def process_ste(self, y, fs):
        """
        Xử lý Short-term energy (STE) và loại bỏ các khoảng im lặng ngắn.
        """
        ste = AudioUtils.calculate_ste(y, self.frame_size, self.hop_size)
        normalized_ste = AudioUtils.normalize_ste(ste)
        start, end = AudioUtils.find_voiced_segments(normalized_ste, self.best_threshold)
        
        return start, end

    def cut_and_save_segment(self, input_file, output_folder, start_time, end_time):
        """
        Cắt một đoạn từ tệp âm thanh đầu vào từ giây a đến giây b và lưu vào thư mục đầu ra.
        """
        OutputFileManager.create_output_folder(output_folder)

        y, sr = librosa.load(input_file, sr=None)
        start_sample, end_sample = AudioUtils.calculate_sample_indices(start_time, end_time, sr)
        segment = AudioUtils.cut_audio_segment(y, start_sample, end_sample)
        output_file_name = AudioUtils.generate_output_file_name(input_file, start_time, end_time)
        output_file_path = os.path.join(output_folder, output_file_name)
        AudioUtils.save_audio_segment(segment, output_file_path, sr)

    def process_audio_file(self, folder, file_path, output_folder):
        """
        Xử lý một tệp âm thanh WAV để tìm các phân đoạn nói.
        """
        y, fs = librosa.load(file_path)

        start_index, end_index = self.process_ste(y, fs)
        start_time = start_index * self.hop_size / fs
        end_time = end_index * self.hop_size / fs
        self.cut_and_save_segment(file_path, output_folder, start_time, end_time)

    def process_folder(self, folder, input_folder_path, output_folder_path):
        """
        Xử lý tất cả các tệp WAV trong thư mục để tìm các phân đoạn nói.
        """
        for file_name in os.listdir(input_folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(input_folder_path, file_name)
                self.process_audio_file(folder, file_path, output_folder_path)

    def process_folders(self, input_folder, output_folder):
        """
        Xử lý tất cả các thư mục trong thư mục đầu vào.
        """
        for root, dirs, files in os.walk(input_folder):
            for folder in dirs:
                subdirectory_input = os.path.join(root, folder)

                relative_path = os.path.relpath(subdirectory_input, input_folder)
                subdirectory_output = os.path.join(output_folder, relative_path)

                if not os.path.exists(subdirectory_output):
                    os.makedirs(subdirectory_output)
                self.process_folder(folder, subdirectory_input, subdirectory_output)
