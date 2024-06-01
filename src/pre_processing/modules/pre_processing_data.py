import os
import librosa
from loguru import logger
from pre_processing.modules.audio_utils import AudioUtils
from pre_processing.modules.file_utils import OutputFileManager
import numpy as np

class PreProcessor:
    """
    Lớp tiền xử lý âm thanh, bao gồm các phương pháp để cắt và lưu các đoạn âm thanh.
    """
    
    def __init__(self, frame_size=256, hop_size=128, best_threshold=0.0003, target_fs=16000, target_dBFS=-15, min_silence_duration=0.1):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.best_threshold = best_threshold
        self.target_fs = target_fs
        self.target_dBFS = target_dBFS
        self.min_silence_duration = min_silence_duration

    def _process_ste(self, y, fs):
        """
        Xử lý Short-term energy (STE) và loại bỏ các khoảng im lặng ngắn.
        """
        ste = AudioUtils.calculate_ste(y, self.frame_size, self.hop_size)
        normalized_ste = AudioUtils.normalize_ste(ste)
        start, end = AudioUtils.find_voiced_segments(normalized_ste, self.best_threshold)
        return start, end

    def _cut_and_save_segment(self, input_file, output_folder, start_time, end_time, target_seconds=None):
        """
        Cắt một đoạn từ tệp âm thanh đầu vào từ giây start_time đến giây end_time và lưu vào thư mục đầu ra.
        """
        OutputFileManager.create_output_folder(output_folder)

        y, fs = librosa.load(input_file, sr=None)
        start_sample, end_sample = AudioUtils.calculate_sample_indices(start_time, end_time, fs)
        segment = AudioUtils.cut_audio_segment(y, start_sample, end_sample)
        resample_y, resample_fs = AudioUtils.normalize_sample_rate(segment, fs, self.target_fs)

        if target_seconds is not None:
            resample_y = self._pad_audio(resample_y, resample_fs, target_seconds)

        output_file_name = AudioUtils.generate_output_file_name(input_file)
        output_file_path = os.path.join(output_folder, output_file_name)
        AudioUtils.save_audio_segment(resample_y, output_file_path, self.target_fs)
        return output_file_path
    
    def _pad_audio(self, signal, sample_rate, target_seconds):
        """
        Thêm đệm vào tín hiệu âm thanh để đạt độ dài mong muốn.
        
        Args:
            signal (numpy.ndarray): Tín hiệu âm thanh ban đầu.
            sample_rate (int): Tần số mẫu của tín hiệu âm thanh.
            target_seconds (int): Độ dài mong muốn của tín hiệu sau khi thêm đệm (tính bằng giây).

        Returns:
            numpy.ndarray: Tín hiệu âm thanh đã thêm đệm với độ dài mong muốn.
        """
        target_length = target_seconds * sample_rate
        current_length = len(signal)

        if current_length >= target_length:
            return signal[:target_length]
        else:
            padding_length = target_length - current_length
            padded_signal = np.pad(signal, (0, padding_length), mode='constant')
            return padded_signal

    def process_audio_file(self, file_path, output_folder):
        """
        Xử lý một tệp âm thanh WAV để tìm các phân đoạn nói và lưu các đoạn đó vào thư mục đầu ra.
        
        Args:
            file_path (str): Đường dẫn đến tệp âm thanh đầu vào.
            output_folder (str): Thư mục để lưu các tệp âm thanh đã xử lý.
        
        Returns:
            str: Đường dẫn đến tệp âm thanh đã được lưu.
        """
        y, fs = AudioUtils.read_wav_file(file_path)
        volume_normalized_y = AudioUtils.normalize_volume_level(y, self.target_dBFS)
        start_index, end_index = self._process_ste(volume_normalized_y, fs)
        start_time = start_index * self.hop_size / fs
        end_time = end_index * self.hop_size / fs
        return self._cut_and_save_segment(file_path, output_folder, start_time, end_time)

    def process_audio_file_with_padding(self, file_path, output_folder, target_seconds):
        """
        Xử lý một tệp âm thanh WAV, thêm đệm vào các đoạn nói và lưu vào thư mục đầu ra.
        
        Args:
            file_path (str): Đường dẫn đến tệp âm thanh đầu vào.
            output_folder (str): Thư mục để lưu các tệp âm thanh đã xử lý.
            target_seconds (int): Độ dài mong muốn của đoạn âm thanh sau khi thêm đệm (tính bằng giây).
        
        Returns:
            str: Đường dẫn đến tệp âm thanh đã được lưu.
        """
        y, fs = AudioUtils.read_wav_file(file_path)
        volume_normalized_y = AudioUtils.normalize_volume_level(y, self.target_dBFS)
        start_index, end_index = self._process_ste(volume_normalized_y, fs)
        start_time = start_index * self.hop_size / fs
        end_time = end_index * self.hop_size / fs
        return self._cut_and_save_segment(file_path, output_folder, start_time, end_time, target_seconds)
