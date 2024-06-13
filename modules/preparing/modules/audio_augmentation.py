import os
import numpy as np
import librosa
import soundfile as sf

class AudioAugmentation:
    """
    Lớp hỗ trợ tăng cường dữ liệu âm thanh bằng cách thêm nhiễu và dịch chuyển thời gian.
    """
    
    def __init__(self, noise_factor=0.035, shift_range=10000):
        self.noise_factor = noise_factor
        self.shift_range = shift_range

    def _add_noise(self, data):
        """
        Thêm nhiễu ngẫu nhiên vào dữ liệu âm thanh.
        
        Args:
            data (numpy.ndarray): Dữ liệu âm thanh ban đầu.

        Returns:
            numpy.ndarray: Dữ liệu âm thanh đã thêm nhiễu.
        """
        noise_amp = self.noise_factor * np.random.uniform() * np.amax(data)
        return data + noise_amp * np.random.normal(size=data.shape[0])

    def _time_shift(self, data):
        """
        Dịch chuyển thời gian của dữ liệu âm thanh.

        Args:
            data (numpy.ndarray): Dữ liệu âm thanh ban đầu.

        Returns:
            numpy.ndarray: Dữ liệu âm thanh đã dịch chuyển thời gian.
        """
        shift_value = int(np.random.uniform(low=-self.shift_range, high=self.shift_range))
        return np.roll(data, shift_value)
    
    def audio_augmentation(self, file_name, subdirectory_path, output_folder):
        """
        Tăng cường dữ liệu âm thanh bằng cách thêm nhiễu và dịch chuyển thời gian.

        Args:
            file_name (str): Tên tệp âm thanh đầu vào.
            subdirectory_path (str): Đường dẫn đến thư mục chứa tệp âm thanh đầu vào.
            output_folder (str): Thư mục để lưu các tệp âm thanh đã tăng cường.

        Returns:
            tuple: Đường dẫn đến các tệp âm thanh đã xử lý (gốc, thêm nhiễu, dịch chuyển thời gian).
        """
        file_path = os.path.join(subdirectory_path, file_name)
        data, sr = librosa.load(file_path, sr=None)

        # Tạo thư mục đích nếu nó chưa tồn tại
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        original_audio_path = os.path.join(output_folder, file_name)
        sf.write(original_audio_path, data, sr)

        noisy_data = self._add_noise(data)
        noisy_audio_path = os.path.join(output_folder, f'noisy_{file_name}')
        sf.write(noisy_audio_path, noisy_data, sr)

        time_shift_data = self._time_shift(data)
        time_shift_audio_path = os.path.join(output_folder, f'time_shift_{file_name}')
        sf.write(time_shift_audio_path, time_shift_data, sr)
        
        return original_audio_path, noisy_audio_path, time_shift_audio_path
