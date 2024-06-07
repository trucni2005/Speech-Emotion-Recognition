import numpy as np
import librosa
import soundfile as sf
import os
from scipy.signal import resample

class AudioUtils:
    @staticmethod
    def read_wav_file(file_path):
        """
        Đọc tệp WAV và trả về dữ liệu âm thanh (y) và tốc độ lấy mẫu (fs).
        """
        y, fs = librosa.load(file_path, sr=None)
        return y, fs

    @staticmethod
    def calculate_ste(y, frame_size=256, hop_size=128):
        """
        Tính toán năng lượng theo thời gian ngắn (STE).
        """
        num_frames = (len(y) - frame_size) // hop_size + 1
        ste = np.array([
            np.sum(np.abs(y[i * hop_size : i * hop_size + frame_size]) ** 2)
            for i in range(num_frames)
        ])
        return ste

    @staticmethod
    def find_voiced_segments(ste, threshold):
        """
        Tìm các phân đoạn có giọng nói dựa trên giá trị STE và ngưỡng tốt nhất.
        """
        derivatives = np.gradient(ste)
        increases = np.where(derivatives > threshold)[0]
        decreases = np.where(derivatives < -threshold)[0]

        def find_first_contiguous(arr, span):
            for i in range(len(arr) - span):
                if arr[i + span] - arr[i] <= span:
                    return arr[i]
            return None

        start = find_first_contiguous(increases, 3) or 0
        end = find_first_contiguous(decreases[::-1], 3)
        end = end if end is not None else len(ste)
        
        return start, end

    @staticmethod
    def normalize_sample_rate(audio_data, orig_sr, target_sr):
        """
        Chuẩn hóa tốc độ lấy mẫu của dữ liệu âm thanh.
        """
        if orig_sr != target_sr:
            num_samples = int(round(len(audio_data) * target_sr / orig_sr))
            resampled_audio = resample(audio_data, num_samples)
            return resampled_audio, target_sr
        return audio_data, orig_sr

    @staticmethod
    def normalize_volume_level(audio, target_dBFS):
        """
        Chuẩn hóa mức âm lượng của dữ liệu âm thanh đến dBFS mục tiêu.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        current_dBFS = 20 * np.log10(rms)
        change_in_dBFS = target_dBFS - current_dBFS
        gain = 10 ** (change_in_dBFS / 20)
        normalized_y = audio * gain
        return normalized_y

    @staticmethod
    def remove_short_silences(starts, ends, hop_size, fs, min_silence_duration=0.3):
        """
        Loại bỏ các khoảng lặng ngắn giữa các phân đoạn có giọng nói.
        """
        min_silence_samples = min_silence_duration * fs / hop_size
        filtered_starts, filtered_ends = [], []

        for i in range(len(ends) - 1):
            if (starts[i + 1] - ends[i]) > min_silence_samples:
                filtered_starts.append(starts[i])
                filtered_ends.append(ends[i])

        return filtered_starts, filtered_ends

    @staticmethod
    def normalize_ste(ste):
        """
        Chuẩn hóa năng lượng theo thời gian ngắn (STE).
        """
        return ste / np.max(ste)

    @staticmethod
    def save_audio_segment(segment, output_file_path, sr):
        """
        Lưu đoạn âm thanh vào tệp WAV.
        """
        sf.write(output_file_path, segment, sr)

    @staticmethod
    def calculate_sample_indices(start_time, end_time, sr):
        """
        Tính chỉ số mẫu bắt đầu và kết thúc từ thời gian (giây).
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return start_sample, end_sample

    @staticmethod
    def cut_audio_segment(y, start_sample, end_sample):
        """
        Cắt đoạn âm thanh từ mẫu bắt đầu đến mẫu kết thúc.
        """
        return y[start_sample:end_sample]

    @staticmethod
    def generate_output_file_name(input_file):
        """
        Tạo tên file cho đoạn cắt.
        """
        file_name = os.path.basename(input_file)
        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_name = f"{file_name_without_extension}.wav"
        return output_file_name
