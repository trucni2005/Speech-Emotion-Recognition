import numpy as np
import librosa
import soundfile as sf
import os

class AudioUtils:
    @staticmethod
    def read_wav_file(file_path):
        """
        Đọc tệp WAV và trả về dữ liệu âm thanh (y) và tốc độ lấy mẫu (fs).
        """
        y, fs = librosa.load(file_path)
        return y, fs
    
    @staticmethod
    def calculate_ste(y, frame_size=256, hop_size=128):
        num_frames = len(y) - frame_size + hop_size
        num_frames = num_frames // hop_size
        ste = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            ste[i] = np.sum(np.abs(y[start:end]) ** 2)

        return ste

    @staticmethod
    def find_voiced_segments(ste, threshold):
        """
        Tìm các phân đoạn nói dựa trên giá trị STE và ngưỡng tốt nhất.
        """
        derivatives = np.gradient(ste)
        increases = np.where(derivatives > threshold)[0]
        decreases = np.where(derivatives < -threshold)[0]

        increase = None
        decrease = None

        for i in range(len(increases) - 3):
            if increases[i + 3] - increases[i] > 50:
                continue
            increase = increases[i]
            break

        for i in range(len(decreases) - 3):
            if decreases[-1-i] - decreases[-1- i- 3] > 50:
                continue
            decrease = decreases[-1-i]
            break


        # Chọn chỉ số bắt đầu và kết thúc của phân đoạn nói
        start = increase if increase else 0
        end = decrease if decrease else len(ste)

        return start, end
    
    @staticmethod
    def remove_short_silences(starts, ends, hop_size, fs, min_silence_duration=0.3):
        """
        Loại bỏ các khoảng lặng ngắn.
        """
        i = 0
        while i < (len(ends) - 1):
            if (starts[i + 1] * hop_size / fs) - (ends[i] * hop_size / fs) <= min_silence_duration:
                starts.pop(i + 1)
                ends.pop(i)
            else:
                i += 1
        
        return starts, ends
    
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
        segment = y[start_sample:end_sample]
        return segment
    
    @staticmethod
    def generate_output_file_name(input_file, start_time, end_time):
        """
        Tạo tên file cho đoạn cắt.
        """
        file_name = os.path.basename(input_file)
        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_name = f"{file_name_without_extension}.wav"
        return output_file_name
    

    