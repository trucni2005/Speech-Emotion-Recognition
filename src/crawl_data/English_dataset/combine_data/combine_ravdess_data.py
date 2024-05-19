import os
from combine_data.utils.combine_data_utils import move_file_to_emotion_folder

class RAVDESSDataProcessor:
    def __init__(self, input_folder_path, output_folder_path, emotion_mapping):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.emotion_mapping = emotion_mapping

    def get_emotion_from_filename(self, file_name):
        """
        Trích xuất cảm xúc từ tên tệp âm thanh.
        """
        parts = file_name.split('-')
        modality = parts[0]
        vocal_channel = parts[1]
        emotion = parts[2]
        emotional_intensity = parts[3]

        if modality == '03' and vocal_channel == '01':
            return self.emotion_mapping.get(emotion)
        return None

    def process_files(self):
        for file_name in os.listdir(self.input_folder_path):
            if file_name.endswith('.wav'):
                input_file_path = os.path.join(self.input_folder_path, file_name)
                emotion = self.get_emotion_from_filename(file_name)
                if emotion:
                    move_file_to_emotion_folder(input_file_path, self.output_folder_path, emotion)