import os
from combine_data.utils.combine_data_utils import move_file_to_emotion_folder

class SAVEEDataProcessor:
    def __init__(self, input_folder_path, output_folder_path, emotion_mapping):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.emotion_mapping = emotion_mapping

    def get_emotion_from_filename(self, file_name):
        """
        Trích xuất cảm xúc từ tên tệp âm thanh.
        """
        for emotion, label in self.emotion_mapping.items():
            if file_name.startswith(emotion):
                return label
        return None

    def process_files(self):
        for file_name in os.listdir(self.input_folder_path):
            if file_name.endswith('.wav'):
                input_file_path = os.path.join(self.input_folder_path, file_name)
                emotion = self.get_emotion_from_filename(file_name)
                if emotion:
                    move_file_to_emotion_folder(input_file_path, self.output_folder_path, emotion)