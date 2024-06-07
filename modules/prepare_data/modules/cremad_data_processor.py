import os
from prepare_data.english_dataset.utils.combine_data_utils import move_file_to_emotion_folder

class CremaDDataProcessor:
    """
    Lớp để xử lý dữ liệu CREMA-D, bao gồm việc phân loại và di chuyển các tệp âm thanh
    vào các thư mục theo cảm xúc.
    """

    def __init__(self, input_folder_path, output_folder_path, emotion_mapping):
        """
        Khởi tạo lớp với đường dẫn thư mục đầu vào, đầu ra và ánh xạ cảm xúc.

        Args:
            input_folder_path (str): Đường dẫn đến thư mục chứa các tệp âm thanh đầu vào.
            output_folder_path (str): Đường dẫn đến thư mục chứa các tệp âm thanh đã xử lý.
            emotion_mapping (dict): Bảng ánh xạ các viết tắt cảm xúc với tên cảm xúc đầy đủ.
        """
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.emotion_mapping = emotion_mapping

    def _get_emotion_from_filename(self, file_name):
        """
        Trích xuất cảm xúc từ tên tệp âm thanh.

        Args:
            file_name (str): Tên tệp âm thanh.

        Returns:
            str: Tên cảm xúc tương ứng hoặc None nếu không tìm thấy.
        """
        parts = file_name.split('_')
        emotion_abbr = parts[2]
        return self.emotion_mapping.get(emotion_abbr)

    def process_files(self):
        """
        Xử lý tất cả các tệp WAV trong thư mục đầu vào và di chuyển chúng
        vào các thư mục theo cảm xúc.
        """
        for file_name in os.listdir(self.input_folder_path):
            if file_name.endswith('.wav'):
                input_file_path = os.path.join(self.input_folder_path, file_name)
                emotion = self._get_emotion_from_filename(file_name)
                if emotion:
                    move_file_to_emotion_folder(input_file_path, self.output_folder_path, emotion)
