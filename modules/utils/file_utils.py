import os

class OutputFileManager:
    @staticmethod
    def create_output_folder(output_folder):
        """
        Kiểm tra và tạo thư mục đầu ra nếu nó không tồn tại.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def generate_output_file_name(input_file):
        """
        Tạo tên file cho đoạn cắt.
        """
        file_name = os.path.basename(input_file)
        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_name = f"{file_name_without_extension}.wav"
        return output_file_name
