import pandas as pd
from modules.pre_processing.modules.pre_processing_data import PreProcessor
from loguru import logger 

class AudioDataPreProcessor:
    def __init__(self, base_input_folder, base_output_folder, target_second=None):
        self.base_input_folder = base_input_folder
        self.base_output_folder = base_output_folder
        self.target_second = target_second
        self.preProcessor = PreProcessor()

    def process_file(self, file_path, output_folder):
        """
        Xử lý một tệp âm thanh và lưu trữ nó.

        Args:
            file_path (str): Đường dẫn đến tệp âm thanh đầu vào.
            output_folder (str): Thư mục đích để lưu trữ tệp đã xử lý.

        Returns:
            str: Đường dẫn đến tệp đã xử lý.
        """
        try:
            if self.target_second is None:
                cleaned_file_path = self.preProcessor.process_audio_file(file_path, output_folder)
            else:
                cleaned_file_path = self.preProcessor.process_audio_file_with_padding(file_path, output_folder, self.target_second)
            return cleaned_file_path
        except Exception as e:
            logger.exception(f"Error processing file: {file_path}, Error: {e}")
            return None

    def process_and_save_audio_files(self, dataset_types):
        """
        Hàm xử lý và lưu trữ các tệp âm thanh cho các tập dữ liệu khác nhau.

        Args:
            dataset_types (list): Danh sách các loại tập dữ liệu (ví dụ: ['train', 'test', 'validation']).

        Returns:
            None
        """
        for dataset_type in dataset_types:
            input_file = rf"{self.base_input_folder}\{dataset_type}_file_paths_with_labels.csv"
            output_folder = rf"{self.base_output_folder}\{dataset_type}"

            df = pd.read_csv(input_file)
            new_data = []

            for index, row in df.iterrows():
                file_path = row['file_path']
                cleaned_file_path = self.process_file(file_path, output_folder)
                if cleaned_file_path is not None:
                    label = row['label']
                    new_data.append({'cleaned_file_path': cleaned_file_path, 'label': label})
                if index % 1000 == 0:
                    logger.info(f"Processed {index} files in {dataset_type} dataset.")

            # Tạo DataFrame mới từ danh sách các dictionary
            new_df = pd.DataFrame(new_data)

            # Lưu DataFrame mới vào tệp CSV
            new_csv_file = rf"{self.base_output_folder}\{dataset_type}_file_paths_with_labels.csv"
            new_df.to_csv(new_csv_file, index=False)
