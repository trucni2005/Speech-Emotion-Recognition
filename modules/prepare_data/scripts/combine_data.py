from loguru import logger
from modules.prepare_data.modules.cremad_data_processor import CremaDDataProcessor
from modules.prepare_data.modules.ravdess_data_processor import RAVDESSDataProcessor
from modules.prepare_data.modules.savee_data_processor import SaveeDataProcessor

class DataCombinationProcessor:
    """
    Lớp để xử lý và kết hợp các tập dữ liệu từ các nguồn khác nhau.
    """

    def __init__(self, config_list):
        """
        Khởi tạo với danh sách các cấu hình cho từng tập dữ liệu.

        Args:
            config_list (list): Danh sách các cấu hình cho từng tập dữ liệu.
        """
        self.config_list = config_list

    def process_datasets(self):
        """
        Xử lý các tập dữ liệu dựa trên cấu hình đã cung cấp.
        """
        logger.info("Processing: Combine data.")
        for config in self.config_list:
            input_folder_path = config["input_folder_path"]
            output_folder_path = config["output_folder_path"]
            emotion_mapping = config["emotion_mapping"]

            processor = None

            if "CREMA-D" in input_folder_path:
                processor = CremaDDataProcessor(input_folder_path, output_folder_path, emotion_mapping)
            elif "RAVDESS" in input_folder_path:
                processor = RAVDESSDataProcessor(input_folder_path, output_folder_path, emotion_mapping)
            elif "SAVEE" in input_folder_path:
                processor = SaveeDataProcessor(input_folder_path, output_folder_path, emotion_mapping)
            else:
                raise ValueError(f"Invalid dataset folder: {input_folder_path}")

            processor.process_files()
