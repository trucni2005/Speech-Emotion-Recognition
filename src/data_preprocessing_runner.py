from modules.pre_processing.scripts.pre_processing_data import AudioDataPreProcessor

class DataPreprocessingRunner:
    def __init__(self, config):
        """
        Khởi tạo lớp với đối tượng cấu hình.
        
        Args:
            config (object): Đối tượng cấu hình chứa các đường dẫn và tham số.
        """
        self.config = config
        self.augmented_path = config.data_paths['splited']['augmented']
        self.clean_path = config.data_paths['splited']['cleaned']
        self.dataset_types = config.dataset_types
        self.target_second = config.target_second_pad_or_trim

    def data_preprocessing(self):
        """
        Hàm kết hợp dữ liệu từ các dataset cấu hình trong tệp config.py.
        """
        data_processor = AudioDataPreProcessor(self.augmented_path, self.clean_path, self.target_second)
        data_processor.process_and_save_audio_files(self.dataset_types)
    
    def run(self):
        """
        Hàm chạy quá trình tiền xử lý dữ liệu.
        """
        self.data_preprocessing()
