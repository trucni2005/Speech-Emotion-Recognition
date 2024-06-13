from modules.preparing.scripts.train_test_validation_split import TrainTestValidationSplit
from modules.preparing.scripts.augment_audio_and_extract_file_path import AudioAugmentationAndFilePathExtraction

class DataPreparationRunner:
    def __init__(self, config):
        """
        Khởi tạo lớp với đối tượng cấu hình.
        
        Args:
            config (object): Đối tượng cấu hình chứa các đường dẫn và tham số.
        """
        self.config = config
        self.split_ratio = config.split_ratio
        self.split_path = config.data_paths['splited']['raw']
        self.augment_path = config.data_paths['splited']['augmented']
        self.combined_path = config.data_paths['combined']
    
    def train_test_validation_split(self, split_ratio):
        """
        Hàm chia dữ liệu thành các tập huấn luyện, kiểm tra và đánh giá.
        """
        splitter = TrainTestValidationSplit(split_ratio)
        splitter.process_folders(self.combined_path, self.split_path)
    
    def augment_audio_and_extract_file_path(self):
        """
        Hàm tăng cường dữ liệu âm thanh và trích xuất đường dẫn tệp.
        """
        augmentor = AudioAugmentationAndFilePathExtraction()
        augmentor.process_folders(self.split_path, self.augment_path)
    
    def run(self):
        """
        Hàm chạy toàn bộ quy trình chuẩn bị dữ liệu.
        """
        print("Splitting data into train, test, and validation sets...")
        self.train_test_validation_split(self.split_ratio)
        print("Augmenting audio and extracting file paths...")
        self.augment_audio_and_extract_file_path()
        print("Data preparation completed.")