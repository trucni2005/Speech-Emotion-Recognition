import os
from modules.feature_extraction.scripts.extract_mel_spectrogram import MelSpectrogramImageExtractor

class MelSpectrogramExtractionRunner:
    def __init__(self, config):
        """
        Khởi tạo lớp với đối tượng cấu hình.
        
        Args:
            config (object): Đối tượng cấu hình chứa các đường dẫn và tham số.
        """
        self.config = config
        self.n_mels = config.n_mels
        self.input_csv = config.split_data_paths_with_labels['cleaned']
        self.output_folder = config.data_paths['images']['base']
        self.fig_size = config.fig_size

    def mel_spectrogram_extracting(self, dataset_type):
        """
        Hàm thực hiện quá trình trích xuất Mel spectrogram từ dữ liệu âm thanh và lưu kết quả.
        
        Args:
            dataset_type (str): Loại tập dữ liệu (train, test hoặc validation).
        """
        dataset_output_folder = os.path.join(self.output_folder, dataset_type)
        dataset_output_csv = os.path.join(self.output_folder, f"{dataset_type}.csv")
        dataset_input_csv = self.input_csv[dataset_type]

        feature_extractor = MelSpectrogramImageExtractor(self.n_mels, self.fig_size)
        feature_extractor.process_folder(dataset_input_csv, dataset_output_folder, dataset_output_csv)

    def run(self):
        """
        Hàm chạy quá trình trích xuất Mel spectrogram cho tất cả các tập dữ liệu.
        """
        datasets = ['train', 'test', 'validation']
        for dataset_type in datasets:
            print(f"Extracting Mel spectrogram for {dataset_type} dataset...")
            self.mel_spectrogram_extracting(dataset_type)
            print(f"Mel spectrogram extraction for {dataset_type} dataset completed.")