import os
from modules.feature_extraction.scripts.extract_statistic_features import StatisticFeatureExtraction

class FeatureExtractionRunner:
    def __init__(self, config):
        self.config = config
        self.n_mfcc_statistics = config.n_mfcc_statistics
        self.input_csv = config.split_data_paths_with_labels['cleaned']
        self.output_csv = config.statistic_feature_csv

    def statistic_feature_extracting(self, n_mfcc, input_csv, output_csv):
        """
        Hàm thực hiện quá trình trích xuất đặc trưng thống kê từ dữ liệu âm thanh và lưu kết quả vào tập tin CSV.
        
        Args:
            n_mfcc (int): Số lượng hệ số MFCC cần trích xuất.
            input_csv (str): Đường dẫn đến tệp CSV chứa đường dẫn của các tệp âm thanh cần xử lý.
            output_csv (str): Đường dẫn đến tệp CSV để lưu trữ các đặc trưng thống kê được trích xuất.
        """
        feature_extractor = StatisticFeatureExtraction(n_mfcc)
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        feature_extractor.process_folder(input_csv, output_csv)
    
    def run_feature_extraction_for_dataset(self, dataset_type):
        """
        Hàm chạy quá trình trích xuất đặc trưng cho một loại tập dữ liệu cụ thể.
        
        Args:
            dataset_type (str): Loại tập dữ liệu (train, test hoặc validation).
        """
        print(f"Running feature extraction for {dataset_type} dataset...")
        self.statistic_feature_extracting(self.n_mfcc_statistics, self.input_csv[dataset_type], self.output_csv[dataset_type])
        print(f"Feature extraction for {dataset_type} dataset completed.")
    
    def run(self):
        datasets = ['train', 'test', 'validation']
        for dataset_type in datasets:
            self.run_feature_extraction_for_dataset(dataset_type)