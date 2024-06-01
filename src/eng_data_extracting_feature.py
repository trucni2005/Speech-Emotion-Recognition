from config.config import Config
from feature_extraction.scripts.extract_statistic_features import StatisticFeatureExtraction

def statistic_feature_extracting(n_mfcc, input_csv, output_csv):
    """
    Hàm thực hiện quá trình trích xuất đặc trưng thống kê từ dữ liệu âm thanh và lưu kết quả vào tập tin CSV.
    
    Args:
        n_mfcc (int): Số lượng hệ số MFCC cần trích xuất.
        input_csv (str): Đường dẫn đến tệp CSV chứa đường dẫn của các tệp âm thanh cần xử lý.
        output_csv (str): Đường dẫn đến tệp CSV để lưu trữ các đặc trưng thống kê được trích xuất.
    """
    feature_extractor = StatisticFeatureExtraction(n_mfcc)
    feature_extractor.process_folder(input_csv, output_csv)
    
def run_feature_extraction_for_dataset(dataset_type, n_mfcc, input_csv, output_csv):
    """
    Hàm chạy quá trình trích xuất đặc trưng cho một loại tập dữ liệu cụ thể.
    
    Args:
        dataset_type (str): Loại tập dữ liệu (train, test hoặc validation).
        n_mfcc (int): Số lượng hệ số MFCC cần trích xuất.
        input_csv (str): Đường dẫn đến tệp CSV chứa đường dẫn của các tệp âm thanh cần xử lý.
        output_csv (str): Đường dẫn đến tệp CSV để lưu trữ các đặc trưng thống kê được trích xuất.
    """
    print(f"Running feature extraction for {dataset_type} dataset...")
    statistic_feature_extracting(n_mfcc, input_csv[dataset_type], output_csv[dataset_type])
    print(f"Feature extraction for {dataset_type} dataset completed.")

if __name__ == '__main__':
    config = Config()
    n_mfcc_statistics = config.n_mfcc_statistics
    input_csv = config.cleaned_data_path_and_label_csv
    output_csv = config.statistic_feature_csv

    datasets = ['train', 'test', 'validation']
    for dataset_type in datasets:
        run_feature_extraction_for_dataset(dataset_type, n_mfcc_statistics, input_csv, output_csv)
