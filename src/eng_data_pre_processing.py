from config.config import Config
from pre_processing.scripts.pre_processing_data import AudioDataPreProcessor

def data_preprocessing(augmented_path, clean_path, dataset_types, target_second = None):
    """
    Hàm kết hợp dữ liệu từ các dataset cấu hình trong tệp config.py
    """
    data_processor = AudioDataPreProcessor(augmented_path, clean_path, target_second)
    data_processor.process_and_save_audio_files(dataset_types)

if __name__ == '__main__':
    config = Config()
    augmented_path = config.eng_augmented_path
    clean_path = config.eng_cleaned_data_path
    dataset_types = config.dataset_types
    # data_preprocessing(augmented_path, clean_path, dataset_types)

    clean_path_with_pad_or_trim = config.eng_cleaned_data_path_with_pad_or_trim
    target_second = config.target_second_pad_or_trim
    data_preprocessing(augmented_path, clean_path_with_pad_or_trim, dataset_types, target_second)
