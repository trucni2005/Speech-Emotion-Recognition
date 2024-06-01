from config.config import Config
from prepare_data.english_dataset.scripts.combine_data import DataCombinationProcessor
from prepare_data.english_dataset.scripts.train_test_validation_split import TrainTestValidationSplit
from prepare_data.english_dataset.scripts.augment_audio_and_extract_file_path import AudioAugmentationAndFilePathExtraction

def combine_data(dataset_config_list):
    """
    Hàm kết hợp dữ liệu từ các dataset cấu hình trong tệp config.py
    """
    data_processor = DataCombinationProcessor(dataset_config_list)
    data_processor.process_datasets()

def train_test_validation_split(combined_data_path, split_path):
    splitter = TrainTestValidationSplit()
    splitter.process_folders(combined_data_path, split_path)

def augment_audio_and_extract_file_path(split_path, augment_path):
    augmentor = AudioAugmentationAndFilePathExtraction()
    augmentor.process_folders(split_path, augment_path)

if __name__ == '__main__':
    config = Config()

    dataset_config_list = [config.CREMA_D, config.RAVDESS, config.SAVEE]
    combine_data(dataset_config_list)

    combined_data_path = config.eng_combined_data_path
    split_path = config.eng_splited_path
    train_test_validation_split(combined_data_path, split_path)

    augment_path = config.eng_augmented_path
    augment_audio_and_extract_file_path(split_path, augment_path)
