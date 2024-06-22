class VietnameseDatasetConfig:
    base_path = "D:/data_analysis/speech_emotion_recognition/data/VietnameseDataset"
    combined_data_path = f"{base_path}/raw/"
    train_test_split_path = f"{base_path}/train_test_splited_data"
    feature_path = f"{base_path}/features"

    n_mels = 512
    data_paths = {
        "combined": combined_data_path,
        "splited": {
            "raw": f"{train_test_split_path}/raw/",
            "augmented": f"{train_test_split_path}/augmented/",
            "cleaned": f"{train_test_split_path}/cleaned/",
        },
        "images": {
            "n_mels": n_mels,
            "base": f"{train_test_split_path}/images/{n_mels}",
            "fig_size": (10, 4)
        },

        "features": {
            "statistics": f"{feature_path}/statistic_features/",
            "features": f"{feature_path}/features/"
        }
    }

    split_data_paths_with_labels = {
        "cleaned": {
            "train": f"{train_test_split_path}/cleaned/train_file_paths_with_labels.csv",
            "validation": f"{train_test_split_path}/cleaned/validation_file_paths_with_labels.csv",
            "test": f"{train_test_split_path}/cleaned/test_file_paths_with_labels.csv",
        },
        "images": {
            "train": f"{train_test_split_path}/images/{data_paths['images']['n_mels']}/train_file_paths_with_labels.csv",
            "validation": f"{train_test_split_path}/images/{data_paths['images']['n_mels']}/validation_file_paths_with_labels.csv",
            "test": f"{train_test_split_path}/images/{data_paths['images']['n_mels']}/test_file_paths_with_labels.csv"
        }
    }

    n_mfcc_statistics = 26
    statistic_feature_csv = {
        "train": f"{feature_path}/statistic_features/{n_mfcc_statistics}/train.csv",
        "test": f"{feature_path}/statistic_features/{n_mfcc_statistics}/test.csv",
        "validation": f"{feature_path}/statistic_features/{n_mfcc_statistics}/validation.csv",
    }

    dataset_types = ["train", "test", "validation"]
    target_second_pad_or_trim = 2
    n_mels = 512
    fig_size = (10, 4)
    normalize_sample_rate = 16000

    split_ratio = (0.7, 0.2, 0.1)

    class ImageConfig:
        def __init__(self, train_test_split_path, n_mels):
            self.train_path = f"{train_test_split_path}/images/{n_mels}/train.csv"
            self.validation_path = f"{train_test_split_path}/images/{n_mels}/validation.csv"
            self.test_path = f"{train_test_split_path}/images/{n_mels}/test.csv"
            
    n_mels_config = {
        128: ImageConfig(train_test_split_path, 128),
        256: ImageConfig(train_test_split_path, 256),
        512: ImageConfig(train_test_split_path, 512)
    }

    class FeatureStatisticsConfig:
        def __init__(self, feature_path, n_mfcc):
            self.train_path = f"{feature_path}/statistic_features/{n_mfcc}/train.csv"
            self.validation_path = f"{feature_path}/statistic_features/{n_mfcc}/validation.csv"
            self.test_path = f"{feature_path}/statistic_features/{n_mfcc}/test.csv"
            
    n_mfcc_config = {
        13: FeatureStatisticsConfig(feature_path, 13),
        20: FeatureStatisticsConfig(feature_path, 20),
        26: FeatureStatisticsConfig(feature_path, 26),
        40: FeatureStatisticsConfig(feature_path, 40),
    }
