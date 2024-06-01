class Config:
    CREMA_D = {
        "input_folder_path": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/uncombined_data/CREMA-D/AudioWAV",
        "output_folder_path": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/combined_data/",
        "emotion_mapping": {
            "HAP": "Happy",
            "NEU": "Neutral",
            "FEA": "Fearful",
            "SAD": "Sad",
            "ANG": "Angry",
            "DIS": "Disgusted"
        }
    }

    RAVDESS = {
        "input_folder_path": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/uncombined_data/RAVDESS",
        "output_folder_path": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/combined_data/",
        "emotion_mapping": {
            "03": "Happy",
            "01": "Neutral",
            "06": "Fearful",
            "04": "Sad",
            "05": "Angry",
            "07": "Disgusted",
            "08": "Surprised"
        }
    }

    SAVEE = {
        "input_folder_path": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/uncombined_data/SAVEE",
        "output_folder_path": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/combined_data/",
        "emotion_mapping": {
            "h": "Happy",
            "n": "Neutral",
            "f": "Fearful",
            "sa": "Sad",
            "a": "Angry",
            "d": "Disgusted",
            "su": "Surprised"
        }
    }

    eng_combined_data_path = "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/combined_data/"
    eng_splited_path = "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/raw/"
    eng_augmented_path = "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/augmented/"
    eng_cleaned_data_path = "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned/"
    eng_cleaned_data_path_with_pad_or_trim = "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned_with_pad_or_trim/"
    dataset_types = ["train", "test", "validation"]
    target_second_pad_or_trim = 2
    n_mfcc_statistics = 20
    n_mfcc_original = 13

    cleaned_data_path_and_label_csv = {
        "train": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned/train_file_paths_with_labels.csv",
        "validation": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned/validation_file_paths_with_labels.csv", 
        "test": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned/test_file_paths_with_labels.csv"
    }

    statistic_feature_csv = {
        "train": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/features/statistic_features/train.csv",
        "test": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/features/statistic_features/test.csv",
        "validation": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/features/statistic_features/validation.csv"
    }

    normalize_sample_rate = 16000

    cleaned_data_with_pad_and_trim_path_and_label_csv = {
        "train": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned_with_pad_or_trim/train_file_paths_with_labels.csv",
        "validation": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned_with_pad_or_trim/validation_file_paths_with_labels.csv", 
        "test": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/train_test_splited_data/cleaned_with_pad_or_trim/test_file_paths_with_labels.csv"
    }

    feature_csv = {
        "train": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/features/features/train.csv",
        "test": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/features/features/test.csv",
        "validation": "D:/data_analysis/speech_emotion_recognition/data/EnglishDataset/features/features/validation.csv"
    }