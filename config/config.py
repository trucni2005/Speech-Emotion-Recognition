class Config:
    CREMA_D = {
        "input_folder_path": "D:/data_analysis/speech_emotion_recognition/data/uncombined_data/CREMA-D/AudioWAV",
        "output_folder_path": "D:/data_analysis/speech_emotion_recognition/data/combined_data/",
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
        "input_folder_path": "D:/data_analysis/speech_emotion_recognition/data/uncombined_data/RAVDESS",
        "output_folder_path": "D:/data_analysis/speech_emotion_recognition/data/combined_data/",
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
        "input_folder_path": "D:/data_analysis/speech_emotion_recognition/data/uncombined_data/SAVEE",
        "output_folder_path": "D:/data_analysis/speech_emotion_recognition/data/combined_data/",
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
