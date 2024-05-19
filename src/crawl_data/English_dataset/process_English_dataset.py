from combine_data.combine_cremad_data import CremaDDataProcessor
from combine_data.combine_ravdess_data import RAVDESSDataProcessor
from combine_data.combine_savee_data import SAVEEDataProcessor
from config import Config

def process_datasets(config):
    input_folder_path = config["input_folder_path"]
    output_folder_path = config["output_folder_path"]
    emotion_mapping = config["emotion_mapping"]

    processor = None
    
    if "CREMA-D" in input_folder_path:
        processor = CremaDDataProcessor(input_folder_path, output_folder_path, emotion_mapping)
    elif "RAVDESS" in input_folder_path:
        processor = RAVDESSDataProcessor(input_folder_path, output_folder_path, emotion_mapping)
    elif "SAVEE" in input_folder_path:
        processor = SAVEEDataProcessor(input_folder_path, output_folder_path, emotion_mapping)
    else:
        raise ValueError("Invalid dataset folder")

    processor.process_files()

def main():
    config_list = [Config.CREMA_D, Config.RAVDESS, Config.SAVEE]

    for config in config_list:
        process_datasets(config)

if __name__ == "__main__":
    main()