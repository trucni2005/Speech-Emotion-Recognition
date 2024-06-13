import os
import pandas as pd
from loguru import logger
from modules.preparing.modules.audio_augmentation import AudioAugmentation

class AudioAugmentationAndFilePathExtraction:
    """
    Lớp để tăng cường âm thanh và trích xuất đường dẫn tệp với nhãn.
    """

    def __init__(self):
        pass

    def _process_file(self, folder, subdirectory_path, output_folder):
        """
        Xử lý tất cả các tệp WAV trong thư mục: tăng cường dữ liệu và tạo DataFrame với cột 'file_path' và 'label'.

        Args:
            folder (str): Tên thư mục hiện tại (cảm xúc).
            subdirectory_path (str): Đường dẫn đến thư mục con chứa các tệp âm thanh.
            output_folder (str): Thư mục đầu ra để lưu các tệp âm thanh tăng cường.

        Returns:
            list: Danh sách các tuple chứa đường dẫn tệp và nhãn.
        """
        label = folder
        file_paths_with_labels = []
        audio_augmentation = AudioAugmentation()

        for file_name in os.listdir(subdirectory_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(subdirectory_path, file_name)
                original_audio_path, noisy_audio_path, time_shift_audio_path = audio_augmentation.audio_augmentation(
                    file_name, subdirectory_path, output_folder
                )
                file_paths_with_labels.append((original_audio_path, label))
                file_paths_with_labels.append((noisy_audio_path, label))
                file_paths_with_labels.append((time_shift_audio_path, label))
        
        return file_paths_with_labels

    def _process_folder(self, input_folder, output_folder):
        """
        Xử lý tất cả các thư mục con trong thư mục đầu vào.

        Args:
            input_folder (str): Thư mục đầu vào chứa các thư mục con với các tệp âm thanh.
            output_folder (str): Thư mục đầu ra để lưu các tệp âm thanh tăng cường.

        Returns:
            list: Danh sách các tuple chứa đường dẫn tệp và nhãn từ tất cả các thư mục con.
        """
        all_file_paths_with_labels = []
        for root, dirs, _ in os.walk(input_folder):
            for folder in dirs:
                logger.info(f"Processing folder: {folder}")
                subdirectory_path = os.path.join(root, folder)
                file_paths_with_labels = self._process_file(folder, subdirectory_path, output_folder)
                all_file_paths_with_labels.extend(file_paths_with_labels)
        
        return all_file_paths_with_labels
    
    def process_folders(self, input_folder, output_folder):
        """
        Xử lý các thư mục đầu vào cho từng loại tập dữ liệu (train, test, validation).

        Args:
            input_folder (str): Thư mục đầu vào chứa các tập dữ liệu.
            output_folder (str): Thư mục đầu ra để lưu các tệp âm thanh tăng cường và CSV.
        """
        for dataset_type in ['train', 'test', 'validation']:
            _input_folder = os.path.join(input_folder, dataset_type)
            _output_folder = os.path.join(output_folder, dataset_type)
            logger.info(f"Processing: {_input_folder}")
            all_file_paths_with_labels = self._process_folder(_input_folder, _output_folder)
            df = pd.DataFrame(all_file_paths_with_labels, columns=['file_path', 'label'])
            csv_path = os.path.join(output_folder, f'{dataset_type}_file_paths_with_labels.csv')
            df.to_csv(csv_path, index=False)
