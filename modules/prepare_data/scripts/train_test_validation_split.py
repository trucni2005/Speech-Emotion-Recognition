import os
import random
import shutil
from loguru import logger

class TrainTestValidationSplit:
    """
    Lớp để chia tệp thành các tập train, test và validation.
    """

    def __init__(self, split_ratio):
        self.split_ratio = split_ratio

    def process_folders(self, input_folder, output_folder):
        """
        Chia các tệp trong thư mục đầu vào thành các tập train, test và validation.

        Args:
            input_folder (str): Thư mục chứa các tệp cần chia.
            output_folder (str): Thư mục đầu ra cho các tập train, test và validation.
        """
        logger.info("Processing: Splitting files into train, test, and validation sets.")
        for root, dirs, files in os.walk(input_folder):
            for folder in dirs:
                subdirectory_input = os.path.join(root, folder)
                relative_path = os.path.relpath(subdirectory_input, input_folder)

                train_output_subdirectory = os.path.join(output_folder, 'train', relative_path)
                test_output_subdirectory = os.path.join(output_folder, 'test', relative_path)
                validation_output_subdirectory = os.path.join(output_folder, 'validation', relative_path)

                os.makedirs(train_output_subdirectory, exist_ok=True)
                os.makedirs(test_output_subdirectory, exist_ok=True)
                os.makedirs(validation_output_subdirectory, exist_ok=True)

                self._process_folder(subdirectory_input, train_output_subdirectory, test_output_subdirectory, validation_output_subdirectory)

    def _process_folder(self, input_subdirectory, train_output_subdirectory, test_output_subdirectory, validation_output_subdirectory):
        """
        Chia các tệp trong thư mục con thành các tập train, test và validation.

        Args:
            input_subdirectory (str): Thư mục con chứa các tệp cần chia.
            train_output_subdirectory (str): Thư mục đầu ra cho tập train.
            test_output_subdirectory (str): Thư mục đầu ra cho tập test.
            validation_output_subdirectory (str): Thư mục đầu ra cho tập validation.
            split_ratio (tuple): Bao gồm ba phần tử, lần lượt là tỉ lệ chia cho train, test và validation.
        """
        files = os.listdir(input_subdirectory)
        random.shuffle(files)

        train_split_index = int(self.split_ratio[0] * len(files))
        test_split_index = train_split_index + int(self.split_ratio[1] * len(files))

        train_files = files[:train_split_index]
        test_files = files[train_split_index:test_split_index]
        validation_files = files[test_split_index:]

        self._copy_files(train_files, input_subdirectory, train_output_subdirectory)
        self._copy_files(test_files, input_subdirectory, test_output_subdirectory)
        self._copy_files(validation_files, input_subdirectory, validation_output_subdirectory)


    def _copy_files(self, files, source_folder, destination_folder):
        """
        Sao chép các tệp từ thư mục nguồn sang thư mục đích.

        Args:
            files (list): Danh sách các tệp cần sao chép.
            source_folder (str): Thư mục nguồn chứa các tệp cần sao chép.
            destination_folder (str): Thư mục đích cho các tệp sao chép.
        """
        for file in files:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy(source_path, destination_path)
