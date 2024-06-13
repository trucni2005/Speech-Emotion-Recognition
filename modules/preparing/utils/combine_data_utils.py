import os
import shutil

def move_file_to_emotion_folder(input_file_path, output_folder_path, emotion):
    """
    Di chuyển tệp âm thanh vào thư mục của cảm xúc tương ứng.
    """
    output_folder = os.path.join(output_folder_path, emotion)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, os.path.basename(input_file_path))
    shutil.move(input_file_path, output_file_path)
