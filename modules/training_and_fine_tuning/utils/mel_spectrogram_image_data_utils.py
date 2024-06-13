import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from loguru import logger

class MelSpectrogramDataUtils:
    @staticmethod
    def read_and_process_image(image_path, target_size=(128, 128)):
        """
        Đọc và xử lý hình ảnh từ đường dẫn cho trước.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Không thể đọc hình ảnh từ đường dẫn: {image_path}")
                return None
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            return img
        except Exception as e:
            logger.error(f"Lỗi khi đọc và xử lý hình ảnh từ đường dẫn: {image_path}, Lỗi: {str(e)}")
            return None

    @staticmethod
    def process_data(csv_file):
        """
        Xử lý dữ liệu từ file CSV chứa đường dẫn hình ảnh và nhãn.
        """
        encoder = OneHotEncoder()
        df = pd.read_csv(csv_file)
        X = []
        y = df['label']
        
        for index, row in df.iterrows():
            image_path = row['file_path']
            img = MelSpectrogramDataUtils.read_and_process_image(image_path)
            if img is not None:
                X.append(img)
        
        X_tensor = np.array(X)
        X_reshaped = X_tensor.reshape(-1, 128, 128, 3)
        y = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
        return X_reshaped, y