import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class SpeechEmotionRecognitionModelUsingMelSpectrogram:
    """
    Mô hình nhận dạng cảm xúc trong giọng nói sử dụng CNN.
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=7, learning_rate=0.00008):
        """
        Khởi tạo các thuộc tính cần thiết cho đối tượng SpeechEmotionRecognitionModel.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.compile_model()
    
    @staticmethod
    def read_and_process_image(image_path, target_size=(128, 128)):
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
            img = SpeechEmotionRecognitionModelUsingMelSpectrogram.read_and_process_image(image_path)
            X.append(img)
        
        X_tensor = np.array(X)
        X_reshaped = X_tensor.reshape(-1, 128, 128, 3)
        y = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
        return X_reshaped, y

    def create_model(self):
        """
        Tạo kiến trúc mô hình CNN.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile_model(self):
        """
        Thiết lập và biên dịch mô hình CNN.
        """
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, n_mels, epochs=50, batch_size=64):
        """
        Huấn luyện mô hình CNN với dữ liệu huấn luyện và kiểm định.
        """
        cnn_model_checkpoint = ModelCheckpoint(f'best_cnn_model_weights_using_mel_spectrogram_{n_mels}.keras', monitor='val_accuracy', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[early_stop, lr_reduction, cnn_model_checkpoint])
        return history

    def summary(self):
        """
        In ra thông tin tóm tắt về mô hình.
        """
        self.model.summary()