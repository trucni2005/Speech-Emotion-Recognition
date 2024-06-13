import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from sklearn.utils import resample
from modules.training_and_fine_tuning.utils.model_utils import ModelUtilities


class CnnModelUsingMelSpectrogram:
    """
    Mô hình nhận dạng cảm xúc trong giọng nói sử dụng CNN.
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=7):
        """
        Khởi tạo các thuộc tính cần thiết cho đối tượng SpeechEmotionRecognitionModel.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        """
        Tạo kiến trúc mô hình CNN.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.2),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.4),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def compile_model(self, learning_rate=0.0008):
        ModelUtilities.compile_model(self.model, learning_rate)
    
    def train_model(self, X_train, y_train, X_val, y_val, model_name, patience_early_stop=5, patience_lr_reduction=3, factor=0.5, epochs=50, batch_size=64, learning_rate = 0.0008):
        self.compile_model(learning_rate)
        history = ModelUtilities.train_model(self.model, X_train, y_train, X_val, y_val, model_name, patience_early_stop, patience_lr_reduction, factor, epochs, batch_size)
        return history
