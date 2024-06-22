import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from modules.training_and_fine_tuning.utils.model_utils import ModelUtilities
import tensorflow.keras.layers as L

class CnnModelUsingFeatureStatistics:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model(input_shape)
    
    def build_model(self, input_shape):
        model = tf.keras.Sequential([

            L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape),
            L.BatchNormalization(),
            L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPooling1D(pool_size=2, strides=2, padding='same'),
            L.Dropout(0.6),

            L.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPooling1D(pool_size=2, strides=2, padding='same'),
            L.Dropout(0.6),

            L.Flatten(),
            L.Dense(256, activation='relu'),
            L.BatchNormalization(),
            L.Dropout(0.6),
            L.Dense(self.num_classes, activation='softmax')
        ])

                
        model.summary(print_fn=logger.info)
        return model
    
    def compile_model(self, learning_rate=0.0008):
        ModelUtilities.compile_model(self.model, learning_rate)
    
    def train_model(self, X_train, y_train, X_val, y_val, model_name, patience_early_stop=5, patience_lr_reduction=3, factor=0.5, epochs=50, batch_size=64, learning_rate = 0.0008):
        self.compile_model(learning_rate)
        history = ModelUtilities.train_model(self.model, X_train, y_train, X_val, y_val, model_name, patience_early_stop, patience_lr_reduction, factor, epochs, batch_size)
        return history
