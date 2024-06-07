import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras.layers as L

class SpeechEmotionRecognitionModelUsingFeatureStatistics:
    def __init__(self, train_file_path, test_file_path, val_file_path, n_mfcc, learning_rate=0.00008, patience=5, batch_size=64, epochs=50):
        self.n_mfcc = n_mfcc
        X_train, y_train = self.load_data(train_file_path)
        X_test, y_test = self.load_data(test_file_path)
        
        X_val, y_val = self.load_data(val_file_path)
        X_train, X_val, X_test = self.preprocess_data(X_train, X_val, X_test)
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)
        x_traincnn = np.expand_dims(X_train, axis=2)
        x_valcnn = np.expand_dims(X_val, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)
        self.model = self.build_model((X_train.shape[1], 1))
        self.compile_model(learning_rate)
        self.train(x_traincnn, y_train, x_valcnn, y_val, epochs, batch_size, patience)
        self.evaluate(x_testcnn, y_test)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            L.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
            L.BatchNormalization(),
            L.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPooling1D(pool_size=2, strides=2, padding='same'),
            L.Dropout(0.2),

            L.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPooling1D(pool_size=2, strides=2, padding='same'),
            L.Dropout(0.3),

            L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            L.BatchNormalization(),
            L.MaxPooling1D(pool_size=2, strides=2, padding='same'),
            L.Dropout(0.4),

            L.Flatten(),
            L.Dense(256, activation='relu'),
            L.BatchNormalization(),
            L.Dropout(0.3),
            L.Dense(7, activation='softmax')
        ])
        model.summary(print_fn=logger.info)
        return model

    def compile_model(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
        cnn_model_checkpoint = ModelCheckpoint(f'best_cnn_model_weights_using_feature_statistics_{self.n_mfcc}.keras', monitor='val_accuracy', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[early_stop, lr_reduction, cnn_model_checkpoint])
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        logger.info("Test Loss: {}, Test Accuracy: {}", loss, accuracy)

    @staticmethod
    def preprocess_data(X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    @staticmethod
    def encode_labels(y_train, y_val, y_test):
        encoder = OneHotEncoder()
        y_train_encoded = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
        y_val_encoded = encoder.fit_transform(np.array(y_val).reshape(-1,1)).toarray()
        y_test_encoded = encoder.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
        return y_train_encoded, y_val_encoded, y_test_encoded

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        X = data.drop(columns=['label', 'file_path'])
        y = data['label']
        return X, y