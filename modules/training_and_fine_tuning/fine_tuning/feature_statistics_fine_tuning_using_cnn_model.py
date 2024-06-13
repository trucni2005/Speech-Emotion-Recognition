import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import class_weight

class FineTuningModel:
    def __init__(self, model_path, learning_rate=0.0001):
        self.model = load_model(model_path)
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        X = data.drop(columns=['label', 'file_path'])
        y = data['label']
        return X, y

    def encode_labels(self, y_train, y_val, y_test):
        encoder = OneHotEncoder()
        y_train_encoded = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
        y_val_encoded = encoder.transform(np.array(y_val).reshape(-1,1)).toarray()
        y_test_encoded = encoder.transform(np.array(y_test).reshape(-1,1)).toarray()
        self.emotion_labels = encoder.categories_[0]
        return y_train_encoded, y_val_encoded, y_test_encoded

    def fit_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16):
        # Định nghĩa các callback
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=4, verbose=1, factor=0.5, min_lr=0.00001)
        
        # Huấn luyện mô hình
        self.history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), 
                                      batch_size=batch_size, callbacks=[early_stop, lr_reduction])

    def evaluate_model(self, X_test, y_test):
        self.test_loss, self.test_accuracy = self.model.evaluate(X_test, y_test)
        print("Test Accuracy:", self.test_accuracy)

    def plot_training_history(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.emotion_labels, yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

