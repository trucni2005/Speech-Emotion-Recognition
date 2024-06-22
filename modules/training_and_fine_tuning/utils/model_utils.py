import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import numpy as np

class ModelUtilities:
    @staticmethod
    def compile_model(model, learning_rate):
        """
        Thiết lập và biên dịch mô hình CNN.
        """
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    @staticmethod
    def load_model(model_path):
        """
        Tải mô hình đã huấn luyện từ file.
        """
        model = load_model(model_path)
        return model
    
    @staticmethod
    def summary(model):
        """
        In ra thông tin tóm tắt về mô hình.
        """
        model.summary()
    
    @staticmethod
    def train_model(model, X_train, y_train, X_val, y_val, model_name, patience_early_stop=5, patience_lr_reduction=3, factor=0.5, epochs=50, batch_size=64):
        model_checkpoint = ModelCheckpoint(f'{model_name}.keras', monitor='val_accuracy', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=patience_early_stop, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=patience_lr_reduction, verbose=1, factor=factor, min_lr=0.00001)
        
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[early_stop, lr_reduction, model_checkpoint])
        return history

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Đánh giá mô hình trên tập dữ liệu kiểm tra.
        """
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print("Test Accuracy:", test_accuracy)
        return test_loss, test_accuracy

    @staticmethod
    def plot_training_history(history_dict):
        # Lấy các giá trị loss từ lịch sử
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        # Lấy các giá trị accuracy từ lịch sử
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']

        # Số lượng epoch
        epochs = range(1, len(loss_values) + 1)

        # Vẽ đồ thị Loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss_values, 'r-', label='Training loss')  # Đường màu đỏ cho training loss
        plt.plot(epochs, val_loss_values, 'b-', label='Validation loss')  # Đường màu xanh cho validation loss
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Vẽ đồ thị Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc_values, 'r-', label='Training acc')  # Đường màu đỏ cho training accuracy
        plt.plot(epochs, val_acc_values, 'b-', label='Validation acc')  # Đường màu xanh cho validation accuracy
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(model, X_test, y_test, emotion_labels):
        """
        Vẽ ma trận nhầm lẫn cho bộ dữ liệu kiểm tra.
        """
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix_normalized(model, X_test, y_test, emotion_labels):
        """
        Vẽ ma trận nhầm lẫn chuẩn hóa cho bộ dữ liệu kiểm tra.

        Args:
        - model: The trained model used for prediction.
        - X_test: Test data features.
        - y_test: True labels for the test data.
        - emotion_labels: List of emotion labels.
        """
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
        
        # Normalize the confusion matrix
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_normalized_percentage = conf_matrix_normalized * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_normalized_percentage, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
