import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

class ModelEvaluator:
    def __init__(self, models):
        self.models = models

    def _split_data(self, data):
        y = data['label']
        X = data.drop(['label', 'file'], axis=1)
        return X, y

    def _train_and_predict(self, model, X_train, y_train, X_test):
        # Khởi tạo model
        clf = model()

        # Train model
        clf.fit(X_train, y_train)

        # Dự đoán trên tập test
        y_pred = clf.predict(X_test)

        return y_pred

    def _cross_validate(self, model, X, y):
        kf = KFold(n_splits=10, shuffle=False)
        acc_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_pred = self._train_and_predict(model, X_train, y_train, X_test)

            acc = accuracy_score(y_test, y_pred)
            acc_scores.append(acc)

        avg_acc = np.mean(acc_scores)
        return avg_acc
    
    def _clean_data(self, df):
        # Kiểm tra và thay thế các giá trị vô cùng bằng NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # In ra số lượng giá trị NaN trước khi loại bỏ
        print("Number of NaNs before drop:", df.isna().sum().sum())
        
        # Loại bỏ các hàng có giá trị NaN
        df.dropna(inplace=True)
        
        # In ra số lượng hàng và cột sau khi làm sạch dữ liệu
        print("Shape after cleaning:", df.shape)
        
        return df
    
    def evaluate_models(self, data):
        X, y = self._split_data(data)
        X = self._clean_data(X)
        
        best_model = None
        best_avg_acc = 0.0

        # Loop through models
        for model in self.models:
            avg_acc = self._cross_validate(model, X, y)
            
            # Lưu mô hình có độ chính xác trung bình cao nhất
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_model = model

        # Trả về mô hình có độ chính xác trung bình cao nhất
        return best_model, best_avg_acc
