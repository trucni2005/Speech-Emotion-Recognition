import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from loguru import logger

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

    def _cross_validate(self, model, X, y, scaler=None):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        acc_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if scaler:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            y_pred = self._train_and_predict(model, X_train, y_train, X_test)

            acc = accuracy_score(y_test, y_pred)
            acc_scores.append(acc)

        avg_acc = np.mean(acc_scores)
        return avg_acc

    def evaluate_models(self, data):
        logger.info("Finding best model....")
        X, y = self._split_data(data)
        
        best_model = None
        best_avg_acc = 0.0
        best_scaler = None

        scalers = [
            ("None", None),
            ("StandardScaler", StandardScaler()),
            ("MinMaxScaler", MinMaxScaler()),
            ("MaxAbsScaler", MaxAbsScaler()),
            ("RobustScaler", RobustScaler()),
            ("Normalizer", Normalizer())
        ]

        results = defaultdict(list)

        for scaler_name, scaler in scalers:
            for model in self.models:
                avg_acc = self._cross_validate(model, X, y, scaler)
                logger.info(f'Model {model} - scaler: {scaler}, accuracy: {avg_acc}')
                results[scaler_name].append((model, avg_acc))

                if avg_acc > best_avg_acc:
                    best_avg_acc = avg_acc
                    best_model = model
                    best_scaler = scaler_name

        logger.info("Best Model: {}", best_model)
        logger.info("Best Scaler: {}", best_scaler)
        logger.info("Best Average Accuracy: {}", best_avg_acc)

        return best_model, best_scaler, best_avg_acc
