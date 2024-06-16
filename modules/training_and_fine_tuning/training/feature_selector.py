import os
import pandas as pd
from loguru import logger
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from modules.training_and_fine_tuning.utils.feature_statistics_data_utils import DataUtils

class FeatureSelector:
    def __init__(self, config, n_mfcc=20):
        self.config = config
        self.n_mfcc = n_mfcc
        self.features_combo = [
            ['mfcc', 'zcr', 'rms', 'pitch'], 
            ['mfcc', 'rms', 'pitch'], 
            ['mfcc', 'zcr', 'pitch'], 
            ['mfcc', 'zcr', 'rms']
        ]
        self.statistics = ['mean', 'variance', 'max', 'min', 'median', '25th_percentile', '75th_percentile']
        self.models = [RandomForestClassifier(random_state=42), ExtraTreesClassifier(random_state=42)]
        self.best_features = None
        self.best_accuracy = 0

    def generate_feature_combinations(self, features):
        n = len(features)
        all_combinations = []
        for r in range(n-2):
            for combo in combinations(range(n), r):
                selected_features = [features[i] for i in range(n) if i not in combo]
                all_combinations.append(selected_features)
        return all_combinations

    def get_n_mfcc_paths(self):
        config_entry = self.config.n_mfcc_config.get(self.n_mfcc)
        if config_entry is None:
            raise ValueError(f"No configuration found for n_mfcc={self.n_mfcc}")
        train_path = config_entry.train_path
        validation_path = config_entry.validation_path
        test_path = config_entry.test_path
        return train_path, validation_path, test_path

    def test_features(self, X_train, X_val, y_train, y_val, features, statistics):
        results = []
        for model in self.models:
            combined_features = []
            for f in features:
                for s in statistics:
                    combined_features.append(f'{f}_{s}')

            features_to_use = []
            for f in combined_features:
                cols_starting_with_feature = [col for col in X_train.columns if col.startswith(f)]
                features_to_use.extend(cols_starting_with_feature)

            X_train_selected = X_train[features_to_use]
            X_val_selected = X_val[features_to_use]

            model.fit(X_train_selected, y_train)
            y_val_pred = model.predict(X_val_selected)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            results.append({
                'model': type(model).__name__,
                'accuracy': val_accuracy,
                'features': features,
                'statistics': statistics
            })

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_features = results[-1]
                logger.debug(f"Best accuracy: {self.best_accuracy}, Best features: {self.best_features}")

        return results

    def find_best_features(self):
        train_path, validation_path, _ = self.get_n_mfcc_paths()
        X_train, y_train = DataUtils.load_data(train_path)
        X_val, y_val = DataUtils.load_data(validation_path)
        statistic_combos = self.generate_feature_combinations(self.statistics)

        for features in self.features_combo:
            for statistics in statistic_combos:
                logger.info(f"Testing features: {features}, statistics: {statistics}")
                self.test_features(X_train, X_val, y_train, y_val, features, statistics)

        return self.best_features