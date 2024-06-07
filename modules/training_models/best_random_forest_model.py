import os
import pandas as pd
from loguru import logger
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class ModelTester:
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
        self.best_pipeline = None
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
        train_path = self.config.n_mfcc_config[self.n_mfcc].train_path
        validation_path = self.config.n_mfcc_config[self.n_mfcc].validation_path
        test_path = self.config.n_mfcc_config[self.n_mfcc].test_path
        return train_path, validation_path, test_path

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        X = data.drop(columns=['label', 'file_path'])
        y = data['label']
        return X, y

    def test_models(self, X_train, X_val, y_train, y_val, features, statistics):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for model in self.models:
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            result = {
                'model': type(model).__name__,
                'accuracy': val_accuracy,
                'pipeline_details': {
                    'features': features,
                    'statistics': statistics
                }
            }

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_pipeline = result
                logger.debug(f"Best accuracy: {self.best_accuracy}, Best pipeline: {self.best_pipeline}")

    def find_best_pipeline(self):
        train_path, validation_path, _ = self.get_n_mfcc_paths()
        X_train, y_train = self.load_data(train_path)
        X_val, y_val = self.load_data(validation_path)
        statistic_combo = self.generate_feature_combinations(self.statistics)

        for features in self.features_combo:
            for statistic in statistic_combo:
                combined_features = []
                for f in features:
                    for s in statistic:
                        combined_features.append(f'{f}_{s}')
                
                features_to_use = []
                logger.info(f"Testing pipeline with features: {features}, statistics: {statistic}")
                for f in combined_features:
                    cols_starting_with_feature = [col for col in X_train.columns if col.startswith(f)]
                    features_to_use.extend(cols_starting_with_feature)
                    
                X_train_selected = X_train[features_to_use]
                X_val_selected = X_val[features_to_use]

                self.test_models(X_train_selected, X_val_selected, y_train, y_val, features, statistic)

        return self.best_pipeline