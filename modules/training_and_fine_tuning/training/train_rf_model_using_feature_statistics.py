from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

class HyperparameterTuner:
    def __init__(self, config, n_mfcc=20, best_features=None):
        self.config = config
        self.n_mfcc = n_mfcc
        self.best_features = best_features
        self.models = RandomForestClassifier(random_state=42)
        self.best_pipeline = None
        self.best_accuracy = 0

    def get_n_mfcc_paths(self):
        config_entry = self.config.n_mfcc_config.get(self.n_mfcc)
        if config_entry is None:
            raise ValueError(f"No configuration found for n_mfcc={self.n_mfcc}")
        train_path = config_entry.train_path
        validation_path = config_entry.validation_path
        test_path = config_entry.test_path
        return train_path, validation_path, test_path

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        X = data.drop(columns=['label', 'file_path'])
        y = data['label']
        return X, y

    def test_model(self, X_train, X_val, y_train, y_val):
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
            'model__bootstrap': [True, False]
        }

        results = []
        for model in self.models:
            pipeline = Pipeline([
                ('scaler', 'passthrough'),  # Placeholder cho scaler
                ('model', model)
            ])
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_val_pred = best_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            results.append({
                'model': type(model).__name__,
                'accuracy': val_accuracy,
                'pipeline_details': {
                    'features': self.best_features['features'],
                    'statistics': self.best_features['statistics'],
                    'best_params': grid_search.best_params_
                }
            })

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_pipeline = results[-1]
                logger.debug(f"Best accuracy: {self.best_accuracy}, Best pipeline: {self.best_pipeline}")

        return results

    def find_best_pipeline(self):
        train_path, validation_path, _ = self.get_n_mfcc_paths()
        X_train, y_train = self.load_data(train_path)
        X_val, y_val = self.load_data(validation_path)

        combined_features = []
        for f in self.best_features['features']:
            for s in self.best_features['statistics']:
                combined_features.append(f'{f}_{s}')

        features_to_use = []
        for f in combined_features:
            cols_starting_with_feature = [col for col in X_train.columns if col.startswith(f)]
            features_to_use.extend(cols_starting_with_feature)

        X_train_selected = X_train[features_to_use]
        X_val_selected = X_val[features_to_use]

        self.test_model(X_train_selected, X_val_selected, y_train, y_val)

        return self.best_pipeline