import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

class DataUtils:
    @staticmethod
    def load_csv(file_path):
        """
        Đọc dữ liệu từ file CSV.
        """
        return pd.read_csv(file_path)
    
    @staticmethod
    def scale_features(data):
        """
        Chuẩn hóa các đặc trưng.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler
    
    @staticmethod
    def apply_pca(data, n_components):
        """
        Áp dụng PCA để giảm số chiều dữ liệu.
        """
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data)
        return pca_data, pca
    
    @staticmethod
    def describe_data(data):
        """
        Trả về thống kê mô tả của dữ liệu.
        """
        return data.describe()
    
    @staticmethod
    def balance_data(X, y):
        """
        Cân bằng dữ liệu bằng cách resampling.
        """
        from sklearn.utils import resample
        from collections import Counter
        
        # Đếm số lượng mẫu cho mỗi lớp
        counter = Counter(y)
        
        # Tìm lớp có ít mẫu nhất và lớp có nhiều mẫu nhất
        min_samples = min(counter.values())
        max_samples = max(counter.values())
        
        # Khởi tạo danh sách lưu trữ các chỉ số của các mẫu được giữ lại
        indices_to_keep = []
        indices_to_add = []
        
        for label in counter.keys():
            class_indices = np.where(y == label)[0]
            
            if len(class_indices) > min_samples:
                np.random.shuffle(class_indices)
                indices_to_keep.extend(class_indices[:min_samples])
            elif len(class_indices) < max_samples:
                indices_to_keep.extend(class_indices)
                indices_to_add.extend(resample(class_indices, replace=True, n_samples=max_samples - len(class_indices), random_state=42))
        
        indices_to_keep.extend(indices_to_add)
        X_balanced = X[indices_to_keep]
        y_balanced = y[indices_to_keep]
        
        return X_balanced, y_balanced
    
    @staticmethod
    def load_data(file_path):
        data = pd.read_csv(file_path)
        X = data.drop(columns=['label', 'file_path'])
        y = data['label']
        return X, y
    
    def encode_labels(y_train, y_val, y_test):
        encoder = OneHotEncoder()
        y_train_encoded = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
        y_val_encoded = encoder.fit_transform(np.array(y_val).reshape(-1,1)).toarray()
        y_test_encoded = encoder.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    @staticmethod
    def preprocess_data(X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    @staticmethod
    def prepare_data(train_file_path, test_file_path, val_file_path):
        X_train, y_train = DataUtils.load_data(train_file_path)
        X_test, y_test = DataUtils.load_data(test_file_path)
        X_val, y_val = DataUtils.load_data(val_file_path)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = DataUtils.preprocess_data(X_train, X_val, X_test)
        
        y_train, y_val, y_test = DataUtils.encode_labels(y_train, y_val, y_test)
        
        x_traincnn = np.expand_dims(X_train_scaled, axis=2)
        x_valcnn = np.expand_dims(X_val_scaled, axis=2)
        x_testcnn = np.expand_dims(X_test_scaled, axis=2)
        
        return x_traincnn, y_train, x_valcnn, y_val, x_testcnn, y_test, scaler