import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


class DataUtils:
    @staticmethod
    def load_csv(file_path):
        """
        Đọc dữ liệu từ file CSV.

        Args:
            file_path (str): Đường dẫn đến file CSV.

        Returns:
            pd.DataFrame: Dữ liệu đã đọc dưới dạng DataFrame của pandas.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def get_n_mfcc_paths(n_mfcc, config):
        """
        Lấy đường dẫn cho dữ liệu huấn luyện, xác thực và kiểm tra dựa trên số lượng MFCC.

        Args:
            n_mfcc (int): Số lượng MFCC.
            config (object): Đối tượng cấu hình chứa các đường dẫn.

        Returns:
            tuple: Các đường dẫn cho dữ liệu huấn luyện, xác thực và kiểm tra.
        """
        train_path = config.n_mfcc_config[n_mfcc].train_path
        validation_path = config.n_mfcc_config[n_mfcc].validation_path
        test_path = config.n_mfcc_config[n_mfcc].test_path
        return train_path, validation_path, test_path

    @staticmethod
    def scale_features(data):
        """
        Chuẩn hóa các đặc trưng bằng StandardScaler.

        Args:
            data (pd.DataFrame): Dữ liệu đầu vào cần chuẩn hóa.

        Returns:
            np.ndarray: Dữ liệu đã được chuẩn hóa.
            StandardScaler: Đối tượng scaler đã được huấn luyện.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    @staticmethod
    def apply_pca(data, n_components):
        """
        Áp dụng PCA để giảm số chiều dữ liệu.

        Args:
            data (np.ndarray): Dữ liệu đầu vào.
            n_components (int): Số lượng thành phần chính cần giữ lại.

        Returns:
            np.ndarray: Dữ liệu đã được biến đổi bằng PCA.
            PCA: Đối tượng PCA đã được huấn luyện.
        """
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data)
        return pca_data, pca

    @staticmethod
    def describe_data(data):
        """
        Trả về thống kê mô tả của dữ liệu.

        Args:
            data (pd.DataFrame): Dữ liệu đầu vào.

        Returns:
            pd.DataFrame: Thống kê mô tả của dữ liệu.
        """
        return data.describe()

    @staticmethod
    def balance_data(X, y):
        """
        Cân bằng dữ liệu bằng cách resampling.

        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            y (np.ndarray): Nhãn tương ứng.

        Returns:
            np.ndarray: Dữ liệu đầu vào đã được cân bằng.
            np.ndarray: Nhãn tương ứng đã được cân bằng.
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
    def load_data(file_path, key=None):
        """
        Đọc dữ liệu từ file CSV và tách thành đặc trưng và nhãn.

        Args:
            file_path (str): Đường dẫn đến file CSV.
            key (str, optional): Tiền tố của các cột đặc trưng. Nếu None, sẽ sử dụng tất cả các cột trừ 'label' và 'file_path'.

        Returns:
            tuple: Đặc trưng X và nhãn y.
        """
        data = pd.read_csv(file_path)
        if key is not None:
            key_columns = [col for col in data.columns if col.startswith(key)]
            X = data[key_columns]
        else:
            X = data.drop(columns=['label', 'file_path'])
        y = data['label']
        return X, y

    @staticmethod
    def encode_labels_for_train(y_train, y_val):
        """
        Mã hóa nhãn cho dữ liệu huấn luyện và xác thực.

        Args:
            y_train (np.ndarray): Nhãn huấn luyện.
            y_val (np.ndarray): Nhãn xác thực.

        Returns:
            tuple: Nhãn huấn luyện và xác thực đã được mã hóa OneHot.
        """
        encoder = OneHotEncoder()
        y_train_encoded = encoder.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
        y_val_encoded = encoder.fit_transform(np.array(y_val).reshape(-1, 1)).toarray()
        return y_train_encoded, y_val_encoded

    @staticmethod
    def encode_labels_for_test(y_test):
        """
        Mã hóa nhãn cho dữ liệu kiểm tra.

        Args:
            y_test (np.ndarray): Nhãn kiểm tra.

        Returns:
            np.ndarray: Nhãn kiểm tra đã được mã hóa OneHot.
        """
        encoder = OneHotEncoder()
        y_test_encoded = encoder.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()
        return y_test_encoded

    @staticmethod
    def preprocess_data(X_train, X_val):
        """
        Chuẩn hóa dữ liệu huấn luyện và xác thực.

        Args:
            X_train (np.ndarray): Đặc trưng huấn luyện.
            X_val (np.ndarray): Đặc trưng xác thực.

        Returns:
            tuple: Đặc trưng huấn luyện và xác thực đã được chuẩn hóa, và đối tượng scaler đã được huấn luyện.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    @staticmethod
    def prepare_data_for_train_model(train_file_path, val_file_path, key=None):
        """
        Chuẩn bị dữ liệu cho mô hình huấn luyện từ file huấn luyện và xác thực.

        Args:
            train_file_path (str): Đường dẫn đến file CSV chứa dữ liệu huấn luyện.
            val_file_path (str): Đường dẫn đến file CSV chứa dữ liệu xác thực.
            key (str, optional): Tiền tố của các cột đặc trưng. Nếu None, sẽ sử dụng tất cả các cột trừ 'label' và 'file_path'.

        Returns:
            tuple: Đặc trưng và nhãn của dữ liệu huấn luyện và xác thực, và đối tượng scaler đã được huấn luyện.
        """
        X_train, y_train = DataUtils.load_data(train_file_path, key)
        X_val, y_val = DataUtils.load_data(val_file_path, key)

        X_train = X_train.loc[:, ~X_train.columns.str.contains('^Unnamed')]
        X_val = X_val.loc[:, ~X_val.columns.str.contains('^Unnamed')]

        X_train_scaled, X_val_scaled, scaler = DataUtils.preprocess_data(X_train, X_val)

        y_train_encoded, y_val_encoded = DataUtils.encode_labels_for_train(y_train, y_val)

        x_train_cnn = np.expand_dims(X_train_scaled, axis=2)
        x_val_cnn = np.expand_dims(X_val_scaled, axis=2)

        return x_train_cnn, y_train_encoded, x_val_cnn, y_val_encoded, scaler

    @staticmethod
    def prepare_data_for_test_model(test_file_path, scaler, key=None):
        """
        Chuẩn bị dữ liệu để đánh giá mô hình từ file test.

        Args:
            test_file_path (str): Đường dẫn đến file CSV chứa dữ liệu test.
            scaler (StandardScaler): Đối tượng scaler được dùng để chuẩn hóa dữ liệu test.
            key (str, optional): Tiền tố của các cột đặc trưng. Nếu None, sẽ sử dụng tất cả các cột trừ 'label' và 'file_path'.

        Returns:
            np.ndarray: Đặc trưng đã chuẩn hóa và mở rộng chiều cho mô hình CNN.
            np.ndarray: Nhãn đã được mã hóa OneHot.
        """
        X_test, y_test = DataUtils.load_data(test_file_path, key)

        X_test = X_test.loc[:, ~X_test.columns.str.contains('^Unnamed')]

        X_test_scaled = scaler.transform(X_test)

        y_test_encoded = DataUtils.encode_labels_for_test(y_test)

        x_test_cnn = np.expand_dims(X_test_scaled, axis=2)

        return x_test_cnn, y_test_encoded
    
    # @staticmethod
    # def prepare_data_for_rf_model(train_file_path, test_file_path, val_file_path):
    #     # Đọc dữ liệu từ các file vào DataFrame
    #     train_data = pd.read_csv(train_file_path)
    #     test_data = pd.read_csv(test_file_path)
    #     val_data = pd.read_csv(val_file_path)
        
    #     # Gộp train và validation vào train
    #     train_data = pd.concat([train_data, val_data], ignore_index=True)
        
    #     # Chia features và labels
    #     X_train = train_data.drop(columns=['label', 'file_path'])
    #     y_train = train_data['label']
    #     X_test = test_data.drop(columns=['label', 'file_path'])
    #     y_test = test_data['label']
        
    #     # Chuẩn hóa dữ liệu bằng StandardScaler
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.transform(X_test)
        
    #     return X_train, X_test, y_train, y_test