import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_emotion_distribution(folder_path):
    num_files_in_folders = []

    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)
        num_files = len(os.listdir(subfolder_path))
        num_files_in_folders.append((folder_name, num_files))

    folders, num_files = zip(*num_files_in_folders)
    colors = plt.cm.viridis(np.linspace(0, 1, len(folders)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(folders, num_files, color=colors)
    plt.xlabel('Cảm xúc')
    plt.ylabel('Số lượng mẫu')
    plt.title('Số lượng mẫu của các loại cảm xúc')

    legend_labels = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    plt.legend(legend_labels, folders)

    plt.xticks()
    plt.tight_layout()
    plt.show()

def count_samples_in_subfolders(base_folder):
    emotions = []
    dataset_counts = {
        'train': [],
        'validation': [],
        'test': []
    }

    dataset_types = ['train', 'validation', 'test']

    for dataset_type in dataset_types:
        dataset_type_path = os.path.join(base_folder, dataset_type)
        for emotion_folder in os.listdir(dataset_type_path):
            emotion_path = os.path.join(dataset_type_path, emotion_folder)
            if os.path.isdir(emotion_path):
                if dataset_type == 'train':
                    emotions.append(emotion_folder)
                
                # Count number of files in the emotion folder
                num_files = len(os.listdir(emotion_path))
                dataset_counts[dataset_type].append(num_files)
    
    return emotions, dataset_counts['train'], dataset_counts['validation'], dataset_counts['test']

def plot_sample_distribution(emotions, train_counts, validation_counts, test_counts, figsize=(12, 2)):
    # Ensure all lists have the same length
    max_len = max(len(train_counts), len(validation_counts), len(test_counts))
    train_counts += [0] * (max_len - len(train_counts))
    validation_counts += [0] * (max_len - len(validation_counts))
    test_counts += [0] * (max_len - len(test_counts))
    
    # Convert to pandas DataFrame
    data = {
        'Cảm xúc': emotions,
        'Số lượng (Huấn luyện)': train_counts,
        'Số lượng (Kiểm định)': validation_counts,
        'Số lượng (Kiểm tra)': test_counts
    }
    df = pd.DataFrame(data)
    
    # Display as table
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.table(cellText=df.values,
              colLabels=df.columns,
              cellLoc='center',
              loc='center')
    plt.title('Phân phối mẫu theo cảm xúc sau khi chia dữ liệu', fontweight='bold', fontsize=16, fontname='Arial')
    plt.show()


def count_samples_per_label(base_folder):
    """
    Đếm số lượng mẫu cho mỗi nhãn và loại tập dữ liệu từ các file CSV trong thư mục base_folder.

    Parameters:
    - base_folder (str): Đường dẫn đến thư mục chứa các file CSV.

    Returns:
    - list: Danh sách các tuples (dataset_type, label, count) chứa thông tin số lượng mẫu.
    """
    dataset_types = ['train', 'validation', 'test']
    result = []

    for dataset_type in dataset_types:
        csv_file = os.path.join(base_folder, f"{dataset_type}_file_paths_with_labels.csv")
        df = pd.read_csv(csv_file)

        label_counts = df['label'].value_counts()

        for label, count in label_counts.items():
            result.append((dataset_type, label, count))

    return result

def plot_sample_distribution_table(sample_counts):
    """
    Vẽ bảng phân phối mẫu theo cảm xúc và tập dữ liệu từ danh sách sample_counts.

    Parameters:
    - sample_counts (list): Danh sách các tuples (dataset_type, label, count).

    Returns:
    - None
    """
    # Convert list of tuples to pandas DataFrame
    df = pd.DataFrame(sample_counts, columns=['Dataset Type', 'Emotion', 'Count'])

    # Pivot the DataFrame to have emotions as rows and dataset types as columns
    pivot_df = df.pivot(index='Emotion', columns='Dataset Type', values='Count')

    # Display as table
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=pivot_df.values,
             colLabels=pivot_df.columns,
             rowLabels=pivot_df.index,
             cellLoc='center',
             loc='center')
    plt.title('Phân phối mẫu theo cảm xúc và tập dữ liệu', fontweight='bold', fontsize=16)
    plt.show()