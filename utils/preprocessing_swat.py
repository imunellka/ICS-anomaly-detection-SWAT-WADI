import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def preprocess_swat_data(
    train_path: str,
    test_path: str,
    validation_split: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает и обрабатывает данные SWaT, масштабирует их и сохраняет в формате .npy.

    Args:
        train_path (str): Путь к файлу с обучающими данными (нормальные).
        test_path (str): Путь к файлу с тестовыми данными (с аномалиями).
        validation_split (float): Доля данных для валидации (от тестовой выборки).

    Returns:
        Tuple: Массивы (train_scaled, validation_scaled, test_scaled, validation_labels, test_labels)
    """

    df = pd.read_excel(train_path)
    df.columns = df.iloc[0, :]
    df = df.drop(0)
    df.columns = df.columns.str.replace(' ', '')
    df = df.drop(["Timestamp", "Normal/Attack"], axis=1)

    test_val = pd.read_excel(test_path)
    test_val.columns = test_val.iloc[0, :]
    test_val = test_val.drop(0)
    test_val.columns = test_val.columns.str.replace(' ', '')

    # Получение меток аномалий (1 — аномалия, 0 — норма)
    labels = [float(label != 'Normal') for label in test_val["Normal/Attack"].values]

    val_len = int(validation_split * len(labels))
    validation = test_val.iloc[:val_len].drop(["Timestamp", "Normal/Attack"], axis=1)
    test = test_val.iloc[val_len:].drop(["Timestamp", "Normal/Attack"], axis=1)

    validation_labels = labels[:val_len]
    test_labels = labels[val_len:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df)
    validation_scaled = scaler.transform(validation)
    test_scaled = scaler.transform(test)

    # Сохранение
    np.save('labels_test.npy', np.array(test_labels))
    np.save('labels_validation.npy', np.array(validation_labels))
    np.save('train_swat.npy', train_scaled)
    np.save('test_swat.npy', test_scaled)
    np.save('validation_swat.npy', validation_scaled)

    return train_scaled, validation_scaled, test_scaled, np.array(validation_labels), np.array(test_labels)
