import numpy as np
from typing import Union


def get_windows(
    X: Union[np.ndarray, list],
    window_size: int = 40,
    batch_size: int = 10000,
    stride: int = 1
) -> np.ndarray:
    """
    Разбивает последовательность на перекрывающиеся окна с заданным шагом.

    Args:
        X (np.ndarray or list): Последовательность или массив с временными рядами.
        window_size (int): Размер одного окна.
        batch_size (int): Количество элементов, обрабатываемых за раз.
        stride (int): Шаг между окнами.

    Returns:
        np.ndarray: Массив с формой (кол-во окон, window_size, ...), готовый для подачи в модель.
    """
    X = np.asarray(X)
    result = []

    for i in range(0, len(X) - window_size + 1, batch_size):
        end = min(i + batch_size, len(X) - window_size + 1)
        batch = [X[j:j + window_size] for j in range(i, end, stride)]
        result.extend(batch)

    return np.array(result)
