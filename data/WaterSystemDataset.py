import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WaterSystemDataset(Dataset):
    ''' Dataset class generator on SWaT/WADI dataset.
    Args:
        - path: <str> preprocessed dataset numpy file path
        - feature_idx: <list<int>> choose features you want to use by index
        - start_idx: <int> choose period you want to use by index
        - end_idx: <int> choose period you want to use by index
        - windows_size: <int> history length you want to use
        - sliding: <int> history window moving step
    '''

    def __init__(self, path,
                 feature_idx: list,
                 start_idx: int,
                 end_idx: int,
                 window_size: int = 100,
                 sliding:int=1,
                 labels_path = None):
        self.data = np.load(path, allow_pickle=True).take(feature_idx, axis=1)[start_idx:end_idx]
        self.data = torch.Tensor(self.data)
        if labels_path is not None:
            labels = np.load(labels_path)[start_idx:end_idx]
        else:
            labels = None
        self.labels = labels
        self.window_size = window_size
        self.sliding = sliding

    def __len__(self):
        return int((self.data.shape[0] - self.window_size) / self.sliding) - 1

    def __getitem__(self, index):
        '''
        Returns:
            input: <np.array> [num_feature, windows_size]
            output: <np.array> [num_feature]
        '''
        start = index * self.sliding
        end = index * self.sliding + self.window_size

        if self.labels is None:
            return self.data[start:end, :], []  # self.data[end+1, :]
        else:
            window_label = int(np.any(self.labels[start:end]))  # 1, если есть хoтя бы oдна аномалия
            return self.data[start:end, :], window_label