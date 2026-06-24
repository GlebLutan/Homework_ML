import numpy as np
import typing


class MinMaxScaler():
    def __init__(self) -> None:
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, data: np.ndarray) -> None:
        self.data_min_ = np.min(data, axis=0)
        self.data_max_ = np.max(data, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            self.fit(data)
        self.data_range_[self.data_range_ == 0] = 1
        return (data - self.data_min_) / self.data_range_



class StandardScaler():
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> None:
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            self.fit(data)
        self.std_[self.std_ == 0] = 1
        return (data - self.mean_) / self.std_
