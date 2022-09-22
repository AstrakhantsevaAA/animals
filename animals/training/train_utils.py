from collections import defaultdict
from enum import Enum

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from animals.config import system_config
from animals.data_utils.dataset import ImagesDataset


class Phase(Enum):
    train = "train"
    val = "val"
    test = "test"


def create_dataloader(augmentations_intensity):
    train_features = pd.read_csv(
        system_config.raw_data_dir / "train_features.csv", index_col="id"
    )
    train_labels = pd.read_csv(
        system_config.raw_data_dir / "train_labels.csv", index_col="id"
    )

    species_labels = sorted(train_labels.columns.unique())
    dataloader = defaultdict()
    x_train, x_eval, y_train, y_eval = train_test_split(
        train_features, train_labels, stratify=train_labels, test_size=0.25
    )
    shuffle = True

    for phase in Phase:
        if phase == Phase.val:
            augmentations_intensity, shuffle = 0.0, False
            x_train, y_train = x_eval, y_eval

        dataset = ImagesDataset(
            system_config.raw_data_dir,
            x_train,
            y_train,
            augmentations_intensity,
            tuple(species_labels),
        )
        dataloader[phase] = DataLoader(dataset, batch_size=32, shuffle=shuffle)

    return dataloader
