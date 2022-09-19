from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from animals.data_utils.dataset import ImagesDataset


def train_model():
    data_dir = Path("/home/alenaastrakhantseva/PycharmProjects/animals/data/raw")
    train_features = pd.read_csv(data_dir / "train_features.csv", index_col="id")
    test_features = pd.read_csv(data_dir / "test_features.csv", index_col="id")
    train_labels = pd.read_csv(data_dir / "train_labels.csv", index_col="id")

    frac = 0.5

    y = train_labels.sample(frac=frac, random_state=1)
    x = train_features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(
        x, y, stratify=y, test_size=0.25
    )

    train_dataset = ImagesDataset(data_dir, x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(
            100, 8
        ),  # final dense layer outputs 8-dim corresponding to our target classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 1

    tracking_loss = {}

    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch}")

        # iterate through the dataloader batches. tqdm keeps track of progress.
        for batch_n, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            # 1) zero out the parameter gradients so that gradients from previous batches are not used in this step
            optimizer.zero_grad()

            # 2) run the foward step on this batch of images
            outputs = model(batch["image"])

            # 3) compute the loss
            loss = criterion(outputs, batch["label"])
            # let's keep track of the loss by epoch and batch
            tracking_loss[(epoch, batch_n)] = float(loss)

            # 4) compute our gradients
            loss.backward()
            # update our weights
            optimizer.step()

    torch.save(model, "model.pth")


if __name__ == "__main__":
    train_model()
