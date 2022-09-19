import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
import torch.optim as optim
from clearml import Task
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from animals.data_utils.dataset import ImagesDataset
from animals.config import torch_config, system_config
from animals.nets.define_net import define_net


@hydra.main(version_base=None, config_path=".", config_name="config")
def train_model(cfg: DictConfig):
    task = Task.init(project_name="my project", task_name="my task") if cfg.train.log_clearml else None
    logger = None if task is None else task.get_logger()

    train_features = pd.read_csv(system_config.raw_data_dir / "train_features.csv", index_col="id")
    test_features = pd.read_csv(system_config.raw_data_dir / "test_features.csv", index_col="id")
    train_labels = pd.read_csv(system_config.raw_data_dir / "train_labels.csv", index_col="id")

    frac = 0.5

    y = train_labels.sample(frac=frac, random_state=1)
    x = train_features.loc[y.index].filepath.to_frame()

    x_train, x_eval, y_train, y_eval = train_test_split(
        x, y, stratify=y, test_size=0.25
    )

    train_dataset = ImagesDataset(system_config.raw_data_dir, x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    model = define_net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.to(torch_config.device)

    iters = len(train_dataloader)
    running_loss = 0

    for epoch in cfg.train.epochs:
        print(f"Starting epoch {epoch}")
        for batch_n, batch in tqdm(enumerate(train_dataloader), total=iters):
            optimizer.zero_grad()
            outputs = model(batch["image"].to(torch_config.device))
            loss = criterion(outputs, batch["label"].to(torch_config.device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        logger.report_scalar(f"Loss", "train", iteration=epoch, value=running_loss / iters)
    torch.save(model, "model.pth")

    if logger is not None:
        logger.flush()


if __name__ == "__main__":
    train_model()
