from typing import Any, Optional

import hydra
import pandas as pd
import torch
import torch.optim as optim
from clearml import Logger, Task
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from animals.config import torch_config
from animals.nets.define_net import define_net
from animals.training.train_utils import Phase, create_dataloader


def train_one_epoch(
        model,
        dataloader: DataLoader,
        optimizer: Any,
        criterion: Any,
        epoch: int,
        logger: Logger,
):
    running_loss = 0
    iters = len(dataloader)
    print(f"Starting epoch {epoch}")
    for batch_n, batch in tqdm(enumerate(dataloader), total=iters):
        optimizer.zero_grad()
        outputs = model(batch["image"].to(torch_config.device))
        loss = criterion(outputs, batch["label"].to(torch_config.device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        logger.report_scalar(
            f"Running_loss",
            "train",
            iteration=(epoch + 1) * batch_n,
            value=running_loss / (batch_n + 1),
        )
    logger.report_scalar(f"Loss", "train", iteration=epoch, value=running_loss / iters)


def evaluation(
        model,
        eval_dataloader: DataLoader,
        criterion: Any,
        epoch: int,
        logger: Optional[Logger],
):
    preds_collector = []
    model.eval()
    running_loss = 0.0
    iters = len(eval_dataloader)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            logits = model.forward(batch["image"].to(torch_config.device))
            preds = nn.functional.softmax(logits, dim=1)
            preds_df = pd.DataFrame(
                preds.cpu().detach().numpy(),
                index=batch["image_id"],
                columns=eval_dataloader.dataset.species_labels,
            )
            preds_collector.append(preds_df)

            loss = criterion(logits, batch["label"].to(torch_config.device))
            running_loss += loss.item()

    eval_preds_df = pd.concat(preds_collector)
    eval_predictions = eval_preds_df.idxmax(axis=1)
    eval_true = eval_dataloader.dataset.label.idxmax(axis=1)
    correct = (eval_predictions == eval_true).sum()
    accuracy = correct / len(eval_predictions)
    logger.report_scalar(f"Accuracy", "eval", iteration=epoch, value=accuracy)
    logger.report_scalar(f"Loss", "eval", iteration=epoch, value=running_loss / iters)


@hydra.main(version_base=None, config_path=".", config_name="config")
def train_model(cfg: DictConfig):
    task = (
        Task.init(project_name="animals", task_name=cfg.train.task_name)
        if cfg.train.log_clearml
        else None
    )
    logger = None if task is None else task.get_logger()

    dataloader = create_dataloader(cfg.train.augmentations_intensity)

    model = define_net(cfg.net.freeze_grads)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(cfg.train.epochs):
        train_one_epoch(
            model, dataloader[Phase.train], optimizer, criterion, epoch, logger
        )
        evaluation(model, dataloader[Phase.val], criterion, epoch, logger)
    torch.save(model, "model.pth")

    if logger is not None:
        logger.flush()


if __name__ == "__main__":
    train_model()
