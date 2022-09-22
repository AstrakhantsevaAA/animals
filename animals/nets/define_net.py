import torchvision.models as models
from torch import nn

from animals.config import torch_config


def define_net(freeze_grads: bool = False):
    model = models.resnet50(pretrained=True)

    if freeze_grads:
        for params in model.parameters():
            params.requires_grad_ = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(
            100, 8
        ),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    model.to(torch_config.device)

    return model
