from typing import Optional

import numpy as np
import torch
from albumentations import (
    Blur,
    CoarseDropout,
    Compose,
    Downscale,
    HorizontalFlip,
    OneOf,
    RandomRotate90,
    Rotate,
    ShiftScaleRotate,
    VerticalFlip,
)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(
            self,
            data_dir,
            x_df,
            y_df=None,
            augmentations_intensity: float = 0.0,
            species_labels: Optional[tuple] = None,
    ):
        self.data_dir = data_dir
        self.data = x_df
        self.label = y_df
        self.species_labels = species_labels
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        if augmentations_intensity > 0:
            self.augmentation = Compose(
                [
                    OneOf(
                        [HorizontalFlip(), VerticalFlip()],
                        p=0.1,
                    ),
                    OneOf(
                        [Rotate(p=1.0, limit=30), RandomRotate90(p=1.0)],
                        p=0.1,
                    ),
                    ShiftScaleRotate(p=1.0, rotate_limit=30),
                    Blur(blur_limit=3, p=0.1),
                    CoarseDropout(max_height=7, max_width=7, p=0.1),
                    Downscale(
                        scale_min=0.6,
                        scale_max=0.9,
                        p=0.7,
                    ),
                ],
                p=augmentations_intensity,
            )
        else:
            self.augmentation = None

    def __getitem__(self, index):
        image = Image.open(self.data_dir / self.data.iloc[index]["filepath"]).convert(
            "RGB"
        )
        if self.augmentation is not None:
            image = self.augmentation(image=np.array(image))["image"]
            image = self.transform(Image.fromarray(image))
        else:
            image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)
