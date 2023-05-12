import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.transforms as transforms

import os
import pickle
import random

import sys
import visarl.util.constants as constants
import einops


class MetaworldData(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, args, dataset_path, transform=None, stage="train"):
        print(f"loading dataset from: {dataset_path}")
        dataset = pickle.load(open(dataset_path, "rb"))

        for k, v in dataset.items():
            dataset[k] = np.array(dataset[k])

        image = dataset["frames"][:, 0]
        self.image = torch.from_numpy(image).float()

        if "saliency" in dataset.keys():
            # stupid, only has saliency for current frame not the next frame
            saliency = dataset["saliency"]
            self.saliency = torch.from_numpy(saliency)
        else:
            self.saliency = torch.zeros_like(self.image)

        print(self.image.shape, self.saliency.shape)

        self.transform = transform
        self.stage = stage

        self.saliency_transform = transforms.Compose([])

    def __getitem__(self, index):
        image, saliency = self.transform(self.image[index]), self.saliency_transform(
            self.saliency[index]
        )
        hflip = random.random() < 0.5
        if hflip and self.stage == "train":
            image = transforms.RandomHorizontalFlip(1.0)(image)
            saliency = transforms.RandomHorizontalFlip(1.0)(saliency)
        return image, saliency

    def __len__(self):
        return self.image.shape[0]


class RealWorldData(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, args, dataset_path, transform=None, stage="train"):
        print(f"loading dataset from: {dataset_path}")
        dataset = pickle.load(open(dataset_path, "rb"))

        for k, v in dataset.items():
            dataset[k] = np.array(dataset[k])

        images = dataset["frames"]
        self.images = torch.from_numpy(images).float()

        if self.images.shape[1] != 3:
            self.images = einops.rearrange(self.images, "b h w c -> b c h w")

        if "saliency" in dataset.keys():
            # stupid, only has saliency for current frame not the next frame
            saliency = dataset["saliency"]
            self.saliency = torch.from_numpy(saliency)
        else:
            self.saliency = torch.zeros_like(self.image)

        if len(self.saliency.shape) == 3:
            self.saliency = einops.rearrange(self.saliency, "b h w -> b 1 h w")

        print(self.images.shape, self.saliency.shape)

        # Repeat images and saliency
        # self.images = einops.repeat(
        #     self.images, "b c h w -> (repeat b) c h w", repeat=20
        # )
        # self.saliency = einops.repeat(
        #     self.saliency, "b c h w -> (repeat b) c h w", repeat=20
        # )

        self.transform = transform
        self.stage = stage
        self.saliency_transform = transforms.Compose([])

    def __getitem__(self, index):
        index = index % self.images.shape[0]
        image, saliency = self.transform(self.images[index]), self.saliency_transform(
            self.saliency[index]
        )
        hflip = random.random() < 0.5
        if hflip and self.stage == "train":
            image = transforms.RandomHorizontalFlip(1.0)(image)
            saliency = transforms.RandomHorizontalFlip(1.0)(saliency)
        return image, saliency

    def __len__(self):
        return self.images.shape[0] * 20


if __name__ == "__main__":
    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            # transforms.RandomResizedCrop(
            #     224, scale=(0.2, 1.0), interpolation=3
            # ),  # 3 is bicubic
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            lambda x: x / 255.0,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    args = AttrDict(task_name="reach-v2")
    data_path = "/data/anthony/visual_saliency_rl/replay_buffers/reach-v2_step_500000_with_saliency.pkl"
    data = MetaworldData(args, data_path, transform_train)
    print(data[0])
    import ipdb

    ipdb.set_trace()
