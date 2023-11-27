from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

__DATASET__ = {}

mean, std, data_length, mu = (-581, 490, 1, 0.0194)

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, train=True):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, train=train)


def get_dataloader(dataset: Dataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

@register_dataset(name='aapm')
class AAPMDataset(Dataset):
  def __init__(self, root, train=True):
    super(AAPMDataset, self).__init__()
    self.path = root
    self.len = data_length
    self.train = train
  
  def __getitem__(self, index):
    if self.train:
      x = np.load(self.path + "full_dose/" + str(index + 1) + ".npy")
      x = torch.from_numpy(x).float()
      x = (x - mu) / mu * 1000
      x = (x - mean) / std
      x = torch.unsqueeze(x, 0)
      return x
    else:
      x = np.load(self.path + "full_dose/" + str(index + 1) + ".npy")
      x = torch.from_numpy(x).float()
      x = (x - mu) / mu * 1000
      x = (x - mean) / std
      x = torch.unsqueeze(x, 0)
      y = np.load(self.path + "quarter_dose/" + str(index + 1) + ".npy")
      y = torch.from_numpy(y).float()
      y = (y - mu) / mu * 1000
      y = (y - mean) / std
      y = torch.unsqueeze(y, 0)
      return (x, y)

  def __len__(self):
    return self.len 