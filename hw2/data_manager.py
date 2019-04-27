'''
data_manager.py

A file that loads saved features and convert them into PyTorch DataLoader.
'''
import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader


# Class based on PyTorch Dataset
class GTZANDataset(Dataset):
  def __init__(self, x, y, hparams, aug=False):
    self.x = x
    self.y = y
    self.hparams = hparams
    self.aug = aug

  def __getitem__(self, index):
    if self.aug and (self.hparams.aug_size > 0):
      x = self.x[index].copy()

      # augment time
      i = np.random.choice(self.hparams.time_size - self.hparams.aug_size)
      x[:, i:i + self.hparams.aug_size, :] = self.hparams.mask_value

      i = np.random.choice(self.hparams.time_size - self.hparams.aug_size)
      x[:, i:i + self.hparams.aug_size, :] = self.hparams.mask_value

      # augment frequency
      i = np.random.choice(self.hparams.num_mels - self.hparams.aug_size)
      x[:, :, i:i + self.hparams.aug_size] = self.hparams.mask_value

      i = np.random.choice(self.hparams.num_mels - self.hparams.aug_size)
      x[:, :, i:i + self.hparams.aug_size] = self.hparams.mask_value

      return x, self.y[index]
    else:
      return self.x[index], self.y[index]

  def __len__(self):
    return self.x.shape[0]


# Function to get genre index for the given file
def get_label(file_name, hparams):
  genre = file_name.split('/')[-2]
  label = hparams.genres.index(genre)

  return label


# Function for loading entire data from given dataset and return numpy array
def load_dataset(set_name, hparams):
  x = []
  y = []

  dataset_path = os.path.join(hparams.feature_path, set_name)
  for path in sorted(glob(f'{dataset_path}/*/*.npy')):
    data = np.load(path)
    label = get_label(path, hparams)
    x.append(data)
    y.append(label)

  x = np.stack(x)
  y = np.stack(y)

  return x, y


# Function to load numpy data and normalize, it returns dataloader for train, valid, test
def get_dataloader(hparams):
  x_train, y_train = load_dataset('train', hparams)
  x_valid, y_valid = load_dataset('valid', hparams)
  x_test, y_test = load_dataset('test', hparams)

  x_train = np.expand_dims(x_train, 1)
  x_valid = np.expand_dims(x_valid, 1)
  x_test = np.expand_dims(x_test, 1)

  # TODO:
  x_train = np.concatenate([x_train, x_valid])
  y_train = np.concatenate([y_train, y_valid])

  mean = np.mean(x_train)
  std = np.std(x_train)
  x_train = (x_train - mean) / std
  x_valid = (x_valid - mean) / std
  x_test = (x_test - mean) / std

  train_set = GTZANDataset(x_train, y_train, hparams, aug=True)
  valid_set = GTZANDataset(x_valid, y_valid, hparams, aug=False)
  test_set = GTZANDataset(x_test, y_test, hparams, aug=False)

  train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
  valid_loader = DataLoader(valid_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)
  test_loader = DataLoader(test_set, batch_size=hparams.batch_size // 10 * 10, shuffle=False, drop_last=False)

  return train_loader, valid_loader, test_loader
