'''
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import data_manager
from collections import defaultdict
from hparams import hparams
from densenet import DenseNet
# from torchvision.models import ResNet
from resnet import ResNet


# Wrapper class to run PyTorch model
class Runner(object):
  def __init__(self, hparams):
    # self.model = models.Baseline(hparams)
    # self.model = ResNet([2, 2, 2, 2], num_classes=len(hparams.genres), zero_init_residual=True)
    self.model = DenseNet(growth_rate=16, block_config=(4, 4, 4), drop_rate=hparams.drop_rate,
                          num_classes=len(hparams.genres))
    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    self.scheduler = StepLR(self.optimizer, step_size=hparams.lr_step, gamma=hparams.factor)
    self.learning_rate = hparams.learning_rate
    self.stopping_rate = hparams.stopping_rate
    self.device = torch.device("cpu")

    if hparams.device > 0:
      torch.cuda.set_device(hparams.device - 1)
      self.model.cuda(hparams.device - 1)
      self.criterion.cuda(hparams.device - 1)
      self.device = torch.device("cuda:" + str(hparams.device - 1))

  # Accuracy function works like loss function in PyTorch
  def accuracy(self, source, target):
    source = source.max(1)[1].long().cpu()
    target = target.cpu()
    correct = (source == target).sum().item()

    return correct / float(source.size(0))

  # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
  def run(self, dataloader, mode='train'):
    self.model.train() if mode is 'train' else self.model.eval()

    epoch_loss = 0
    epoch_acc = 0
    for batch, (x, y) in enumerate(dataloader):
      x = x.to(self.device)
      y = y.to(self.device)

      prediction = self.model(x)
      if mode == 'test':
        prediction = prediction.reshape(-1, 10, prediction.shape[1])
        prediction = prediction.mean(dim=1)
        y = y[torch.arange(0, len(y), 10)]
      loss = self.criterion(prediction, y)
      acc = self.accuracy(prediction, y)

      if mode is 'train':
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

      epoch_loss += prediction.size(0) * loss.item()
      epoch_acc += prediction.size(0) * acc

    epoch_loss = epoch_loss / len(dataloader.dataset)
    epoch_acc = epoch_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc

  # Early stopping function for given validation loss
  def step_scheduler(self, epoch):
    # self.scheduler.step(loss, epoch)
    self.scheduler.step(epoch)
    self.learning_rate = self.optimizer.param_groups[0]['lr']
    stop = self.learning_rate < self.stopping_rate

    return stop


def device_name(device):
  if device == 0:
    device_name = 'CPU'
  else:
    device_name = 'GPU:' + str(device - 1)

  return device_name


def main():
  train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
  runner = Runner(hparams)

  print('Training on ' + device_name(hparams.device))
  for epoch in range(hparams.num_epochs):
    if epoch % hparams.lr_step == 0:
      runner.optimizer.state = defaultdict(dict)

    train_loss, train_acc = runner.run(train_loader, 'train')
    # valid_loss, valid_acc = runner.run(valid_loader, 'eval')
    test_loss, test_acc = runner.run(test_loader, 'test')

    # print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f] [Test Acc: %.4f]" %
    #       (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc, test_acc * 10))
    print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Test Acc: %.4f]" %
          (epoch + 1, hparams.num_epochs, train_loss, train_acc, test_acc * 10))

    runner.step_scheduler(epoch + 1)

    # if runner.early_stop(valid_loss, epoch + 1):
    #   break

  # test_loss, test_acc = runner.run(test_loader, 'test')
  print("Training Finished")
  print("Test Accuracy: %.2f%%" % (100 * test_acc * 10))


if __name__ == '__main__':
  main()
