import torch
import torch.nn as nn

class DQN1(nn.Module):
  def __init__(self):
    super().__init__()
    self.DQS_seq = nn.Sequential()
    self.DQS_seq.append(nn.Linear(96, 128))
    self.DQS_seq.append(nn.ReLU())
    self.DQS_seq.append(nn.Linear(128, 64))
    self.DQS_seq.append(nn.ReLU())
    self.DQS_seq.append(nn.Linear(64, 16))
    self.DQS_seq.append(nn.ReLU())
    self.DQS_seq.append(nn.Linear(16, 9))
  def forward(self, x):
    return self.DQS_seq(x)

