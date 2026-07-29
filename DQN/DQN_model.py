import torch
import torch.nn as nn

class DQN1(nn.Module):
  def __init__(self):
    super().__init__()
    DQS_seq = nn.Sequential()
    DQS_seq.append(nn.Linear(96, 128))
    DQS_seq.append(nn.ReLU())
    DQS_seq.append(nn.Linear(128, 64))
    DQS_seq.append(nn.ReLU())
    DQS_seq.append(nn.Linear(64, 16))
    DQS_seq.append(nn.ReLU())
    DQS_seq.append(nn.Linear(16, 9))
  def forward(self, x):
    return DQS_seq1(x)

