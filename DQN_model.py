import torch
import torch.nn as nn

class DQN1(nn.Module):
  def __init__(self):
    super().__init__(self)
    DQS_seq1 = nn.Sequential()
    DQS_seq1.append(nn.Linear(96, 128))
    DQS_seq1.append(nn.ReLU())
    DQS_seq1.append(nn.Linear(128, 64))
    DQS_seq1.append(nn.ReLU())
    DQS_seq1.append(nn.Linear(64, 16))
    DQS_seq1.append(nn.ReLU())
    DQS_seq1.append(nn.Linear(16, 9))
  def forward(self, x):
    return DQS_seq1(x)

class DQS2(nn.Module):
  def __init(self):
    super().__init__(self)
    DQS_seq2 = nn.Sequential()
    DQS_seq2.append(nn.Linear(96, 128))
    DQS_seq2.append(nn.ReLU())
    DQS_seq2.append(nn.Linear(128, 64))
    DQS_seq2.append(nn.ReLU())
    DQS_seq2.append(nn.Linear(64, 16))
    DQS_seq2.append(nn.ReLU())
    DQS_seq2.append(nn.Linear(16, 9))
  def forward(self, x):
    return DQS_seq2(x)
    
