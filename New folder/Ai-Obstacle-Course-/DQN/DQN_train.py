import torch
import numpy as np
from DQN_model import DQN1


epsilon = 1
epsilon_decay = 0.9
epsilon_min = 0.05
lr = 0.0001
discount = 0.9
num_episodes = 100000

network1 = DQN1()
network2 = DQN1()

optimizer1 = torch.optim.Adam(network1.parameters(), lr)
optimizer2 = torch.optim.Adam(network2.parameters(), lr)
criterion = torch.nn.MSELoss()
