import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class SimpleNet(nn.Module):

    def __init__(self, insize, outsize, activation=lambda x: x):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.activation = activation

        self.fc1 = nn.Linear(self.insize, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.steering.weight, 0, 1e-4)
        self.acceleration = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.acceleration.weight, 0, 1e-4)
        self.brake = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.brake.weight, 0, 1e-4)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        out1 = t.tanh(self.steering(x))
        out2 = t.sigmoid(self.acceleration(x))
        out3 = t.sigmoid(self.brake(x))
        out = t.cat((out1, out2, out3), 1) 
        return out

class DoubleInputNet(nn.Module):

    def __init__(self, firstinsize, secondinsize, outsize, activation=lambda x: x):
        super().__init__()
        self.firstinsize = firstinsize
        self.secondinsize = secondinsize
        self.outsize = outsize
        self.activation = activation

        self.fc1_1 = nn.Linear(firstinsize, 64)
        self.fc1_2 = nn.Linear(secondinsize, 64)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, self.outsize)

    def forward(self, firstin, secondin):
        x1 = nn.functional.relu(self.fc1_1(firstin))
        x2 = nn.functional.relu(self.fc1_2(secondin))
        x = t.cat([x1, x2], dim=1)
        x = nn.functional.relu(self.fc2(x))
        return self.activation(self.head(x))