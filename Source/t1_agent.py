import torch
import numpy as np
# import matplotlib.pyplot as plt
from agent.ddpg import Ddpg
from agent.simple_network import SimpleNet
from agent.simple_network import DoubleInputNet
from collections import namedtuple
from collections import defaultdict
from collections import deque
class Agent(object):
    def __init__(self, dim_action=3):
        self.stear_hist = deque([0,0])
        hyperparams = {
            "lowpass_steer": True,
            "alpha": .7 # Low pass filter constant
        }
        HyperParams = namedtuple("HyperParams", hyperparams.keys())
        self.hyprm = HyperParams(**hyperparams)
        self.valuenet = DoubleInputNet(29, dim_action, 1)
        self.policynet = SimpleNet(29, dim_action, \
                        activation=torch.nn.functional.tanh)
        
        self.agent = Ddpg(self.valuenet, self.policynet)
        #load model
        print("loading model")
        
        try:
            self.valuenet.load_state_dict(torch.load('valuemodel.pth'))
            self.valuenet.eval()
            self.policynet.load_state_dict(torch.load('policymodel.pth'))
            self.policynet.eval()
            print("model load successfully")  
        except:
            print("cannot find the model")

    def act(self, ob):
        torch_state = self.agent._totorch(ob, torch.float32).view(1, -1)
        action, _ = self.agent.act(torch_state)
        action[1] = (action[1]+1)/2
        if self.hyprm.lowpass_steer == True:
            action[0] = self.hyprm.alpha*action[0] + \
                (1-self.hyprm.alpha)*self.stear_hist[-1]
            self.stear_hist.append(action[0])
            self.stear_hist.pop()
        return action # random action
