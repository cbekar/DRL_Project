import numpy as np
import torch
import gym
import random
from collections import namedtuple
from collections import defaultdict
from collections import deque
from agent.ddpg import Ddpg
from agent.simple_network import SimpleNet
from agent.simple_network import DoubleInputNet
from agent.random_process import OrnsteinUhlenbeckProcess
from gym_torcs import TorcsEnv
from agent.PrioritizedReplayBuffer import PriorityBuffer
import json
from t1_agent import Agent
# import sys

## To able to run this code you should change gym_torcs.py termination parameters in reward part.
def test(device):
    # env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace28.xml")
    env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")    
    datalog = defaultdict(list)
    agent = Agent()
    for eps in range(20):
        state = env.reset(relaunch=eps%100 == 0, render=True, sampletrack=True)
        epsisode_reward = 0
        episode_value = 0
        i=0
        while(1):
            i+=1
            action = agent.act(state)
            # print(action[0].item())
            datalog["steer"].append(action[0].item())
            datalog["accel"].append(action[1].item())
            datalog["break"].append(action[2].item())
            next_state, reward, done, _ = env.step(np.concatenate([action[:2], [-1]]))
            # agent.push(state, action, reward, next_state, done,hyprm.gamma)
            epsisode_reward += reward
            if done:
                break
            state = next_state
            datalog["epsiode length"].append(i)
            datalog["reward"].append(reward)
                              
            print("saving log")
            with open('data_log_test.json', 'w') as f:
                json.dump(datalog,f)

        avearage_reward = torch.mean(torch.tensor(datalog["total reward"][-20:])).item()
        print("\r Process percentage: {:2.1f}%, Average reward: {:2.3f}".format(eps, avearage_reward), end="", flush=True)
            
            

        

if __name__ == "__main__":
    
    test( "cuda" if torch.cuda.is_available() else 'cpu')
    
