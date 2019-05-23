import numpy as np
import torch
import gym
import random
from collections import namedtuple
from collections import defaultdict
from agent.ddpg import Ddpg
from agent.simple_network import SimpleNet
from agent.simple_network import DoubleInputNet
from agent.random_process import OrnsteinUhlenbeckProcess
from gym_torcs import TorcsEnv
from agent.PriotrizedReplayBuffer import PrioirtyBuffer


def train(device):
    #env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace28.xml")
    env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")
    insize = env.observation_space.shape[0]
    outsize = env.action_space.shape[0]
    train_indicator = 1
    hyperparams = {
                "lrvalue": 0.001,
                "lrpolicy": 0.001,
                "gamma": 0.985,
                "episodes": 30000,
                "buffersize": 2**13,#300000,
                "tau": 0.01,
                "batchsize": 32,
                "start_sigma": 0.9,
                "end_sigma": 0.1,
                "theta": 0.15,
                "maxlength": 1000,
                "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    datalog = defaultdict(list)
    
    valuenet = DoubleInputNet(insize, outsize, 1)
    policynet = SimpleNet(insize, outsize, activation=torch.nn.functional.tanh)
    
    agent = Ddpg(valuenet, policynet, buffer=PrioirtyBuffer(hyprm.buffersize))#buffersize=hyprm.buffersize)
    agent.to(device)
    
    #load model
    print("loading model")
    try:
        valuenet.load_state_dict(torch.load('valuemodel.pth'))
        valuenet.eval()
        policynet.load_state_dict(torch.load('policymodel.pth'))
        policynet.eval()
        print("model load successfully")
    except:
        print("cannot find the model")

    for eps in range(hyprm.episodes):
        state = env.reset(relaunch=eps%100 == 0, render=True, sampletrack=True)
        epsisode_reward = 0
        episode_value = 0
        sigma = (hyprm.start_sigma-hyprm.end_sigma)*(max(0, 1-eps/hyprm.episodes)) + hyprm.end_sigma
        randomprocess = OrnsteinUhlenbeckProcess(hyprm.theta, sigma, outsize)
        for i in range(hyprm.maxlength):
            torch_state = agent._totorch(state, torch.float32).view(1, -1)
            action, value = agent.act(torch_state)
            action = train_indicator * randomprocess.noise() + action.to("cpu").squeeze()
            action.clamp_(-1, 1)
            action[1] = (action[1]+1)/2
            next_state, reward, done, _ = env.step(np.concatenate([action[:2], [-1]]))
            agent.push(state, action, reward, next_state, done,hyprm.gamma)
            epsisode_reward += reward

            if len(agent.buffer) > hyprm.batchsize:
                value_loss, policy_loss = agent.update(hyprm.gamma, hyprm.batchsize, hyprm.tau, hyprm.lrvalue, hyprm.lrpolicy, hyprm.clipgrad)
                if random.uniform(0, 1) < 0.01:
                    datalog["td error"].append(value_loss)
                    datalog["average policy value"].append(policy_loss)

            if done:
                break
            state = next_state
        datalog["epsiode length"].append(i)
        datalog["total reward"].append(epsisode_reward)

        avearage_reward = torch.mean(torch.tensor(datalog["total reward"][-20:])).item()
        print("\r Process percentage: {:2.1f}%, Average reward: {:2.3f}".format(eps/hyprm.episodes*100, avearage_reward), end="", flush=True)

        if np.mod(eps, 100) == 0:
            if (train_indicator):
                print("saving model")
                torch.save(valuenet.state_dict(), 'valuemodel.pth')
                torch.save(policynet.state_dict(), 'policymodel.pth')

if __name__ == "__main__":
    train( "cuda" if torch.cuda.is_available() else 'cpu')
    