# """ 
# Evaluate an agent based on average number of
# steps to finish an environment
# """
# from tqdm import tqdm
# import os
# import os.path as osp
# import pickle
# import numpy as np
import gym
# import sys
# import time
# from rlpyt.agents.pg.categorical import CategoricalPgAgent
# from Network import *
# from collections import namedtuple

# def simulateAgentFile (agentFile, render=False) :
#     """ Load rlpyt agent from file and simulate  """
#     state_dict = torch.load(
#         agentFile, 
#         map_location=torch.device('cpu')) 
#     agent = CategoricalPgAgent(AcrobotNet)
#     env = gym.make('Acrobot-v1')
#     EnvSpace = namedtuple('EnvSpace', ['action', 'observation'])
#     agent.initialize(EnvSpace(env.action_space, env.observation_space))
#     agent.load_state_dict(state_dict)
#     simulateAgent(agent, render)

# def simulateAgent (agent, render=False) : 
#     """ 
#     Simulate agent on environment till the task
#     is over and return the number of steps taken
#     """
#     env = gym.make('Acrobot-v1', render_mode='human')
#     done = False
#     trajectory = []
#     s = torch.tensor(env.reset()).float()
#     a = torch.tensor(0)
#     r = torch.tensor(0).float()
#     i = 0
#     while not done : 
#         i += 1
#         if render: 
#             env.render()
#             time.sleep(0.05)
#         a = agent.step(s, a, r).action
#         s_, r, done, info = env.step(a.item())
#         s_ = torch.tensor(s_).float()
#         r = torch.tensor(r).float()
#         s = s_
#     if render: 
#         env.render()
#         time.sleep(0.05)
#     env.close()
#     return i

"""
Evaluate an agent based on average number of
steps to finish an environment and save video
"""
"""
Evaluate an agent based on average number of
steps to finish an environment and save video
"""
from tqdm import tqdm
import os
import os.path as osp
import pickle
import numpy as np
# import gymnasium as gym
import sys
import time
import torch
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from Network import *
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

def simulateAgentFile(agentFile):
    """ Load rlpyt agent from file and simulate """
    state_dict = torch.load(
        agentFile, 
        map_location=torch.device('cpu'))
    agent = CategoricalPgAgent(AcrobotNet)
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    EnvSpace = namedtuple('EnvSpace', ['action', 'observation'])
    agent.initialize(EnvSpace(env.action_space, env.observation_space))
    agent.load_state_dict(state_dict)
    return simulateAgent_2(agent, env)

def simulateAgent_2(agent, env):
    """ 
    Simulate agent on environment till the task
    is over and return the number of steps taken
    """
    done = False
    s = torch.tensor(env.reset()[0]).float()
    a = torch.tensor(0)
    r = torch.tensor(0).float()
    i = 0
    
    # Set up the display
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    
    while not done:
        i += 1
        # Render and display current frame
        screen = env.render()
        plt.imshow(screen)
        ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(plt.gcf())
        time.sleep(0.1)  # Add small delay to make animation visible
        
        a = agent.step(s, a, r).action
        s_, r, done, truncated, info = env.step(a.item())
        print(f"State: {s.numpy()} --> {s_.numpy()}")  # Print state transition
        s_ = torch.tensor(s_).float()
        r = torch.tensor(r).float()
        s = s_
    
    # Show final frame
    screen = env.render()
    plt.imshow(screen)
    ipythondisplay.clear_output(wait=True)
    ipythondisplay.display(plt.gcf())
    time.sleep(0.1)
    
    env.close()
    plt.close()
    return i