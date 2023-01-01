import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
import datetime
import os
from collections import namedtuple, deque
import random
import datetime

TIME_STAMP = "2022-12-27-09-56-00"
FILE_NAME = "9937_213_dqn"
PATH = f'.\\{FOLDER}\\1-channel-state\\{TIME_STAMP}\{FILE_NAME}.pt'
DQN_1 = True
if DQN_1:
    CHANNEL = 1
    from dqn_1_channel import get_state
    from dqn_1_channel import policy
    from dqn_1_channel import EPSILON_PAIR, GAMMA, FOLDER
else :
    CHANNEL = 4
    from dqn_4_channel import get_state
    from dqn_4_channel import policy
    from dqn_4_channel import EPSILON_PAIR, GAMMA, FOLDER

class Player():
    def __init__(self):
        self.env = UnityEnvironment(file_name="EXE\Client\CRML", seed=1, side_channels=[], worker_id=int(1),no_graphics = False) ## work_id need to be int 
        self.env.reset()
        self.behavior = list(self.env.behavior_specs)[0]
        self.pi = policy((CHANNEL, 7, 21), 5, epsilon_pair = EPSILON_PAIR, gamma = GAMMA)
        self.pi.load_state_dict(torch.load(PATH))
        self.pi.eval()
    def Play(self):
        while True:
            self.env.reset()
            state, _, _ = get_state(self.env, self.behavior)
            done = False
            
            while not(done):
                # Get State
                next_state, _, done = get_state(self.env, self.behavior)
                if not done:
                    # Set Action
                    action = self.pi.act(state)
                    actionTuple = ActionTuple()
                    action_t = np.asarray([[action]])
                    actionTuple.add_discrete(action_t)
                    
                    self.env.set_actions(self.behavior, actionTuple)
                # Next Step
                state = next_state
                self.env.step()
    def Close(self):
        self.env.close()


def main():
    player = Player()
    try:
        player.Play()
    except KeyboardInterrupt:
        print("Process ended, Close Enviroment")
        player.Close()
    except Exception as e:
        player.Close()
        raise e
       

if __name__ == "__main__":
    main()
        