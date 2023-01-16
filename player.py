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

PATH = ".\\model\\2022-12-31-15-46-00\\0_1900_370_model_state_dict.pt"
from src.model import Network, GAMMA, STATE_DIM, NUM_AGENTS

def get_state(env, behavior):
    decision_steps, terminal_steps = env.get_steps(behavior)
    
    if len(terminal_steps) != 0:
        step = terminal_steps[terminal_steps.agent_id[0]]
        state = None
        done = True
    else:
        step = decision_steps[decision_steps.agent_id[0]]
        state = np.array(step.obs)
        state = np.vstack((
            state[126:147],
            state[105:126],
            state[84:105],
            state[63:84],
            state[42:63],
            state[21:42],
            state[0:21],
        ))
        state = np.hstack(((np.where(state == -1, 1, 0)), 
                            (np.where(state == 0, 1, 0)), 
                            (np.where(state == 1, 1, 0)),
                            (np.where(state == 2, 1, 0))))
        state = state.reshape(1, 4, 7, 21)
        done = False

    return [state, step.reward, done]

class Player():
    def __init__(self):
        self.env = UnityEnvironment(file_name="EXE\Client\CRML", seed=1, side_channels=[], worker_id=int(1), no_graphics = False) ## work_id need to be int 
        self.env.reset()
        self.behavior = list(self.env.behavior_specs)[0]
        self.time_stamp=datetime.datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d-%H-%M-%S")
        self.network = Network(STATE_DIM, 5, gamma=GAMMA, name='master', 
            timestamp=self.time_stamp, load=True, path=PATH)
        self.network.eval()
    def Play(self):
        while True:
            self.env.reset()
            done = False
            (hx, cx) = (torch.zeros(1, self.network.linear_input_size), torch.zeros(1, self.network.linear_input_size))
            while not(done):
                # Get State
                state, _, done = get_state(self.env, self.behavior)
                if not done:
                    # Set Action
                    action, _, _ = self.network.take_action(state,(hx, cx))
                    actionTuple = ActionTuple()
                    action_t = np.asarray([[action]])
                    actionTuple.add_discrete(action_t)
                    
                    self.env.set_actions(self.behavior, actionTuple)
                # Next Step
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
        