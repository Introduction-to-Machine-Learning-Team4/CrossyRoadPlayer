import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
import datetime
import os
from collections import namedtuple, deque
import random
import datetime


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOUBLE = True
FOLDER = 'model-ddqn' if DOUBLE else 'model-dqn'
GAMMA = 0.9
EPSILON_PAIR = (0.2, 0.01)
N_EPOCHS = 10000
BATCH_SIZE = 32
N_UPDATES = 3
N_BATCHES_PER_EPOCH = 3
TAR_NET_UPDATE_PERIOD = 5
time_stamp = datetime.datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d-%H-%M-%S")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

loss_rec = []

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class policy(nn.Module):
    
    def __init__(self, state_dim = (1, 1), act_dim = 1, struct = None, 
                 epsilon_pair = (0.9, 0.05), gamma = 0.99):
        
        super(policy, self).__init__()
        self.size = state_dim
        c, h, w = state_dim
        
        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        
        
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        
        convw = w
        convh = h
        linear_input_size = convw * convh * 32
        
        self.fc1 = nn.Linear(linear_input_size, linear_input_size // 4)
        self.fc2 = nn.Linear(linear_input_size // 4, act_dim)
        self.relu = nn.ReLU()
        
        self.act_dim = act_dim
        self.epi_start, self.epi_end = epsilon_pair
        self.epi = self.epi_start
        self.gamma = gamma
        self.loss = nn.MSELoss()
        
    
    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        x = x.to(DEVICE)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        
        return x
    
    def act(self, state): 
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32)
            state = state.to(DEVICE)
            q_value = self.forward(state.reshape((1,4,7,21)))
            
        if np.random.rand() < self.epi:
            act_idx = np.random.choice(self.act_dim)
        
        else:
            act_idx = q_value.argmax().item()

        return act_idx
    
    def decay_epsilon(self, portion):
        self.epi = self.epi_end + (self.epi_start - self.epi_end) * np.exp(portion * (-3))
        # print(self.epi)

def train_DQN(pi, tar_pi, optimizer, memory, n_updates = 1, double = False):
    if (len(memory) < BATCH_SIZE):
        return
    
    batch = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*batch))
    states = torch.from_numpy(np.stack(batch.state, axis = 0)).to(DEVICE)
    actions = torch.tensor(batch.action, device = DEVICE).view(-1, 1)
    
    
    with torch.no_grad():
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=DEVICE, dtype=torch.bool)
        
        non_final_states = torch.stack([torch.from_numpy(s) 
            for s in batch.next_state if s is not None]).to(DEVICE)

        rewards = torch.tensor(batch.reward, device = DEVICE)
    
    losses = np.zeros(n_updates)
    for i in range(n_updates):
        if double:
            q_values = pi.forward(states)
            with torch.no_grad():
                max_actions = q_values.max(1)[1].view(-1, 1)
                max_actions = max_actions[non_final_mask]
                q_values_tar = torch.zeros(BATCH_SIZE, device = DEVICE)
                q_values_tar[non_final_mask] = tar_pi.forward(
                    non_final_states).gather(1, max_actions).view(-1)
                q_values_tar = q_values_tar * tar_pi.gamma + rewards
                q_values_tar = q_values_tar.detach()
            q_values = q_values.gather(1, actions).view(-1) 
        else:
            q_values = pi.forward(states).gather(1, actions).view(-1)   
            with torch.no_grad():
                q_values_tar = torch.zeros(BATCH_SIZE, device = DEVICE)
                q_values_tar[non_final_mask] = tar_pi.forward(non_final_states).max(1)[0]
                q_values_tar = q_values_tar * tar_pi.gamma + rewards
                q_values_tar = q_values_tar.detach()
        
        loss = pi.loss(q_values, q_values_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[i] = loss.item()
    
    return losses.mean()

def get_state(env, behavior):
    decision_steps, terminal_steps = env.get_steps(behavior)
    
    if len(terminal_steps) == 0:
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
                    (np.where(state ==  0, 1, 0)), 
                    (np.where(state ==  1, 1, 0)),
                    (np.where(state ==  2, 1, 0))))
        state = state.reshape(4, 7, 21)
        done = False
    else:
        step = terminal_steps[terminal_steps.agent_id[0]]
        state = None
        done = True
    return [state, step.reward, done]

def main():
    env = UnityEnvironment(file_name="EXE\Headless\CRML", seed=1, side_channels=[], worker_id=int(3)) ## work_id need to be int 
    env.reset()
    behavior = list(env.behavior_specs)[0]
    pi = policy((4, 7, 21), 5, epsilon_pair = EPSILON_PAIR, gamma = GAMMA)
    pi.to(DEVICE)
    tar_pi = policy((4, 7, 21), 5, epsilon_pair = EPSILON_PAIR, gamma = GAMMA)
    tar_pi.to(DEVICE)
    tar_pi.load_state_dict(pi.state_dict())
    optimizer = torch.optim.RMSprop(pi.parameters(), lr = 0.01)
    memory = ReplayMemory(4000)
    score_rec = 0
    
    try:
        best_score_list = []
        for epi in range(N_EPOCHS):
            score = 0
            env.reset()
            state, _, _ = get_state(env, behavior)
            done = False
            
            current_score = 0
            best_score    = 0

            while not(done):
                action = pi.act(state)

                if action == 1:
                    current_score += 1
                elif action == 2:
                    current_score -= 1
                if current_score > best_score:
                    best_score = current_score
                actionTuple = ActionTuple()
                action_t = np.asarray([[action]])
                actionTuple.add_discrete(action_t) ## please give me a INT in a 2d nparray!!
                
                env.set_actions(behavior, actionTuple)
                env.step()
                next_state, reward, done = get_state(env, behavior)
                memory.push(state, action, next_state, reward)
                score += reward
                state = next_state
                losses = np.zeros(N_BATCHES_PER_EPOCH)
                
            for i in range(N_BATCHES_PER_EPOCH):
                print(f"train iter {i}")
                loss = train_DQN(pi, tar_pi, optimizer, memory, n_updates = N_UPDATES, double=DOUBLE)
                if (loss != None):
                    losses[i] = loss.item()
         
            pi.decay_epsilon(epi / N_EPOCHS)
            loss_avg = losses.mean()
            loss_rec.append(loss_avg)
            print(f'Episode {epi}\nloss: {loss_avg}\ntotal_reward: {score}\nbest_score: {best_score}\n')

            best_score_list.append(best_score)
           
            
            if not os.path.isdir(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}'):
                os.mkdir(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}')
            if best_score >= 200 and best_score > score_rec:
                score_rec = best_score
                torch.save(pi, f'.\\{FOLDER}\\4-channel-state\\{time_stamp}\{epi}_{best_score}_dqn.pt')
                # torch.save(pi.state_dict, f'.\\{FOLDER}\\4-channel-state\\{time_stamp}\{epi}_{best_score}_dqn_state_dict.pt')
        
        env.close()

        if not os.path.isdir(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}'):
            os.mkdir(f'\{FOLDER}\{time_stamp}\4-channel-state')
        
        with open(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}\score.txt', 'w') as fh:
            for i, s in enumerate(best_score_list):
                fh.write(f"{i}: {s}\n")

        with open(f'.\\{FOLDER}\\1-channel-state\\{time_stamp}\loss.txt', 'w') as fh:
            for i, s in enumerate(loss_rec):
                fh.write(f"{i}: {s}\n")
        
        with open(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}\\info.txt', 'w') as fh:
            fh.write(f"DDQN: {DOUBLE}\n")
            fh.write(f"GAMMA: {GAMMA}\n")
            fh.write(f"EPSILON_PAIR: {EPSILON_PAIR}\n")
            fh.write(f"N_EPOCHS: {N_EPOCHS}\n")
            fh.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            fh.write(f"N_UPDATES: {N_UPDATES}\n")
            fh.write(f"N_BATCHES_PER_EPOCH: {N_BATCHES_PER_EPOCH}\n")
            fh.write(f"TAR_NET_UPDATE_PERIOD: {TAR_NET_UPDATE_PERIOD}\n")

        import matplotlib.pyplot as plt
        plt.plot(best_score_list)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}\score.png')
        plt.close()

        plt.plot(loss_rec)
        plt.ylabel('Loss')
        plt.xlabel('Step')
        plt.savefig(f'.\\{FOLDER}\\4-channel-state\\{time_stamp}\loss.png')
        plt.close()

    except Exception as e:
        env.close()
        raise e

if __name__ == '__main__':
    main()  