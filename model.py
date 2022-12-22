import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from shared_adam import SharedAdam
import datetime
import os

NUM_GAMES = 10000  # Maximum training episode for master agent
MAX_EP = 50       # Maximum training episode for slave agent

class Network(nn.Module):
    """
    Neural network components in A3C architecture
    """
    def __init__(self, state_dim=60, action_dim=5, gamma=0.95, name='test', timestamp=None ,load=False, path_actor='', path_critic=''):
        """
        Argument:
            state_dim -- dim of state
            action_dim -- dim of action space 
            gamma -- discount factor (0.9 or 0.95 recommanded)
        =====================================================
        TODO:
            * finetune the parameter of these neural networks 
            * try LSTM or CNN 
        """
        super().__init__()

        # from torchsummary import summary

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # (in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1) # (in_channels, out_channels, kernel_size)
        self.lstm = nn.LSTMCell(32*7*7, 67) # (input_size, hidden_size)

        # Actor
        # FIXME: Adjust the shape
        self.net_actor = nn.Linear(67, action_dim)
        # print(self.net_actor)
        # self.net_actor = nn.Sequential(
        #     nn.Conv2d(1, 10, (1,1)),
        #     nn.Flatten(0,-1),
        #     nn.ReLU(),
        #     nn.Linear(350, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 5)
        # )
        # Critic
        # FIXME: Adjust the shape
        self.net_critic = nn.Linear(67, 1)
        # self.net_critic = nn.Sequential(
        #     nn.Conv2d(1, 3, (1,1)),
        #     nn.Flatten(0,-1),
        #     nn.ReLU(),
        #     nn.Linear(105, 1)
        # )

        # load models
        if(load == True):
            self.net_actor.load_state_dict(torch.load(path_actor))
            self.net_critic.load_state_dict(torch.load(path_critic))

        self.gamma = gamma
       
        self.states  = np.array([])
        self.actions = []
        self.rewards = []

        # self.scores  = []

        self.name = name
        self.timestamp = timestamp

    def forward(self, state, lstm_par): 
        """
        Forward the state into neural networks
        Argument:
            state -- input state received from Unity environment 
        Return:
            logits -- probability of each action being taken
            value  -- value of critic
        """
        # nn.init.xavier_normal_(self.net_actor.layer[0].weight)
        # nn.init.xavier_normal_(self.net_critic.layer[0].weight)
        # FIXME: Adjust the shape
        # for i in range(state.shape[0]):
        #     s = state[i,:,:].reshape(1, 5, 7)
        #     s = self.conv(s)
        #     s = s.view(-1, 32*3*3)
        #     if (i == 0):
        #         logits = self.net_actor(s)
        #         value  = self.net_critic(s)
        #     else:
        #         logits = torch.vstack((logits, self.net_actor(s)))
        #         value  = torch.vstack((value, self.net_critic(s))) 
        (hx, cx) = lstm_par
        print("state 1 size: " + str(state.size()))
        state = F.relu(self.conv1(state))
        print("state 2 size: " + str(state.size()))
        state = F.relu(self.conv2(state))
        print("state 3 size: " + str(state.size()))
        state = state.view(-1, 32*7*7)
        print("state 4 size: " + str(state.size()))
        hx, cx = self.lstm(state, (hx, cx))
        print("state 5 size: " + str(state.size()))
        state = hx

        logits = self.net_actor(state)
        value = self.net_critic(state)

        return logits, value, (hx, cx)

    def record(self, state, action, reward):
        """
        Record <state, action, reward> after taking this action
        """
        if(self.states.size == 0):
            self.states = state
        else:
            self.states = np.append(self.states, state, axis=0)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def reset(self):
        """
        Reset the record 
        """
        self.states  = np.array([])
        self.actions = []
        self.rewards = []

    def take_action(self, state, lstm_par):
        """
        Argument:
            state -- input state received from Unity environment 
        Return:
            action -- the action with MAXIMUM probability
        """
        state = torch.tensor(state, dtype=torch.float)
        pi, v, (hx, cx)= self.forward(state, lstm_par)
        # print(f'pi: {pi} {pi.shape}')
        probs = torch.softmax(pi, dim=0)
        dist = torch.distributions.Categorical(probs)
        # print(dist.sample().numpy())
        action = dist.sample().numpy()
        return action, (hx, cx)

    def calc_R(self, done, lstm_par):
        """
        TODO: 
        """
        # print(self.states.shape)
        states = torch.tensor(self.states, dtype=torch.float)
        # print(states.shape)
        pi, v, (hx, cx) = self.forward(states[-1], lstm_par)

        R = v * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return, pi, v, (hx, cx)

    def calc_loss(self, done, lstm_par):
        """
        TODO: 
        """
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns, pi, values, _ = self.calc_R(done, lstm_par)

        # pi, values = self.forward(states)
        values = values.squeeze()
        # print(f'debug: {values.shape} {returns.shape}')
        critic_loss = (returns - values) ** 2
        # print(f'pi: {pi} {pi.shape}')
        probs = torch.softmax(pi, dim=0)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss
    
    def save(self):
        """
        Save the model parameters into .pt & .txt files
        """
        if not os.path.isdir(f'.\model\{self.timestamp}'):
            os.mkdir(f'.\model\{self.timestamp}')
        
        torch.save(self.net_actor.state_dict(), f'.\model\{self.timestamp}\{self.name}_actor.pt')
        torch.save(self.net_critic.state_dict(), f'.\model\{self.timestamp}\{self.name}_critic.pt')
        
        with open(f'.\model\{self.timestamp}\{self.name}_actor.txt', 'w') as fh:
            fh.write("Model's state_dict:\n")
            for param_tensor in self.net_actor.state_dict():
                fh.write(f'{param_tensor} \t {self.net_actor.state_dict()[param_tensor].size()}')
        
        with open(f'.\model\{self.timestamp}\{self.name}_critic.txt', 'w') as fh:
            fh.write("Model's state_dict:\n")
            for param_tensor in self.net_critic.state_dict():
                fh.write(f'{param_tensor} \t {self.net_critic.state_dict()[param_tensor].size()}')
        
        with open(f'.\model\{self.timestamp}\{self.name}_record.txt', 'w') as fh:
            fh.write("Index \t\t action \t reward:\n")
            for index, action, reward in zip(range(len(self.rewards)), self.actions, self.rewards):
                fh.write(f'{index:<10} \t {action.squeeze():<10} \t {reward.squeeze():<10}\n')

class Agent(mp.Process):
    """
    Master agent in A3C architecture
    """
    def __init__(self, state_dim=60, action_dim=5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_stamp=datetime.datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d-%H-%M-%S")
        self.global_network = Network(state_dim, action_dim, gamma=0.95, name='master', 
            timestamp=self.time_stamp,
            load=False, path_actor='.\model\master_actor.pt', path_critic='.\model\master_critic.pt') # global network
        
        self.global_network.share_memory() # share the global parameters in multiprocessing
        self.opt = SharedAdam(self.global_network.parameters(), lr=1e-5, betas=(0.92, 0.999)) # global optimizer
        self.global_ep, self.res_queue = mp.Value('i', 0), mp.Queue()
        
    def close(self):
        """
        Close all slave agent created (debugging usage)
        """
        [w.env.close() for w in self.workers]

    def run(self):
        """
        Initilize all slave agents and start parallel training
        """
        self.workers = [Worker(self.global_network, self.opt, 
                            self.state_dim, self.action_dim, 0.9, 
                            self.global_ep, i, self.global_network.timestamp, self.res_queue) 
                                for i in range(mp.cpu_count() - 0)]
        res = []
        # parallel training
        [w.start() for w in self.workers]
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in self.workers]
        # [w.save() for w in self.workers]

        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\ep_reward.png')
        plt.show()
    
    def save(self):
        self.global_network.save()

class Worker(mp.Process):
    """
    Slave agnet in A3C architecture
    """
    def __init__(self, global_network, optimizer, 
            state_dim, action_dim, gamma, global_ep, name, timestamp, res_queue):
        super().__init__()
        self.local_network = Network(state_dim, action_dim, gamma=0.95, name=f'woker{name}', timestamp=timestamp)
        self.global_network = global_network
        self.optimizer = optimizer
        self.g_ep = global_ep       # total episodes so far across all workers
        self.l_ep = None
        self.gamma = gamma          # reward discount factor
        self.res_queue = res_queue
        self.name = f'{name}'
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def pull(self):
        """
        pull the hyperparameter from global network to local network
        """
        self.local_network.load_state_dict(self.global_network.state_dict()) # pull
    
    def push(self):
        """
        push the hyperparameter from local network to global network (consider gradient) 
        """
        for local_param, global_param in zip(self.local_network.parameters(), self.global_network.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad # push

    def run(self):
        """
        Initilize Unity environment and start training
        """
        self.env = UnityEnvironment(file_name="EXE\CRML", seed=1, side_channels=[], worker_id=int(self.name)) ## work_id need to be int 
        self.env.reset()
        self.local_network.reset()
        # self.l_ep = 0
        self.behavior = list(self.env.behavior_specs)[0]

        while self.g_ep.value < NUM_GAMES:
            new_ep = True
            done = False
            self.l_ep = 0
            self.env.reset()
            score = 0
            self.local_network.reset()
            self.pull()

            while not done:
                if new_ep:
                    # initialize lstm parameters with zeros
                    (hx, cx) = (torch.zeros(1, self.state_dim), torch.zeros(1, self.state_dim)) # (batch_size, hidden_size)
                    # or with random values
                    # (hx, cx) = (torch.radn(1, self.state_dim), torch.radn(1, self.state_dim)) # (batch_size, hidden_size)
                    new_ep = False

                decision_steps, terminal_steps = self.env.get_steps(self.behavior)
                step = None

                

                if len(terminal_steps) != 0:
                    step = terminal_steps[terminal_steps.agent_id[0]]
                    state = np.array(step.obs) # [:,:49] ## Unity return
                    # FIXME: Adjust the shape
                    # state = np.vstack((
                    #     state[42:49],
                    #     state[35:42],
                    #     state[28:35],
                    #     state[21:28],
                    #     state[14:21],
                    #     state[7:14],
                    #     state[0:7],
                    # ))
                    state = state.reshape(1,7,7)
                    # state = state[:,2:7,:]
                    # state = state[2:6,:].reshape(1,4,7)
                    # print(state)
                    # print('\n')
                    done = True
                else:
                    step = decision_steps[decision_steps.agent_id[0]]
                    # Add noise
                    state = np.array(step.obs) + np.random.rand(*np.array(step.obs).shape) if self.g_ep.value < 1000 else np.array(step.obs)  # [:,:49] ## Unity return
                    # FIXME: Adjust the shape
                    # state = np.vstack((
                    #     state[42:49],
                    #     state[35:42],
                    #     state[28:35],
                    #     state[21:28],
                    #     state[14:21],
                    #     state[7:14],
                    #     state[0:7],
                    # ))
                    state = state.reshape(1,7,7)
                    # state = state[:,2:7,:] # (1, 5, 7)
                    # state = state[2:6,:].reshape(1,4,7)
                    # print(state)
                    # print('\n')
                    action, (hx, cx) = self.local_network.take_action(state, (hx, cx))

                    actionTuple = ActionTuple()

                    action = np.asarray([[action]])
                    
                    actionTuple.add_discrete(action) ## please give me a INT in a 2d nparray!!
                    
                    self.env.set_actions(self.behavior, actionTuple)
                reward = step.reward ## Unity return
                score += reward
                self.local_network.record(state, action, reward)
                
                if (self.l_ep % MAX_EP == 0 and self.l_ep != 0) or done == True:
                    # print(f'For debug usage: self.l_ep={self.l_ep}')
                    # detach current lstm parameters
                    cx = cx.detach()
                    hx = hx.detach()
                    loss = self.local_network.calc_loss(done, (hx, cx))
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.push()   
                    self.optimizer.step()
                    self.pull()
                    # self.local_network.reset()

                self.env.step()

                self.l_ep += 1
            
            with self.g_ep.get_lock():
                self.g_ep.value += 1
            
            self.res_queue.put(score)

            print(f'Worker {self.name}, episode {self.g_ep.value}, reward {score}')

        self.res_queue.put(None)

        self.save()

    def save(self):
        """
        Save the current model
        """
        self.local_network.save()


if __name__ == '__main__':
    None