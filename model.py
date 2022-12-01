import torch
import torch.multiprocessing as mp
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
import torch.nn as nn
from shared_adam import SharedAdam
# from train import 
NUM_GAMES = 300
MAX_EP = 5

class Network(nn.Module):
    def __init__(self, state_dim=60, action_dim=5, gamma=0.95):
        """
        Argument:
        * state_dim -- dim of state
        * action_dim -- dim of action space 
        * gamma -- discount factor (0.9 or 0.95 recommanded)
        =====================================================
        TODO: finetune the parameter of these neural networks 
        """
        super().__init__()

        # Actor
        self.net_actor = nn.Sequential(
            # nn.Conv2d(state_dim, 30, 3),
            nn.Linear(state_dim, 30),
            nn.ReLU(),
            nn.Linear(30, action_dim)
        )

        # Critic
        self.net_critic = nn.Sequential(
            nn.Linear(state_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )

        self.gamma = gamma
       
        self.states  = []
        self.actions = []
        self.rewards = []

    def forward(self, state): 
        """
        Forward the state into neural networks
        Argument:
            state
        Return:
            logits -- probability of each action being taken
            value  -- value of critic
        """
        # nn.init.xavier_normal_(self.net_actor.layer[0].weight)
        # nn.init.xavier_normal_(self.net_critic.layer[0].weight)
        logits = self.net_actor(state)
        value = self.net_critic(state)
        return logits, value

    def record(self, state, action, reward):
        """
        Record <state, action, reward> after taking this action
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def reset(self):
        """
        Reset the record 
        """
        self.states  = []
        self.actions = []
        self.rewards = []

    def take_action(self, state):
        """
        Argument:
            state
        Return:
            action -- the action with MAXIMUM probability
        """
        state = torch.tensor(state, dtype=torch.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().numpy()[0]
        return action

    def calc_R(self, done):
        """
        TODO: 
            This funciton is implemented in trivial way now,
            one may modify it into some efficient way
        """
        states = torch.tensor(self.states, dtype=torch.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def calc_loss(self, done):
        """
        TODO: 
        """
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = torch.softmax(pi, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

class Agent:

    def __init__(self, state_dim=60, action_dim=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_network = Network(state_dim, action_dim) # global network

        self.global_network.share_memory() # share the global parameters in multiprocessing
        self.opt = SharedAdam(self.global_network.parameters(), lr=1e-4, betas=(0.92, 0.999)) # global optimizer
        self.global_ep, self.res_queue = mp.Value('i', 0), mp.Queue()
        self.workers = [Worker(self.global_network, self.opt, 
                            self.state_dim, self.action_dim, 0.9, self.global_ep, i) 
                                for i in range(mp.cpu_count() - 7)]
        
    def close(self):
        [w.env.close() for w in self.workers]

    def run(self):  
    
        # parallel training
        [s.start() for s in self.workers]
        res = []  # record episode reward to plot
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [s.join() for s in self.workers]

class Worker(mp.Process): 
    """
    
    """
    def __init__(self, global_network, optimizer, 
            state_dim, action_dim, gamma, global_ep, name):
        super().__init__()
        self.local_network = Network(state_dim, action_dim)
        self.global_network = global_network
        self.optimizer = optimizer
        self.g_ep = global_ep       # total episodes so far across all workers
        self.l_ep = 0
        self.gamma = gamma          # reward discount factor
        
        self.name = f'woker {name}'
        self.env = UnityEnvironment(file_name="EXE\CRML", seed=1, side_channels=[], worker_id=name)
        self.env.reset()
        self.behavior = list(self.env.behavior_specs)[0]
        
    def pull(self, global_network):
        """
        pull the hyperparameter from global network to local network
        """
        None
    
    def push(self):
        """
        push the hyperparameter from local network to global network (consider gradient) 
        """
        None

    def close(self):
        self.env.close()

    def run(self):
        self.l_ep = 0
        while self.g_ep.value < NUM_GAMES:
            done = False
            self.env.reset()
            score = 0
            self.local_network.reset()
            while not done:
                decision_steps, terminal_steps = self.env.get_steps(self.behavior)
                step = None
                if len(terminal_steps) != 0:
                    step = terminal_steps[terminal_steps.agent_id[0]]
                    state = step.obs ## Unity return
                    done = True
                else:
                    step = decision_steps[decision_steps.agent_id[0]]
                    state = step.obs ## Unity return
                    action = self.local_network.take_action(state)
                    # FIXME: not compatible with current unity env., some slight adjustment is needed
                    actionTuple = ActionTuple()
                    # print(type(action), action)
                    if self.l_ep % MAX_EP:
                        action = np.asarray([[1]])
                    else:
                        action = np.asarray([[action]])
                    # print(type(action), action)
                    actionTuple.add_discrete(action) ## please give me a INT in a 2d nparray!!
                    # state_new, reward, done = self.env.set_actions(action) 
                    # actionTuple.add_continuous(np.array([[]])) ## please give me a INT in a 2d nparray!!
                    self.env.set_actions(self.behavior, actionTuple)
                reward = step.reward ## Unity return
                score += reward
                self.local_network.record(state, action, reward)
                if self.l_ep % MAX_EP == 0 or done:
                    loss = self.local_network.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_network.parameters(), self.global_network.parameters()):
                        global_param._grad = local_param.grad # push
                    self.optimizer.step()
                    self.local_network.load_state_dict(self.global_network.state_dict()) # pull
                    # self.local_network.reset()
                self.l_ep += 1
                self.env.step()
                # state = state_new
            with self.g_ep.get_lock():
                self.g_ep.value += 1
            print(f'{self.name}, episode {self.g_ep.value}, reward {score}')
            # print(f'{self.local_network.states}')
            # print(f'{self.local_network.actions}')
            # print(f'{self.local_network.rewards}')


if __name__ == '__main__':
    None