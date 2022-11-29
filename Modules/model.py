import torch
import torch.multiprocessing as mp
import torch.nn as nn
from worker import Worker
from .shared_adam import SharedAdam

class Network(nn.Module):
    def __init__(self, state_dim=60, action_dim=5, gamma=0.95):
        """
        Argument:
        * state_dim -- dim of state
        * action_dim -- dim of action space 
        * gamma -- discount factor (0.9 or 0.95 recommanded)
        =====================================================
        TODO: finetune the parameter og these neural networks 
        """
        super().__init__()

        # Actor
        self.net_actor = nn.Sequential(
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
        state = torch.tensor(state, dtype=T.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        dist = torch.Categorical(probs)
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
        dist = torch.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

class Agent:

    def __init__(self, state_dim=60, action_dim=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_network = Network(state_dim, action_dim) # global network
        
    
    def run(self):  
        self.global_network.share_memory() # share the global parameters in multiprocessing
        self.opt = SharedAdam(self.global_network.parameters(), lr=1e-4, betas=(0.92, 0.999)) # global optimizer
        self.global_ep, res_queue = mp.Value('i', 0), mp.Queue()
        self.workers = [Worker(self.global_network, self.opt, 
                            self.state_dim, self.action_dim, self.global_ep, i) 
                                for i in range(mp.cpu_count())]

        # parallel training
        [s.start() for s in self.slave_agents]
        res = []  # record episode reward to plot
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [s.join() for s in self.slave_agents]
        

if __name__ == '__main__':
    None