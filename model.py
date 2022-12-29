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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
MC = False
TD = not MC
STATE_SHRINK = False
STATE_DIM = (4, 5, 21) if STATE_SHRINK else (4, 7, 21)
GRADIENT_ACC = True
GAMMA  = 0.90
LAMBDA = 0.95
LR = 1e-4

NUM_GAMES = 1e3                   # Maximum training episode for slave agent to update master agent
MAX_STEP  = 10                    # Maximum step for slave agent to accumulate gradient
MAX_EP    = 5

class Network(nn.Module):
    """
    Neural network components in A3C architecture
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=5, gamma=0.95, name='test', timestamp=None ,load=False, path_actor='', path_critic=''):
        """
        Argument:
            state_dim -- dim of state
            action_dim -- dim of action space 
            gamma -- discount factor (0.9 or 0.95 recommanded)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        c, h, w = self.state_dim

        convw = w
        convh = h
        self.linear_input_size = convw * convh * 32

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.Flatten(0,-1),
        )

        self.lstm = nn.LSTMCell(self.linear_input_size, self.linear_input_size)

        # Actor
        self.net_actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.linear_input_size, self.linear_input_size // 4),
            nn.ReLU(),
            nn.Linear(self.linear_input_size // 4, self.action_dim),
            nn.ReLU()
        )

        # Critic
        self.net_critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.linear_input_size, 1),
            nn.ReLU()
        )
        self.gamma = gamma
       
        self.states  = np.array([])
        self.actions = []
        self.rewards = []

        self.entropy = 0
        
        self.values     = []
        self.entropies  = []
        self.log_probs  = []

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
        state = state.reshape(1, 4, 5, 21) if STATE_SHRINK else state.reshape(1, 4, 7, 21)
        s = self.conv(state)
        s = s.view(-1, 32 * 5 * 21) if STATE_SHRINK else s.view(-1, 32 * 7 * 21)

        hx, cx = self.lstm(s, lstm_par)
        s = hx

        logits = self.net_actor(s)
        value  = self.net_critic(s)

        return torch.squeeze(logits), value, (hx, cx)

    def record(self, state, action, reward, value):
        """
        Record <state, action, reward> after taking this action
        """
        if(self.states.size == 0):
            self.states = state
        else:
            self.states = np.append(self.states, state, axis=0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

        self.entropy = 0
    
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
        state = torch.tensor(np.array(state), dtype=torch.float)
        pi, v, (hx, cx) = self.forward(state, lstm_par)
        
        # Add the `detach` to prevent "backward through a graph a second time"
        probs = torch.softmax(pi.detach(), dim=0)
        log_probs = torch.log_softmax(pi.detach(), dim=0)
        dist = torch.distributions.Categorical(probs)

        entropy = - (log_probs * probs).sum(-1)
        self.entropy += entropy

        self.entropies.append(entropy)
        self.log_probs.append(log_probs)
        
        action = dist.sample().numpy()
        return action, v, (hx, cx)

    def calc_R(self, done, lstm_par, normalize = False):
        """
        Monte-Carlo method implementation
        """
        states = torch.tensor(self.states, dtype=torch.float)
        
        pi, v, (hx, cx) = self.forward(states[-1], lstm_par)

        R = v * (1 - int(done))
    
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        if(normalize == True):
            batch_return = (batch_return - batch_return.mean()) / batch_return.std()
        
        return batch_return, pi, v, (hx, cx)

    def calc_loss(self, done, lstm_par):
        """
        Monte-Carlo method implementation
        """
        states = torch.tensor(self.states, dtype=torch.float)
        # print(self.actions)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns, pi, values, (hx, cx)= self.calc_R(done, lstm_par)

        values = values.squeeze()
        critic_loss = (returns - values) ** 2
        
        probs = torch.softmax(pi, dim=0)
        dist = torch.distributions.Categorical(probs)
        # print(dist)

        # m = self.distribution(mu, sigma)
        # entropy = 0.5 + 0.5 * np.log(2 * np.pi) + torch.log(m.scale)  # exploration

        log_probs = dist.log_prob(actions)
        # print(f'{log_probs.shape}, {entropy.shape}')
        # print(f'{log_probs}, {entropy}')
        actor_loss = - log_probs * (returns-values) + torch.full(log_probs.shape, self.entropy) * 0.005

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def calc_loss_v2(self, done):
        """
        Temporal difference method implementation
        """
        R = torch.zeros((1, 1), dtype=torch.float)
        gae = torch.zeros((1, 1), dtype=torch.float)
        actor_loss = torch.zeros((1, 1), dtype=torch.float, requires_grad=True)
        critic_loss = torch.zeros((1, 1), dtype=torch.float, requires_grad=True)
        entropy_loss = torch.zeros((1, 1), dtype=torch.float)
        next_value = R

        for value, log_policy, reward, entropy in list(zip(self.values, self.log_probs, 
                                                        self.rewards, self.entropies))[::-1]:
            gae = gae * self.gamma * LAMBDA + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * self.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
        
        total_loss = (actor_loss + critic_loss + 0.3 * entropy_loss).mean()
        
        return total_loss
    
    def save(self):
        """
        Save the model parameters into .pt & .txt files
        """
        if not os.path.isdir(f'.\model\{self.timestamp}'):
            os.mkdir(f'.\model\{self.timestamp}')

        # Calculate total time
        end = datetime.datetime.now().replace(second=0, microsecond=0)
        start = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d-%H-%M-%S")
        timedelta = end - start

        # Ouput parameters
        with open(f'.\model\{self.timestamp}\parameters.txt', 'w') as fh:
            fh.write(f'Timestamp: {self.timestamp}\n')
            fh.write(f'Training time: {timedelta}\n')
            fh.write(f'State dimension: {self.state_dim}\n')   # Input dimension
            fh.write(f'Action dimension: {self.action_dim}\n') # Output dimension
            fh.write(f'Maximum training episode for master agent: {NUM_GAMES}\n')
            fh.write(f'Maximum training episode for slave agent: {MAX_EP}\n')
            if MC:
                fh.write(f'Loss calculation method: MC\n')
            if TD:
                fh.write(f'Loss calculation method: TD\n')
            fh.write(f'State shrink: {STATE_SHRINK}')
            fh.write(f'Gradirnt accumulatoin: {GRADIENT_ACC}\n')
            fh.write(f'GAMMA: {GAMMA}\n')
            fh.write(f'LAMBDA: {LAMBDA}\n')
            fh.write(f'Learning rate: {LR}\n')
            fh.write(f'Iterations: {NUM_GAMES}\n')
            fh.write(f'============================================================\n')
            fh.write(f'{self}')
            # fh.write(f'1st convolution network:\n{self.conv1}\n')
            # fh.write(f'1st convolution network state dict:\n{self.conv1.state_dict()}\n')
            # fh.write(f'\n-----\n')
            # fh.write(f'2st convolution network:\n{self.conv2}\n')
            # fh.write(f'2st convolution network state dict:\n{self.conv2.state_dict()}\n')
            # fh.write(f'\n-----\n')
            # fh.write(f'LSTM network:\n{self.lstm}\n')
            # fh.write(f'LSTM network state dict:\n{self.lstm.state_dict()}\n')
            # fh.write(f'\n-----\n')
            # # fh.write(f'lstm:\n{self.lstm}\n')
            # fh.write(f'Actor network:\n{self.net_actor}\n')
            # fh.write(f'Actor network state dict:\n{self.net_actor.state_dict()}\n')
            # fh.write(f'\n-----\n')
            # fh.write(f'Critic network:\n{self.net_critic}\n')
            # fh.write(f'Critic network state dict:\n{self.net_critic.state_dict()}\n')

class Agent(mp.Process):
    """
    Master agent in A3C architecture
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_stamp=datetime.datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d-%H-%M-%S")
        self.global_network = Network(state_dim, action_dim, gamma=GAMMA, name='master', 
            timestamp=self.time_stamp,
            load=False, path_actor='.\model\master_actor.pt', path_critic='.\model\master_critic.pt') # global network
        
        self.global_network.share_memory() # share the global parameters in multiprocessing
        self.opt = SharedAdam(self.global_network.parameters(), lr=LR, betas=(0.92, 0.999)) # global optimizer
        self.global_ep, self.res_queue, self.score_queue, self.loss_queue = \
            mp.Value('i', 0), mp.Manager().Queue(), mp.Manager().Queue(), mp.Manager().Queue()

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
                                self.state_dim, self.action_dim, GAMMA, 
                                self.global_ep, i, self.global_network.timestamp, 
                                self.res_queue,self.score_queue,self.loss_queue) 
                                    for i in range(mp.cpu_count() - 15)]
        # parallel training
        [w.start() for w in self.workers]

        # record episode reward to plot
        res   = []
        score = []
        loss  = []
          
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
            
        while True:
            sc = self.score_queue.get()
            if sc is not None:
                score.append(sc)
            else:
                break
            
        while True:
            los = self.loss_queue.get()
            if los is not None:
                loss.append(los)
            else:
                break
        
        [w.join() for w in self.workers]

        # plot
        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\ep_reward.png')
        plt.close()
        
        plt.plot(score)
        plt.ylabel('game score')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\gamescore.png')
        plt.close()

        plt.plot(loss)
        plt.ylabel('entropy loss')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\loss.png')
        plt.close()

        self.save()
        
    def save(self):
        # self.global_network.save()
        # torch.save(self.global_network, f'.\model\{self.time_stamp}\{self.name}_model.pt')
        torch.save(self.global_network.state_dict(), f'.\model\{self.time_stamp}\{self.name}_model_state_dict.pt')

class Worker(mp.Process):
    """
    Slave agnet in A3C architecture
    """
    def __init__(self, global_network, optimizer, 
            state_dim, action_dim, gamma, global_ep, name, timestamp, res_queue, score_queue,loss_queue):
        super().__init__()
        self.local_network = Network(state_dim, action_dim, gamma=0.95, name=f'woker{name}', timestamp=timestamp)
        self.global_network = global_network
        self.optimizer = optimizer
        self.g_ep = global_ep       # total episodes so far across all workers
        self.l_ep = None
        self.l_step = None
        self.gamma = gamma          # reward discount factor
        self.res_queue = res_queue
        self.name = f'{name}'
        self.res_queue = res_queue
        self.score_queue = score_queue
        self.loss_queue = loss_queue
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
        if int(self.name) == 0:
            self.env = UnityEnvironment(file_name="EXE\Client\CRML", seed=1, side_channels=[], no_graphics=True, worker_id=int(self.name)) ## work_id need to be int 
        else:
            self.env = UnityEnvironment(file_name="EXE\Client\CRML", seed=1, side_channels=[], no_graphics=True, worker_id=int(self.name)) ## work_id need to be int 
        self.env.reset()
        self.local_network.reset()
        self.pull()
        self.l_ep = 0
        self.behavior = list(self.env.behavior_specs)[0]

        while self.g_ep.value < NUM_GAMES:
            new_ep = True
            done = False
            self.env.reset()
            self.local_network.reset()
            self.l_step = 0
            
            score = 0
            best_score = 0
            current_score = 0
            
            while not done:
                if new_ep:
                    # initialize lstm parameters with zeros
                    (hx, cx) = (torch.zeros(1, self.local_network.linear_input_size), torch.zeros(1, self.local_network.linear_input_size)) # (batch_size, hidden_size)
                    # or with random values
                    # (hx, cx) = (torch.radn(1, self.state_dim), torch.radn(1, self.state_dim)) # (batch_size, hidden_size)
                    new_ep = False

                decision_steps, terminal_steps = self.env.get_steps(self.behavior)
                step = None

                if len(terminal_steps) != 0:
                    step = terminal_steps[terminal_steps.agent_id[0]]
                    state = np.array(step.obs) ## Unity return
                    # FIXME: Adjust the shape for different state size
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
                    state = state.reshape(1, 4, 7, 21)
                    if STATE_SHRINK:
                        state = state[:, :, 2:7, :]
                    done = True
                    if TD:
                        action, value, (hx, cx) = self.local_network.take_action(state, (hx, cx))
                        action = np.asarray([[action]])
                else:
                    step = decision_steps[decision_steps.agent_id[0]]
                    # Add noise
                    state = np.array(step.obs) + 0.1 * np.random.rand(*np.array(step.obs).shape) if self.g_ep.value < 1000 else np.array(step.obs)  # [:,:49] ## Unity return
                    # FIXME: Adjust the shape for different state size
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
                    state = state.reshape(1, 4, 7, 21)
                    if STATE_SHRINK:
                        state = state[:, :, 2:7, :]
                    action, value, (hx, cx) = self.local_network.take_action(state, (hx, cx))

                    actionTuple = ActionTuple()

                    if action == 1:
                        current_score += 1
                    elif action == 2:
                        current_score -= 1
                    if current_score > best_score:
                        best_score = current_score
                    action = np.asarray([[action]])
                    
                    actionTuple.add_discrete(action) ## please give me a INT in a 2d nparray!!
                    
                    self.env.set_actions(self.behavior, actionTuple)
                reward = step.reward ## Unity return
                reward = 0.1 * reward if self.l_step > 10 else reward
                score += reward
                self.local_network.record(state, action, reward, value.detach())
                
                if GRADIENT_ACC :
                    # Do the gradient descent but not update global network directly
                    if (self.l_step % MAX_STEP == 0 and self.l_step != 0) or done == True:
                        # detach current lstm parameters
                        cx = cx.detach()
                        hx = hx.detach()

                        loss = self.local_network.calc_loss(done, (hx, cx))
                        self.loss_queue.put(loss.clone().detach().numpy()) #record loss
                        loss.backward()

                    # update global network (gradient accumulation!)
                    if (self.l_ep % MAX_EP == 0):
                        self.push()   
                        self.optimizer.step()
                        self.pull()
                        self.optimizer.zero_grad()
                else:
                    if (self.l_ep % MAX_STEP == 0 and self.l_ep != 0) or done == True: 
                        # detach current lstm parameters
                        cx = cx.detach()
                        hx = hx.detach()
                        
                        loss = self.local_network.calc_loss(done, (hx, cx)) if MC else self.local_network.calc_loss_v2(done)
                        self.optimizer.zero_grad()
                        self.loss_queue.put(loss.clone().detach().numpy()) #record loss
                        loss.backward()
                        self.push()   
                        self.optimizer.step()
                        self.pull()

                self.env.step()

                self.l_step += 1
            
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                self.l_ep       += 1
            
            self.res_queue.put(score)
            self.score_queue.put(best_score)

            print(f'Worker {self.name}, episode {self.g_ep.value}, reward {score}, score {best_score}')

        self.res_queue.put(None)
        self.score_queue.put(None)
        self.loss_queue.put(None)
        self.save()
        self.env.close()

    def save(self):
        """
        Save the current model
        """
        self.local_network.save()

if __name__ == '__main__':
    None