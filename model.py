import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from shared_adam import SharedAdam
import datetime
import os

NUM_GAMES = 100000  # Maximum total training episode for master agent
MAX_EP    = 10      # Maximum training episode for slave agent to update master agent
MAX_STEP  = 100     # Maximum step for slave agent to accumulate gradient

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

        self.state_dim = state_dim
        self.action_dim = action_dim
   
        # Actor
        # FIXME: Adjust the shape for different state size
        self.net_actor = nn.Sequential(
            nn.Conv2d(1, 10, (1,1)),
            nn.Flatten(0,-1),
            nn.ReLU(),
            nn.Linear(350, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        # Critic
        # FIXME: Adjust the shape for different state size
        self.net_critic = nn.Sequential(
            nn.Conv2d(1, 3, (1,1)),
            nn.Flatten(0,-1),
            nn.ReLU(),
            nn.Linear(105, 1)
        )

        # load models
        if(load == True):
            self.net_actor.load_state_dict(torch.load(path_actor))
            self.net_critic.load_state_dict(torch.load(path_critic))

        self.gamma = gamma
       
        self.states  = np.array([])
        self.actions = []
        self.rewards = []
        
        self.values     = []
        self.entropies  = []
        self.log_probs  = []

        # self.scores  = []

        self.name = name
        self.timestamp = timestamp

    def forward(self, state): 
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
        # FIXME: Adjust the shape for different state size
        for i in range(state.shape[0]):
            s = state[i,:,:].reshape(1, 5, 7)
            if i == 0:            
                logits = self.net_actor(s)
                value  = self.net_critic(s)
            else:
                logits = torch.cat((logits.clone().detach(), self.net_actor(s)), 1)
                value  = torch.cat((value.clone().detach(), self.net_critic(s)), 1) 
        return logits, value

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
    
    def reset(self):
        """
        Reset the record 
        """
        self.states  = np.array([])
        self.actions = []
        self.rewards = []

    def take_action(self, state):
        """
        Argument:
            state -- input state received from Unity environment 
        Return:
            action -- the action with MAXIMUM probability
        """
        state = torch.tensor(state, dtype=torch.float)
        pi, v = self.forward(state)
        
        probs = torch.softmax(pi.detach(), dim=0)
        log_probs = torch.log_softmax(pi.detach(), dim=0)
        dist = torch.distributions.Categorical(probs)
        # print(f'For debug usage: probs={probs}')

        entropy = - (log_probs * probs).sum(-1)
        self.entropies.append(entropy)
        self.log_probs.append(log_probs)
        
        action = dist.sample().numpy()
        return action , v

    def calc_R(self, done, normalize = False):
        """
        TODO: Try GAE
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

        if(normalize == True):
            batch_return = (batch_return - batch_return.mean()) / batch_return.std()
        
        return batch_return

    def calc_loss(self, done):
        """
        TODO:
            * TD version
            * separate critic loss & actor loss?
        """
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        
        critic_loss = (returns - values) ** 2
        
        probs = torch.softmax(pi, dim=0)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def calc_loss_v2(self, done):
        """
        * try TD version
        """
        # states = torch.tensor(self.states, dtype=torch.float)
        # actions = torch.tensor(self.actions, dtype=torch.float)
        # # pi, values = self.forward(states)
        # LAMBDA = 0.99
        # R = 0
        # policyLoss = 0
        # valueLoss = 0
        # gae = 0
        # self.values.append(torch.zeros(1)) # we need to add this for the deltaT equation
        # for i in reversed(range(len(self.rewards))):
        #     R = self.gamma * R + self.rewards[i]
        #     advantage = R - self.values[i]
        #     valueLoss = valueLoss + 0.5 * advantage**2
        #     deltaT = self.rewards[i] + self.gamma * self.values[i + 1] - self.values[i]
        #     gae = gae * self.gamma * LAMBDA + deltaT
        #     policyLoss = policyLoss - self.log_probs[i] * gae - 0.01 * self.entropies[i]
        
        # loss = (policyLoss + valueLoss).mean()
        # return loss
        R = torch.zeros((1, 1), dtype=torch.float)

        gae = torch.zeros((1, 1), dtype=torch.float)
        # if opt.use_gpu:
        #     gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(self.values, self.log_probs, self.rewards, self.entropies))[::-1]:
            # gae = gae * self.gamma * opt.tau
            gae = gae + reward + self.gamma * next_value.clone() - value.clone()
            next_value = value.clone()
            actor_loss = actor_loss + log_policy * gae
            R = R * self.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
        total_loss = - actor_loss + critic_loss - 0.5 * entropy_loss
        return total_loss.mean()
        # with torch.autograd.set_detect_anomaly(True):
        #     for i in reversed(range(len(self.rewards))):
        #         R = self.gamma * R + self.rewards[i]
        #         # print(f'{type(R)}, R')
        #         advantage = R - self.values[i].clone()
        #         # print(f'{type(advantage)}, advantage')
        #         valueLoss = torch.tensor(valueLoss).clone() + 0.5 * advantage.clone()**2
        #         # print(f'{type(valueLoss)}, valueLoss')
        #         deltaT = self.rewards[i] + self.gamma * self.values[i + 1].clone() - self.values[i].clone()
        #         # print(f'{type(deltaT)}, deltaT')
        #         gae = torch.tensor(gae).clone() * self.gamma * LAMBDA + deltaT.clone() 
        #         # print(f'{type(gae)}, gae')
        #         policyLoss = torch.tensor(policyLoss).clone() - torch.tensor(self.log_probs[i]).clone() * torch.tensor(gae).clone() - 0.01 * torch.tensor(self.entropies[i]).clone()
        #         # print(f'{type(policyLoss)}, policyLoss')
        #         loss = (policyLoss + 0.5 * valueLoss).mean()
        #     # from torch.autograd import Variable as v
        #     # loss = v(torch.FloatTensor([2]), requires_grad=True)
        # print(f'For debug usage: {type(loss)}')
        # return loss
        '''
        # returns = self.calc_R(done)

        # next_value = 0
        # trace_decay = 0.9 # FIXME: should be an input parameter
        # advantages = []
        
        # pi, values = self.forward(states)
        # for r, v in zip(reversed(self.rewards), reversed(values)):
        #     td_err = r + next_value * self.gamma - v
        #     advantage = td_err + advantage * self.gamma * trace_decay
        #     advantages.insert(0, advantage)
        # advantages = torch.tensor(advantages)
        # advantages.detach()
        
        # probs = torch.softmax(pi, dim=0)
        # dist = torch.distributions.Categorical(probs)
        # log_probs = dist.log_prob(actions)
        # actor_loss = - (advantages * log_probs).sum()
    
        # critic_loss = F.smooth_l1_loss(returns, values).sum()

        # total_loss = (critic_loss + actor_loss).mean()
    
        # return total_loss
        '''
    
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

        # with open(f'.\model\{self.timestamp}\{self.name}_lstm.txt', 'w') as fh:
        #     fh.write("Model's state_dict:\n")
        #     for param_tensor in self.lstm.state_dict():
        #         fh.write(f'{param_tensor} \t {self.lstm.state_dict()[param_tensor].size()}')
        
        with open(f'.\model\{self.timestamp}\{self.name}_record.txt', 'w') as fh:
            fh.write("Index \t\t action \t reward:\n")
            for index, action, reward in zip(range(len(self.rewards)), self.actions, self.rewards):
                fh.write(f'{index:<10} \t {action.squeeze():<10} \t {reward.squeeze():<10}\n')

        # Ouput parameters
        with open(f'.\model\{self.timestamp}\parameters.txt', 'w') as fh:
            fh.write(f'timestamp: {self.timestamp}\n')
            fh.write(f'state dimension: {self.state_dim}\n') # Input dimension
            fh.write(f'action dimension: {self.action_dim}\n') # Output dimension
            fh.write(f'Maximum training episode for master agent: {NUM_GAMES}\n')
            fh.write(f'Maximum training episode for slave agent: {MAX_EP}\n')
            fh.write(f'============================================================\n')
            # fh.write(f'lstm:\n{self.lstm}\n')
            fh.write(f'actor network:\n{self.net_actor}\n')
            fh.write(f'critic network:\n{self.net_critic}\n')

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
        self.global_ep, self.res_queue, self.score_queue, self.loss_queue = \
            mp.Value('i', 0), mp.Queue(), mp.Queue(), mp.Queue()

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
                            self.global_ep, i, self.global_network.timestamp, self.res_queue,self.score_queue,self.loss_queue) 
                                for i in range(mp.cpu_count() - 7)]
        res = []
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
        # [w.save() for w in self.workers]

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

    def save(self):
        self.global_network.save()

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
        self.env = UnityEnvironment(file_name="EXE\CRML", seed=1, side_channels=[], worker_id=int(self.name)) ## work_id need to be int 
        self.env.reset()
        self.local_network.reset()
        self.pull()
        self.l_ep = 0
        self.behavior = list(self.env.behavior_specs)[0]

        while self.g_ep.value < NUM_GAMES:
            done = False
            self.env.reset()
            self.local_network.reset()
            self.l_step = 0
            
            score = 0
            best_score = 0
            current_score = 0
            
            while not done:
                decision_steps, terminal_steps = self.env.get_steps(self.behavior)
                step = None
                if len(terminal_steps) != 0:
                    step = terminal_steps[terminal_steps.agent_id[0]]
                    state = np.array(step.obs) # [:,:49] ## Unity return
                    # FIXME: Adjust the shape for different state size
                    state = np.vstack((
                        state[42:49],
                        state[35:42],
                        state[28:35],
                        state[21:28],
                        state[14:21],
                        state[7:14],
                        state[0:7],
                    ))
                    state = state.reshape(1,7,7)
                    state = state[:,2:7,:]
                    done = True
                    action, value = self.local_network.take_action(state)
                    # self.local_network.entropies.append(10)
                    # self.local_network.log_probs.append(10)
                else:
                    step = decision_steps[decision_steps.agent_id[0]]
                    # Add noise
                    state = np.array(step.obs) + np.random.rand(*np.array(step.obs).shape) if self.g_ep.value < 1000 else np.array(step.obs)  # [:,:49] ## Unity return
                    # FIXME: Adjust the shape for different state size
                    state = np.vstack((
                        state[42:49],
                        state[35:42],
                        state[28:35],
                        state[21:28],
                        state[14:21],
                        state[7:14],
                        state[0:7],
                    ))
                    state = state.reshape(1,7,7)
                    state = state[:,2:7,:]
                    action, value = self.local_network.take_action(state)

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
                score += reward
                self.local_network.record(state, action, reward, value.detach())
                
                # Do the gradient descent but not update global network directly
                if (self.l_step % MAX_STEP == 0 and self.l_step != 0) or done == True:
                    # print(f'For debug usage: self.l_ep={self.l_ep}')
                    loss = self.local_network.calc_loss_v2(done)
                    loss.requires_grad = True
                    # self.loss_queue.put(loss.detach().numpy()) #record loss
                    self.local_network.reset()
                    loss.backward()
                    

                # update global network (gradient accumulation!)
                if (self.l_ep % MAX_EP == 0):
                    self.push()   
                    self.optimizer.step()
                    self.pull()
                    self.optimizer.zero_grad()
                # if (self.l_ep % MAX_EP == 0 and self.l_ep != 0) or done == True: 
                #     loss = self.local_network.calc_loss_v2(done)
                #     self.optimizer.zero_grad()
                #     loss.backward(retain_graph=True)
                #     self.push()   
                #     self.optimizer.step()
                #     self.pull()

                self.env.step()

                self.l_step += 1
            
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                self.l_ep       += 1
            
            self.res_queue.put(score)
            self.score_queue.put(best_score)

            print(f'Worker {self.name}, episode {self.g_ep.value}, reward {score}')

        self.res_queue.put(None)
        self.score_queue.put(None)
        self.loss_queue.put(None)
        self.save()

    def save(self):
        """
        Save the current model
        """
        self.local_network.save()


if __name__ == '__main__':
    None