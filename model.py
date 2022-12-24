import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
# from torchsummary import summary
from shared_adam import SharedAdam
import datetime
import os
import random
import math
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
# Seed
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

NUM_GAMES = 10000  # Maximum training episode for master agent
MAX_EP = 10000     # Maximum training episode for slave agent

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
        self.conv_in=7
        self.conv_dim=16
        self.fac = 144
        self.fc1s, self.fc2s = 64, self.state_dim
        # initialize LSTM
        self.lstm = nn.LSTMCell(state_dim, state_dim, bias=False) # (input_size, hidden_size)
        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)
        # h0 = torch.randn(self.hidden_layer_num, self.batch_size, self.hidden_feature_dim)
        # c0 = torch.randn(self.hidden_layer_num, self.batch_size, self.hidden_feature_dim)

        # Actor
        self.action_head = nn.Linear(self.fc2s, action_dim)
        self.critic_head  = nn.Linear(self.fc2s, 1)
        self.net = nn.Sequential(
            # nn.Linear(state_dim, 60),
            # nn.ReLU(),
            # nn.Linear(60, 30),
            # nn.ReLU(),
            # nn.Linear(30, action_dim)
            nn.Conv2d(1,self.conv_dim,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.conv_dim,self.conv_dim,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.fac, self.fc1s),
            nn.ReLU(),
            nn.Linear(self.fc1s , self.fc2s),
            nn.ReLU(),
            nn.Linear(self.fc2s, self.fc2s)
        )
        # self.net_actor_linear = nn.Linear(25, action_dim, bias=False)
        # self.net_actor = nn.Linear(state_dim, action_dim, bias=False)

        # Critic
        # self.net_critic_cnn = nn.Sequential(
        #     # nn.Linear(state_dim, 30),
        #     # nn.ReLU(),
        #     # nn.Linear(30, 1)
        #     nn.Conv2d(1,1,kernel_size=3),
        #     nn.Batch()
        # )
        # self.net_critic_linear = nn.Linear(25, 1, bias=False)

        # nn.init.xavier_normal_(self.net_actor.layer[0].weight)
        # nn.init.xavier_normal_(self.net_critic.layer[0].weight)

        # load models
        if(load == True):
            self.net_actor.load_state_dict(torch.load(path_actor))
            self.net_critic.load_state_dict(torch.load(path_critic))

        self.gamma = gamma
       
        self.states  = []
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
        # lstm
        #(hx, cx) = lstm_par
        # x = state.view(-1, self.state_dim)
        # hx, cx = self.lstm(x, (hx, cx)) # update lstm parameters
        
        # state = hx
        value = self.net(state.reshape((1,1,7,7)))
        value=self.critic_head(value)
        logits = self.net(state.reshape((1,1,7,7)))
        (hx, cx) = lstm_par
        x = state.view(-1, self.state_dim)
        hx, cx = self.lstm(x, (hx, cx)) # update lstm parameters
        logits = hx
        logits=nn.functional.relu(logits)
        logits=self.action_head(logits)


        return logits, value,(hx, cx) # return logits, value and lstm parameters to update

    def record(self, state, action, reward):
        """
        Record <state, action, reward> after taking this action
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def score_record(self, score):
        self.scores.append(score)

    def reset(self):
        """
        Reset the record 
        """
        self.states  = []
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
        pi, v , (hx, cx)= self.forward(state, lstm_par)
        probs = torch.softmax(pi, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().numpy()[0]
        return action, (hx, cx)

    def calc_R(self, done, lstm_par):
        """
        TODO: 
        """
        states = torch.tensor(self.states, dtype=torch.float) # FIXME
        pi, v, (hx, cx) = self.forward(states[-1], lstm_par)

        R = v * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        # for reward in self.scores[::-1]:
            # R = reward + self.gamma * R
            # batch_return.append(R)
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

        # pi, values, _ = self.forward(states, lstm_par) # FIXME:
        values = values.squeeze()
        # print(f'debug: {values.shape} {returns.shape}')
        critic_loss = (returns - values) ** 2

        probs = torch.softmax(pi, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss
    def calc_entropy_loss(self, done, lstm_par):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns, pi, values, _ = self.calc_R(done, lstm_par)
        # pi, values, _ = self.forward(states, lstm_par) # FIXME:
        values = values.squeeze()
        # print(f'debug: {values.shape} {returns.shape}')
        critic_loss = (returns - values) ** 2
        probs = torch.softmax(pi, dim=1)
        probs=probs.detach().numpy()
        probs=np.reshape(probs,(-1,))
        loss = -sum([p*math.log(p+1e-5) for p in probs])
        return loss
    def save(self):
        """
        Save the model parameters into .pt & .txt files
        """
        if not os.path.isdir(f'.\model\{self.timestamp}'):
            os.mkdir(f'.\model\{self.timestamp}')
        
        # torch.save(self.net_actor.state_dict(), f'.\model\{self.timestamp}\{self.name}_actor.pt')
        # torch.save(self.net_critic.state_dict(), f'.\model\{self.timestamp}\{self.name}_critic.pt')
        
        # with open(f'.\model\{self.timestamp}\{self.name}_actor.txt', 'w') as fh:
        #     fh.write("Model's state_dict:\n")
        #     for param_tensor in self.net_actor.state_dict():
        #         fh.write(f'{param_tensor} \t {self.net_actor.state_dict()[param_tensor].size()}')
        
        # with open(f'.\model\{self.timestamp}\{self.name}_critic.txt', 'w') as fh:
        #     fh.write("Model's state_dict:\n")
        #     for param_tensor in self.net_critic.state_dict():
        #         fh.write(f'{param_tensor} \t {self.net_critic.state_dict()[param_tensor].size()}')

        # with open(f'.\model\{self.timestamp}\{self.name}_lstm.txt', 'w') as fh:
        #     fh.write("Model's state_dict:\n")
        #     for param_tensor in self.lstm.state_dict():
        #         fh.write(f'{param_tensor} \t {self.lstm.state_dict()[param_tensor].size()}')
        
        # with open(f'.\model\{self.timestamp}\{self.name}_record.txt', 'w') as fh:
        #     fh.write("Index \t\t action \t reward:\n")
        #     for index, action, reward in zip(range(len(self.rewards)), self.actions, self.rewards):
        #         fh.write(f'{index:<10} \t {action.squeeze():<10} \t {reward.squeeze():<10}\n')

        # Ouput parameters
        # with open(f'.\model\{self.timestamp}\parameters.txt', 'w') as fh:
        #     fh.write(f'timestamp: {self.timestamp}\n')
        #     fh.write(f'state dimension: {self.state_dim}\n') # Input dimension
        #     fh.write(f'action dimension: {self.action_dim}\n') # Output dimension
        #     fh.write(f'Maximum training episode for master agent: {NUM_GAMES}\n')
        #     fh.write(f'Maximum training episode for slave agent: {MAX_EP}\n')
        #     fh.write(f'============================================================\n')
            # fh.write(f'lstm:\n{self.lstm}\n')
            # fh.write(f'actor network:\n{self.net_actor}\n')
            # fh.write(f'critic network:\n{self.net_critic}\n')

class Agent(mp.Process):
    """
    Master agent in A3C architecture
    """
    def __init__(self, state_dim=60, action_dim=5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_stamp = datetime.datetime.now().replace(second=0, microsecond=0).strftime("%Y-%m-%d-%H-%M-%S")
        self.global_network = Network(state_dim, action_dim, gamma=0.95, name='master', 
            timestamp=self.time_stamp,
            load=False, path_actor='.\model\master_actor.pt', path_critic='.\model\master_critic.pt') # global network
        
        self.global_network.share_memory() # share the global parameters in multiprocessing
        self.opt = SharedAdam(self.global_network.parameters(), lr=1e-4, betas=(0.92, 0.999)) # global optimizer
        self.global_ep, self.res_queue = mp.Value('i', 0), mp.Manager().Queue()
        # TODO: add loss queue 
        self.score_queue=mp.Manager().Queue()
        self.loss_queue=mp.Manager().Queue()

    def close(self):
        """
        Close all slave agent created (debugging usage)
        """
        [w.env.close() for w in self.workers]
        pass

    def run(self):
        """
        Initilize all slave agents and start parallel training
        """
        self.workers = [Worker(self.global_network, self.opt, 
                            self.state_dim, self.action_dim, 0.9, 
                            self.global_ep, i, self.global_network.timestamp, self.res_queue,self.score_queue,self.loss_queue) 
                                for i in range(mp.cpu_count() - 0)]
        # parallel training
        [w.start() for w in self.workers]

        # record episode reward to plot
        res = []
        score=[]
        loss=[]
        still_running=True
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                print("ter1")
                break
        
        while True:
            sc = self.score_queue.get()
            if sc is not None:
                score.append(sc)
            else:
                print("ter2")
                break
        while True:
            los = self.loss_queue.get()
            if los is not None:
                loss.append(los)
            else:
                print("ter3")
                break
        print(len(res),len(score),len(loss))
        [w.join() for w in self.workers]
        # [w.save() for w in self.workers]
        print("test2")
        # plot
        import matplotlib.pyplot as plt
        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\ep_reward.png')
        plt.clf()
        plt.plot(score)
        plt.ylabel('game score')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\gamescore.png')
        plt.clf()
        plt.plot(loss)
        plt.ylabel('entropy loss')
        plt.xlabel('Step')
        plt.savefig(f'.\model\{self.time_stamp}\loss.png')
    
    def save(self):
        self.global_network.save()

class Worker(mp.Process): 
    """
    Slave agnet in A3C architecture
    """
    def __init__(self, global_network, optimizer, 
            state_dim, action_dim, gamma, global_ep, name, timestamp, res_queue,score_queue,loss_queue):
        super().__init__()
        self.local_network = Network(state_dim, action_dim, gamma=0.95, name=f'woker{name}', timestamp=timestamp)
        self.global_network = global_network
        self.optimizer = optimizer
        self.g_ep = global_ep       # total episodes so far across all workers
        self.l_ep = None
        self.gamma = gamma          # reward discount factor
        self.res_queue = res_queue
        self.score_queue=score_queue
        self.loss_queue=loss_queue
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.name = f'{name}'
        
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
            best_score=0
            current_score=0
            loss_value=0
            die=0
            act=0
            col=0
            while not done:
                beat_high=0
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
                    state = step.obs ## Unity return
                    done = True
                    die=1
                else:
                    step = decision_steps[decision_steps.agent_id[0]] 
                    state = np.array(step.obs) + np.random.rand(*np.array(np.array(step.obs)).shape) if (self.g_ep.value < 100) else np.array(step.obs) ## Unity return
                    
                    action, (hx, cx) = self.local_network.take_action(state, (hx, cx)) # take actions and update lstm parameters
                    
                    actionTuple = ActionTuple()

                    # if (self.l_ep == 1):
                    #     action = np.array([[1]])
                    # else:
                    #     action = np.asarray([[action]])
                    if action==1:
                        current_score+=1
                        act=1
                    elif action==2:
                        current_score-=1
                        act=-1
                    elif action==3:
                        act=3
                        col-=1
                    elif action==4:
                        act=4
                        col+=1
                    if current_score>best_score:
                        best_score=current_score
                        beat_high=1
                    action = np.asarray([[action]])
                    actionTuple.add_discrete(action) ## please give me a INT in a 2d nparray!!
                    
                    self.env.set_actions(self.behavior, actionTuple)
                reward = step.reward ## Unity return
                # -------- Manual Adjust ---------
                # if(reward >= 1): # Beating Highscore
                #     reward = reward - 0.3
                # elif(reward >= 0.1): # moving forward 15 sec
                #     reward = reward + 0.1
                # ----------------------------------------
                if die==1:
                    reward=-10
                elif beat_high==1:
                    reward=10
                elif act==1:
                    reward=1-((100-current_score)/100)**0.5
                else:
                    reward=0
                score += reward
                self.local_network.record(state, action, reward)
                
                if (self.l_ep % MAX_EP == 0 and self.l_ep != 0) or done == True: 
                    # detach current lstm parameters
                    cx = cx.detach()
                    hx = hx.detach()
                    loss = self.local_network.calc_loss(done, (hx, cx))
                    # if done==True:
                    #     self.loss_queue.put(loss.detach().numpy())          #record loss
                    loss_value=self.local_network.calc_entropy_loss(done,(hx,cx))
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.push()   
                    self.optimizer.step()
                    self.pull()

                self.env.step()
                
            self.l_ep += 1

            # if self.l_ep % MAX_EP == 0 and self.l_ep != 0:
            #     loss = self.local_network.calc_loss(done)
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.push()   
            #     self.optimizer.step()
            #     self.pull()
            # self.local_network.score_record(score)

            with self.g_ep.get_lock():
                self.g_ep.value += 1
            self.loss_queue.put(round(loss_value,3))
            self.res_queue.put(round(score,3))
            self.score_queue.put(round(best_score,3))
            print(f'Worker {self.name}, episode {self.g_ep.value}, reward {score}, score {best_score}, loss {loss_value}')
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