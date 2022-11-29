import torch.multiprocessing as mp
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from .model import Network
from .train import NUM_GAMES, MAX_EP

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
        self.env = UnityEnvironment(file_name="CRML", seed=1, side_channels=[])
    
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

    def run(self):
        self.l_ep = 0
        while self.global_network.value < NUM_GAMES:
            done = False
            state = self.env.reset()
            score = 0
            self.local_network.reset()
            while not done:
                action = self.local_network.take_action(state)
                # FIXME: not compatible with current unity env., som slight adjustment is needed
                state_new, reward, done = self.env.set_actions(action) 
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
                state = state_new
            with self.g_ep.get_losk():
                self.g_ep.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)