from src.model import Agent
from src.ddqn import *
import sys

def train(mode='a3c'):
    if mode == 'a3c':
        try:
            agent = Agent()
            agent.start()
            # print('checkpoint1')
            agent.join()
            # print('checkpoint2')
        except Exception as e:
            agent.save()
            agent.close()
            raise e
    elif mode == 'ddqn':
        ddqn()

        

if __name__ == "__main__":
    arg_len = len(sys.argv)
    if arg_len == 3:
        train(str(sys.argv[1]))
    else:
        train(mode='a3c')