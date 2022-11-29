from .model import Agent

NUM_GAMES = 300
MAX_EP = 5

def train():
    agent = Agent(30, 60, 5)
    agent.run()

if __name__ == "__main__":
    train()
