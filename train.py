from model import Agent

NUM_GAMES = 300
MAX_EP = 5

def train():
    try:
        agent = Agent(60, 5)
        agent.run()
    except Exception as e:
        print(str(e))
        agent.close()

if __name__ == "__main__":
    train()
