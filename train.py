from model import Agent

NUM_GAMES = 30
MAX_EP = 5

def train():
    try:
        agent = Agent(30, 5)
        agent.run()
        agent.save()
    except Exception as e:
        agent.close()
        raise e
        

if __name__ == "__main__":
    train()
