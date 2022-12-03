from model import Agent

def train():
    try:
        agent = Agent(30, 5)
        agent.start()
        agent.save()
    except Exception as e:
        agent.close()
        raise e
        

if __name__ == "__main__":
    train()