from model import Agent

def train():
    try:
        agent = Agent(67, 5)
        agent.start()
        agent.join()
        agent.save()
    except Exception as e:
        agent.close()
        raise e
        

if __name__ == "__main__":
    train()