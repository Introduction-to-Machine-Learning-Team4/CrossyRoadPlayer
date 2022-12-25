from model import Agent

def train():
    try:
        agent = Agent(67, 5)
        agent.start()
        print('checkpoint1')
        agent.join()
        print('checkpoint2')
    except Exception as e:
        agent.close()
        raise e
        

if __name__ == "__main__":
    train()