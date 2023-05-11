from dqnAgent import DQNagent
import numpy as np
from catch import CatchEnv

# Create catch environment
env = CatchEnv()

# Get number of legitimate actions
num_actions = env.get_num_actions()
# Get state shape
state_shape = env.state_shape()

# params
path = 'models/Catch_DQN_CNN_simple_2000'
episodes = 100

# Define agent
agent = DQNagent(state_shape, num_actions)

# Load model
model = agent.load(path)

rewards = 0

for e in range(episodes):
    env.reset()

    state, reward, terminal = env.step(1)
    print('\n---------------------------------')
    print('EPISODE', e)

    state = np.transpose(state, (2, 0, 1))
    state = np.expand_dims(state, axis=0)

    while not terminal:
        print('state:', state.shape)

        action = np.argmax(model.predict(state))
        next_state, reward, terminal = env.step(action)
        rewards += reward
        # next_state(84,84,4) -> (1,4,84,84)
        next_state = np.transpose(next_state, (2, 0, 1))
        next_state = np.expand_dims(next_state, axis=0)
        state = next_state
        if terminal:
            print("\nEnd of episode!")
            print("Reward:", reward)
            
print('Average reward:',rewards/episodes)
