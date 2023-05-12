import numpy as np
from catch import CatchEnv


def test_model(model):
    # Create catch environment
    env = CatchEnv()

    # Get number of legitimate actions
    num_actions = env.get_num_actions()
    # Get state shape
    state_shape = env.state_shape()

    # params
    episodes = 10

    rewards = 0

    for e in range(episodes):
        state = env.reset()
        terminal = False

        print('\n---------------------------------')
        print('TEST EPISODE', e)

        state = np.transpose(state, (2, 0, 1))
        state = np.expand_dims(state, axis=0)

        while not terminal:
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

    avg_reward = rewards / episodes
    print('Average reward:', avg_reward)
    return avg_reward
