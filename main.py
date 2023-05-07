from catch import CatchEnv
import random
import numpy as np
import cv2
from dqnAgent import DQNagent

if __name__ == "__main__":

    # Create catch environment
    env = CatchEnv()

    # Get number of legitimate actions
    num_actions = env.get_num_actions()
    # Get state shape
    state_shape = env.state_shape()

    print('Actions:', num_actions)
    print('\nState shape:', state_shape)

    # Parameters
    num_episodes = 10
    batch_size = 32
    render = False
    update_step = 4

    # Define agent
    agent = DQNagent(state_shape, num_actions)

    for episode in range(num_episodes):
        env.reset()

        state, reward, terminal = env.step(1)
        print('EPISODE ', episode)

        # state(84,84,4) -> (1,4,84,84)
        state = np.transpose(state, (2, 0, 1))
        state = np.expand_dims(state, axis=0)

        # play one episode
        while not terminal:

            # get the action from the agent
            print('\nSTATE SHAPE BEFORE ACT:', state.shape)
            action = agent.act(state)
            # make a step with the corresponding action
            next_state, reward, terminal = env.step(action)
            print('\nNEXT STATE SHAPE AFTER STEP:', next_state.shape)

            # next_state(84,84,4) -> (1,4,84,84)
            next_state = np.transpose(next_state, (2, 0, 1))
            next_state = np.expand_dims(next_state, axis=0)
            print('\nNEXT STATE SHAPE AFTER TRANSPOSE:', next_state.shape)

            # add trajectory to the experience replay memory
            agent.remember(state, action, reward, next_state, terminal)
            # update state
            state = next_state
            # state = np.squeeze(next_state)
            # print('\ STATE SHAPE AFTER SQUEEZE:',state.shape)

            # Render steps
            if render:
                cv2.imshow('state', cv2.resize(state[:, :, 0], (84 * 4, 84 * 4)))
                cv2.waitKey(150)

            print("Reward obtained by the agent: {}".format(reward))
            # state = np.squeeze(state)

            if terminal:
                # every REM_STEP update target model
                if episode % update_step == 0:
                    agent.update_target_model()

            agent.replay(batch_size)

        print("End of the episode")


