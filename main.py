from catch import CatchEnv
import random
import numpy as np
import cv2
from dqnAgent import DQNagent
import os
import pickle
from matplotlib import pyplot as plt
from testModel import test_model

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
    num_episodes = 2500
    batch_size = 32
    render = True
    update_step = 4
    save_dir = 'models'
    model_name = 'Catch_DQN_CNN_simple_{}.h5'.format(num_episodes)
    path = os.path.join(save_dir, model_name)
    final_rewards = []
    average_rewards = []

    # Define agent
    agent = DQNagent(state_shape, num_actions)

    for episode in range(1, num_episodes+1):
        # Initialize state with random ball position
        state = env.reset()
        terminal = False

        # state, reward, terminal = env.step(1) # not needed

        print('\n---------------------------------')
        print('EPISODE', episode)

        # state(84,84,4) -> (1,4,84,84)
        state = np.transpose(state, (2, 0, 1))
        state = np.expand_dims(state, axis=0)

        # play one episode
        while not terminal:

            # Render steps
            if render:
                img = np.squeeze(state)
                img = np.transpose(img, (1, 2, 0))
                # stacked
                # cv2.imshow('state', cv2.resize(img[:, :, :], (84 * 4, 84 * 4)))
                # individual frames
                # cv2.imshow('state', cv2.resize(img[:,:,0], (84 * 4, 84 * 4)))
                # cv2.imshow('state1', cv2.resize(img[:, :, 1], (84 * 4, 84 * 4)))
                # cv2.imshow('state2', cv2.resize(img[:, :, 2], (84 * 4, 84 * 4)))
                # cv2.imshow('state3', cv2.resize(img[:, :, 3], (84 * 4, 84 * 4)))
                # cv2.waitKey()

            # get the action from the agent
            action = agent.act(state)

            # make a step with the corresponding action
            next_state, reward, terminal = env.step(action)

            # next_state(84,84,4) -> (1,4,84,84)
            next_state = np.transpose(next_state, (2, 0, 1))
            next_state = np.expand_dims(next_state, axis=0)

            # add trajectory to the experience replay memory
            agent.remember(state, action, reward, next_state, terminal)

            # update state
            state = next_state

            print("\nReward obtained by the agent: {}".format(reward))

            # end of episode
            if terminal:
                # every update_step update target model
                # TODO check if the number of updates should be this small.
                if episode % update_step == 0:
                    agent.update_target_model()

                # keep track of final rewards
                print("\nFINAL reward obtained by the agent: {}".format(reward))
                final_rewards.append(reward)

                # Every 10 episodes enter testing mode
                if episode % 10 == 0:
                    # test
                    average_rewards.append(test_model(agent.get_model()))

                    # save model every 10 episodes
                    print('\nsaving model of episode {} in {}'.format(episode, path))
                    agent.save(path)

            agent.replay(batch_size)

        print("\nEnd of the episode")

    # save final rewards in pickle
    with open('rewards_{}.pkl'.format(num_episodes), 'wb') as f:
        pickle.dump(final_rewards, f)

    # # plot moving average of final rewards
    # N = 10 # window size
    # rm = np.convolve(final_rewards, np.ones(N) / N, mode='valid')
    # plt.plot(rm)
    # plt.show()

    # save average rewards as .npy file
    print('avg shape', len(average_rewards))
    np.save('avg_rewards_{}.npy'.format(num_episodes), average_rewards)



