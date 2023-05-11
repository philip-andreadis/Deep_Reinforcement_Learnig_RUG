from catch import CatchEnv
import random
import numpy as np
import cv2
from dqnAgent import DQNagent
#from google.colab.patches import cv2_imshow
import os
import pickle

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
    num_episodes = 2000
    batch_size = 32
    render = False
    update_step = 4
    save_dir = 'models'
    model_name = 'Catch_DQN_CNN_simple_2000.h5'
    path = os.path.join(save_dir, model_name)
    final_rewards = []

    # Define agent
    agent = DQNagent(state_shape, num_actions)

    for episode in range(num_episodes):
        env.reset()
        terminal = False
        #TODO: Is it okay to go left everytime we start the game? Maybe a random action would be better?
        state, reward, terminal = env.step(1)
        print('\n---------------------------------')
        print('EPISODE', episode)

        # state(84,84,4) -> (1,4,84,84)
        state = np.transpose(state, (2, 0, 1))
        state = np.expand_dims(state, axis=0)

        # play one episode
        while not terminal:

            # get the action from the agent
            # print('\nSTATE SHAPE BEFORE ACT:',state.shape)
            action = agent.act(state)
            # make a step with the corresponding action
            next_state, reward, terminal = env.step(action)
            # print('\nNEXT STATE SHAPE AFTER STEP:',next_state.shape)

            # next_state(84,84,4) -> (1,4,84,84)
            next_state = np.transpose(next_state, (2, 0, 1))
            next_state = np.expand_dims(next_state, axis=0)
            # print('\nNEXT STATE SHAPE AFTER TRANSPOSE:',next_state.shape)

            # add trajectory to the experience replay memory
            agent.remember(state, action, reward, next_state, terminal)
            # update state
            state = next_state
            # state = np.squeeze(next_state)
            # print('\ STATE SHAPE AFTER SQUEEZE:',state.shape)

            # Render steps
            if render:
                img = np.squeeze(state)
                img = np.transpose(img, (1, 2, 0))
                # ~~if want to render pass img to imshow..
                # cv2.imshow('state', cv2.resize(state[:,:,0], (84*4,84*4)))
                # cv2.waitKey(150)
                img = np.squeeze(state)
                img = np.transpose(img, (1, 2, 0))
                # imshow(img)
                # cv2_imshow(img)

            print("\nReward obtained by the agent: {}".format(reward))
            # state = np.squeeze(state)

            # end of episode
            if terminal:
                # every update_step update target model
                # TODO check if the number of updates should be this small.
                if episode % update_step == 0:
                    agent.update_target_model()

                # keep track of final rewards
                print("\nFINAL reward obtained by the agent: {}".format(reward))
                final_rewards.append(reward)

                # save model at the end of each episode
                print('\nsaving model of episode {} in {}'.format(episode, path))
                agent.save(path)

            agent.replay(batch_size)

        print("\nEnd of the episode")

    # save final rewards in pickle
    with open('rewards.pkl', 'wb') as f:
        pickle.dump(final_rewards, f)


