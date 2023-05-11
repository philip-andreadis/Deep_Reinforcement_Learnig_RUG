from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import os
import random
import numpy as np



class DQNagent():
    def __init__(self,state_shape, num_actions):
        self.state_shape = state_shape # state space in (channels, height, width)
        self.num_actions = num_actions # action space

        # Initialize experience replay
        self.memory = deque(maxlen=2000)

        # Initialize discount factor
        self.gamma = 0.95

        # Initialize exploration rate, decay and floor
        self.epsilon = 1
        self.e_decay = 0.9999
        self.e_floor = 0.05

        # Initialize learning rate
        self.lr = 0.01

        # Build model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()

        # Save settings
        self.save_dir = 'models'
        self.model_name = os.path.join(self.save_dir, "Catch_DQN_CNN_simple")

    def _build_model(self):
        '''
            Returns the compiled keras CNN model used by the agent.

            Returns:
                model (keras model): The compiled CNN model
        '''
        # Define input image shape
        print('\nSTATE SHAPE:',self.state_shape)
        X_input = Input(self.state_shape)
        X = X_input

        # Convolutional layers
        X = Conv2D(64, 5, strides=(3, 3), padding="valid", input_shape=self.state_shape, activation="relu",data_format="channels_first")(X)
        X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
        X = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", data_format="channels_first")(X)
        X = Flatten()(X)
        print('\nFLATTEN SHAPE:',X.shape)
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
        X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with the corresponding number of actions for each environment: num of nodes = num of actions
        X = Dense(self.num_actions, activation="linear", kernel_initializer='he_uniform')(X)

        # Compile model
        model = Model(inputs=X_input, outputs=X, name='Catch_DQN_CNN_basic')
        model.compile(loss="mean_squared_error", optimizer=RMSprop(learling_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        # Ouput model architecture
        # model.summary()

        return model


    def remember(self, state, action, reward, next_state, done):
        '''
            Saves the current state, action and reward and the next state to the Experience replay queue.
        '''
        experience = state, action, reward, next_state, done
        self.memory.append((experience))


    def act(self, state):
        '''
            Returns the chosen action based on the epsilon greedy strategy (exploration vs exploitation).

            Parameters:
                state (np.array): The current state

            Returns:
                action (int): The selected action
        '''
        if np.random.rand() <= self.epsilon:
            # Make a random action (exploration)
            return random.randrange(self.num_actions)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state and take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)) # ~~maybe [0] is needed here~~


    def replay(self,batch_size):
        '''
            Computes the temporal difference target using the prediction of the target network for the max Q value and trains the main
            model based on that and the current state, for the provided batch.

            Parameters:
                batch_size (int): Batch size
        '''
        # Randomly sample minibatch from the deque memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        # print('MINIBATCH LEN ', len(minibatch))
        # print('STATE IN MINIBATCH SHAPE ', minibatch[0][0].shape)
        # print('NEXT_STATE IN MINIBATCH SHAPE ', minibatch[0][3].shape)

        # Initializations
        state = np.zeros((batch_size,) + self.state_shape)
        next_state = np.zeros((batch_size,) + self.state_shape)
        action, reward, done = [], [], []

        # Fill them with each instance of the minibatch
        for i in range(len(minibatch)):
            # state[i] = np.transpose(minibatch[i][0],(2,0,1)) # minibatch state (84,84,4) -> (4,84,84)
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            # next_state[i] = np.transpose(minibatch[i][3],(2,0,1)) # minibatch state (84,84,4) -> (4,84,84)
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        # target_old = np.array(target) # needed for prioritized experience replay

        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)

        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            if done[i]:
                # Episode ends
                target[i][action[i]] = reward[i]
            else:
                # current Q Network selects the action
                # a'_max = argmax_a' Q(s', a')
                a = np.argmax(target_next[i])
                # target Q Network evaluates the action
                # Q_max = Q_target(s', a'_max)
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

                # # the key point of Double DQN
                # # selection of action is from model
                # # update is from target model
                # if self.ddqn:  # Double - DQN
                #     # current Q Network selects the action
                #     # a'_max = argmax_a' Q(s', a')
                #     a = np.argmax(target_next[i])
                #     # target Q Network evaluates the action
                #     # Q_max = Q_target(s', a'_max)
                #     target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                # else:  # Standard - DQN
                #     # DQN chooses the max Q value among next actions
                #     # selection and evaluation of action is on the target Q Network
                #     # Q_max = max_a' Q_target(s', a')
                #     target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=batch_size, verbose=0)

        # Epsilon decays
        if self.epsilon > self.e_floor:
            self.epsilon *= self.e_decay
        print('\nEpsilon:',self.epsilon)


    def update_target_model(self):
        '''
            Update the target model with the main models weights.
        '''
        self.target_model.set_weights(self.model.get_weights())
        return

    def load(self, name):
        '''
            Loads the trained model in name and returns it.
        '''
        self.model = load_model(name)
        return self.model

    def save(self, name):
        '''
            Saves trained model in name.
        '''
        self.model.save(name)
