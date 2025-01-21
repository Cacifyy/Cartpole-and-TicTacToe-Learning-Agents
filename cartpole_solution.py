import gymnasium as gym
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from collections import deque

# constants
N_EPISODES = 200  # you can change this if you like
ALPHA = 0.005  # learning rate
GAMMA = 0.99  # discount factor
EPSILON = 1.0  # initial exploration factor
EPSILON_MIN = 0.001  # minimum value for epsilon
EPSILON_DECAY = 0.995  # decay rate for epsilon
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
POLY_DEGREE = 2  # Degree for polynomial features

# Experience replay buffer
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Polynomial feature transformer
poly = PolynomialFeatures(degree=POLY_DEGREE)

def featurize_state(state):
    # Transform the state using polynomial features to increase expressiveness
    state = np.array(state).reshape(1, -1)
    return poly.fit_transform(state)

# instantiate the environment object (set render_mode="human" to visualize the simulation)
env = gym.make("CartPole-v1", render_mode="none")

# list to store episode lengths
episode_lengths = []

# instantiate the history data structures - one for each action
histories = [
    {'state': [], 'reward': []},
    {'state': [], 'reward': []}
]

models = [
    SGDRegressor(learning_rate='constant', eta0=ALPHA),
    SGDRegressor(learning_rate='constant', eta0=ALPHA)
]

# Initialize models with dummy data
initial_state = featurize_state([0, 0, 0, 0])
for model in models:
    model.partial_fit(initial_state, [0])

for episode in range(N_EPISODES):

    # reset the environment for this episode
    state, _ = env.reset()


    # allow 1000 steps before timing out the episode
    for step in range(1000):

        # choose action using epsilon-greedy policy
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, 1)
        else:
            features = featurize_state(state).astype(np.float64)  # Features are changed to float64 type
            q_values = [model.predict(features)[0] for model in models]
            action = np.argmax(q_values)

        # execute the chosen action
        new_state, reward, terminated, truncated, info = env.step(action)

        # default reward is always +1. Let's assign -1 reward if the episode terminates (due to pole falling)
        reward = -1 if terminated else +1

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, new_state, terminated))

        # Sample random mini-batch from replay buffer
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            for b_state, b_action, b_reward, b_new_state, b_terminated in batch:
                b_features = featurize_state(b_state).astype(np.float64)
                b_next_features = featurize_state(b_new_state).astype(np.float64)
                b_next_q_values = [model.predict(b_next_features)[0] for model in models]
                b_best_next_q = np.max(b_next_q_values) if not b_terminated else 0
                b_target = b_reward + GAMMA * b_best_next_q
                models[b_action].partial_fit(b_features, [b_target])

        # start the next episode if this one terminated
        if terminated:
            episode_lengths.append(step)
            break
        else:  # otherwise, update the state and repeat
            state = new_state

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f'episode {episode} lasted {episode_lengths[-1]} steps')

# save episode lengths
pd.Series(name='episode_length', data=episode_lengths).to_csv('episode_lengths.csv')

# plot episode lengths
plt.plot(episode_lengths)
plt.xlabel('episode')
plt.ylabel('length (steps)')
plt.show()
