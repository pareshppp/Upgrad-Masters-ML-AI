import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

# for plumbing code
import collections
from collections import deque
import pickle

# the environment
import gym

# the Agent
from Agent import CartpoleAgent


# breakout environment
env = gym.make('CartPole-v0')

# get size of state and action from environment
state_size = env.observation_space.shape[0] # equal to 4 in case of cartpole 
action_size = env.action_space.n            # equal to 2 in case of cartpole

# agent needs to be initialised outside the loop since the DQN
# network will be initialised along with the agent
agent = CartpoleAgent(action_size=action_size, state_size=state_size)


# to store rewards in each episode
rewards_per_episode, episodes = [], []

# make dir to store model weights
if not os.path.exists("saved_model_weights"):
    os.mkdir("saved_model_weights")

# n_episodes
n_episodes = 400

#### simulation starts ####
for episode in range(n_episodes):

    done = False
    score = 0

    # reset at the start of each episode
    state = env.reset()

    while not done:
        env.render()

        # get action for the current state and take a step in the environment
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        # save the sample <s, a, r, s', done> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)

        # train after each step
        agent.train_model()

        # add reward to the total score of this episode
        score += reward
        state = next_state


    # store total reward obtained in this episode
    rewards_per_episode.append(score)
    episodes.append(episode)

    # epsilon decay
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    # every episode:
    print("episode {0}, reward {1}, memory_length {2}, epsilon {3}".format(episode,
                                                                         score,
                                                                         len(agent.memory),
                                                                         agent.epsilon))
    # every few episodes:
    if episode % 10 == 0:
        # store q-values of some prespecified state-action pairs
        # q_dict = agent.store_q_values()

        # save model weights
        agent.save_model_weights(name="model_weights.h5")

#### simulation complete ####

# save stuff as pickle
def save_pickle(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# make directory
if not os.path.exists("saved_pickle_files"):
    os.mkdir("saved_pickle_files")

# save rewards_per_episode
save_pickle(rewards_per_episode, "saved_pickle_files/rewards_per_episode")


# plot results
with open('saved_pickle_files/rewards_per_episode.pkl', 'rb') as f:
    rewards_per_episode = pickle.load(f)

plt.plot(list(range(len(rewards_per_episode))), rewards_per_episode)
plt.xlabel("episode number")
plt.ylabel("reward per episode")

# save plots in saved_plots/ directory
plt.savefig('rewards.png')

print("Average reward of last 100 episodes is {0}".format(np.mean(rewards_per_episode[-100:]))) 
















