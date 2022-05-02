import numpy as np
import gym

# make the environment of a game
env = gym.make('CartPole-v1')
env.render()

# the action space
# print(env.action_space)

# the state space
# print(env.observation_space)


# one sample step
# observation = env.reset()
# print(observation)

# take an action in the environment
# action = env.action_space.sample()
# observation, reward, done, info = env.step(action)
# print(action)
# print(observation)
# print(reward)



# run for a few episodes
# at the start of each episode, reset the environment 
# for i_episode in range(20):
#     observation = env.reset()
#     total_reward = 0

#     # run for t timesteps
#     for t in range(100):

#     	# render the env
#         env.render()

#         # take a random action and get next state, reward etc.
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         total_reward += reward
        
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             print("Total reward {0}".format(total_reward))
#             break
# env.close()