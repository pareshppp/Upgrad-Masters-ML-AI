# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = 
        self.state_init = 

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        (x,t,d)
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch1(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        (x,t,d) , (p,q)
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)








        if requests >15:
            requests = 15
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
