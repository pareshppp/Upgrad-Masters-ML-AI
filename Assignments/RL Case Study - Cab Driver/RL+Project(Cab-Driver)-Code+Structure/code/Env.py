# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


# p = start location
# q = end location
# X = current location
# T = current Time
# D = current Day of week

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [(X,T,D) for X in range(1, m+1) for T in range(t) for D in range(d)]
        self.state_init = self.state_space[np.random.choice(len(self.state_space))]
        self.time_elapsed = 0

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

#     def state_encod_arch1(self, state):
#         """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
#         # zero vectors
#         # X_enc = np.zeros(shape=(1,m))
#         # T_enc = np.zeros(shape=(1,t))
#         # D_enc = np.zeros(shape=(1,d))
#         X_enc = [0] * m
#         T_enc = [0] * t
#         D_enc = [0] * d
        
#         # get state
#         X, T, D = state
#         X, T, D = (int(X), int(T), int(D))
        
#         # update index
#         X_enc[X] = 1
#         T_enc[T] = 1
#         D_enc[D] = 1
        
#         # concatenate
#         # state_encod = np.concat([X_enc, T_enc, D_enc], axis=1)
#         state_encod = X_enc + T_enc + D_enc
#         state_encod = np.array(state_encod)#.reshape(1, len(state_encod))
        
#         # assert state_encod.shape == (1, m + t + d)
#         # assert state_encod.sum() == 3
        
#         return state_encod

    
    def state_encod_arch1(self, state):
        """
        convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d.
        """

        if not state:
            return

        state_encod = [0] * (m + t + d)

        # encode location
        state_encod[state[0] - 1] = 1

        # encode hour of the day
        state_encod[m + state[1]] = 1

        # encode day of the week
        state_encod[m + t + state[2]] = 1

        return state_encod

#     # Use this function if you are using architecture-2 
#     def state_encod_arch2(self, state, action):
#         """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        
        
#         return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        
        map_location_poisson_lambda = {
            1: 2,
            2: 12,
            3: 4,
            4: 7,
            5: 8
        }
        # if location == 0:
        #     requests = np.random.poisson(2)

        requests = np.random.poisson(map_location_poisson_lambda[location])

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        # offline
        if (0, 0) not in actions:
            actions.append((0,0))
            possible_actions_index.append(20)

        return possible_actions_index, actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        # if offline
        if action == (0, 0):
            reward = -1 * C
        else:
            # get state and acion values
            X,T,D = state
            X, T, D = (int(X), int(T), int(D))
            p,q = action
            # if current and start location are different
            if X != p:
                # time taken to go from current loc to start location
                duration_curr_start = Time_matrix[X-1][p-1][T][D]
                # update T, D
                T, D = self.update_time_day(duration_curr_start, T, D)
            else:
                duration_curr_start = 0
            # time taken to go from start loc to end location
            duration_start_end = Time_matrix[p-1][q-1][T][D]
            # calculate reward
            reward = (R * duration_start_end) - C * (duration_start_end + duration_curr_start)
        return reward


    def update_time_day(self, travel_time, T, D):
        """Takes in travel_time, current time and current day as input and returns updated time and day"""
        assert travel_time <= 24, "Travel time is more than 24 hours."
        # update time
        T += travel_time
        # if next day
        if T >= 24:
            D += 1
            T = T - 24
        # reset day of week
        if D == 7:
            D = 0
            
        return int(T), int(D)

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        # self.time_elapsed += t1+t2
        # new_state[0] = action[1]
        # new_state[1] = (state[1]+t1+t2)%24
        # new_state[2] = (state[2]+ if (state[1]+t1+t2)<24 then 0 else 1)%7
        # terminal = self.time_elapsed>=720
        
        p, q = action
        X, T, D = state
        X, T, D = (int(X), int(T), int(D))
        
        # if offline
        if action == (0, 0):
            # update time and day with 1 hour
            T, D = self.update_time_day(1, T, D)
            # update total time elapsed
            self.time_elapsed += 1
            
        # if current location is not start location
        # then first move from current location to start location
        if not X == p:
            travel_time = Time_matrix[X-1][p-1][T][D]
            # update time and day
            T, D = self.update_time_day(travel_time, T, D)
            # update current loc to start loc
            X = p
            # update total time elapsed
            self.time_elapsed += travel_time
        
        # move from start location to end location
        travel_time = Time_matrix[X-1][q-1][T][D]
        # update time and day
        T, D = self.update_time_day(travel_time, T, D)
        # update current loc to end loc
        X = q
        # update total time elapsed
        self.time_elapsed += travel_time
        
        # get next state
        next_state = (int(X), int(T), int(D))
        
        # check for terminal
        terminal = self.time_elapsed >= 720
        
        return next_state, terminal




    def reset(self):
        # reset the accumulated hours, this includes hours for waiting with action (0, 0)
        self.time_elapsed = 0
        
        # re-set the initial state to a random one at begenining of each episode
        self.state_init = self.state_space[random.randint(0, len(self.state_space)-1)]
        
        return self.action_space, self.state_space, self.state_init
