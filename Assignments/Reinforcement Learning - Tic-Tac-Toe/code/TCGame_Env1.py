from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        # check rows
        if (curr_state[0] + curr_state[1] + curr_state[2]) == 15 or \
            (curr_state[3] + curr_state[4] + curr_state[5]) == 15 or \
            (curr_state[6] + curr_state[7] + curr_state[8]) == 15:
                return True
        
        # check cols
        if (curr_state[0] + curr_state[3] + curr_state[6]) == 15 or \
            (curr_state[1] + curr_state[4] + curr_state[7]) == 15 or \
            (curr_state[2] + curr_state[5] + curr_state[8]) == 15:
                return True
            
        # check diagonals
        if (curr_state[0] + curr_state[4] + curr_state[8]) == 15 or \
            (curr_state[6] + curr_state[4] + curr_state[2]) == 15:
                return True
            
        return False
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        position, value = curr_action
        next_state = curr_state.copy()
        next_state[position] = value
        assert len(curr_state) == 9
        assert len(next_state) == 9
        assert next_state != curr_action
        return next_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        # -state_transition(curr_state, curr_action)
        # -Next_state
        # -IsTerminal(Next_state)
        #     Win
        #         reward = 10
        #     tie
        #         reward = 0        
        # -else
        # 	## Env Move
        #     ## action_space
        #     -RandomAction
        #     -state_transition(Next_state, random_action)
        #     -Updated_Next_state
        #     -IsTerminal(Updated_Next_state)
        #         Win
        #             reward = -10
        #         tie
        #             reward = 0
        #     -else
        #     	reward = -1
        # reward, Next_state/Updated_Next_state, terminal
        
        Next_state = self.state_transition(curr_state, curr_action)
        
        terminal = self.is_terminal(Next_state)
        
        if terminal[1] == 'Win':
            reward = 10
        elif terminal[1] == 'Tie':
            reward = 0
        else: # Resume for Env move
            
            env_action_space = [x for x in self.action_space(Next_state)[1]]
            next_action = tuple(np.asarray(env_action_space)[np.random.choice(len(env_action_space), 1)][0])
            
            Next_state = self.state_transition(Next_state, next_action)
            
            terminal = self.is_terminal(Next_state)
        
            if terminal[1] == 'Win':
                reward = 10
            elif terminal[1] == 'Tie':
                reward = 0
            else:
                reward = -1 # Resume gets -1 reward per extra timestep
                
        return (Next_state, reward, terminal[0])


    def reset(self):
        return self.state
