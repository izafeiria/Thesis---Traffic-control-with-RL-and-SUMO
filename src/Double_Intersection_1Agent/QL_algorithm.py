### QL algorithm for 1 agent system of 2-intersections network

import numpy as np
import pandas as pd

class Testing:
        
    def choose(self, q_table, state, action_space,ts):
        
        if str(state) not in q_table:
            action = int(action_space.sample()) 

        else:
            act = np.argmax(q_table[str(state)])
            

            if act == 3 :
                action = 1
            if act == 0 and ts == '1':
                action = 0
            if act == 0 and ts == 'gneJ13' :
                  action = 1
            if act == 1 and ts=='1' :
                  action = 1
            if act == 1 and ts == 'gneJ13':
                action = 0
            if act == 2:
                action = 0

            
        return action
        
class EpsilonGreedy:

    def __init__(self, initial_epsilon, min_epsilon, decay):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
    
    def choose(self, q_table, state, action_space,ts):
        
      
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())

        else:
              if str(state) not in q_table:
                  action = int(action_space.sample())
                  print('random', action)
              else:
                  act = np.argmax(q_table[str(state)])

                  if act == 0 :
                      action = 1
                  if act == 3 and ts == '1':
                      action = 1
                  if act == 3 and ts == 'gneJ13' :
                        action = 0
                  if act == 2 and ts=='1' :
                        action = 0
                  if act == 2 and ts == 'gneJ13':
                      action = 1
                  if act == 1:
                      action = 0
        

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)  
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon

class QLAgent:

    def __init__(self, starting_state, state_space, action_space, alpha, gamma, exploration_strategy, qtable):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        
        if qtable == None:
            self.q_table = {str(self.state): [0 for _ in range(4)]}
        else: 
            self.q_table = qtable
            
        
        self.exploration = exploration_strategy
        self.acc_reward = 0
        
    def act(self,ts):
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space,ts)
        return self.action

    def learn(self, next_state, reward, done=False):

        if str(next_state) not in self.q_table:
            self.q_table[str(next_state)] = [0 for _ in range(4)]


        s = str(self.state)
        s1 = str(next_state)
        a = self.action
   

        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward