# base libbraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# learning libraries
import torch as tt
import torch.nn as nn
import torch.optim as optim

# custom libraries
import action
import inference


class teammate():

    # initialize the feature values of teammate
    def __init__(self,tmno):

        self.goal_angle = 0
        self.proximity_op = 0
        self.pass_angle = 0
        self.x_pos = 0
        self.y_pos = 0
        self.grid = []
        self.uniform_num = 0
        self.action = None
        self.actions = np.zeroes(tmno)

    # method to infer action of a teammate 
    def t_infer(self):

        # return action for an opponent
        return self.action

    # returns action set for all opponents
    def t_action(self):

        for i in range(self.numopp):

            self.actions[i] = self.sim_action()
        
        return self.actions


        





















