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


class opp:

    def __init__(self,numopp):

        self.numopp = numopp
        self.xpos = np.zeroes(numopp)
        self.ypos = np.zeroes(numopp)
        self.grid = []
        self.uni_no = np.zeroes(numopp)
        self.ballpos = None
        self.actions = np.zeroes(numopp)
        self.action = None

    # call some prediction method to predict the next action of an opponent agent based on current state
    def sim_action(self):



        # return action for an opponent
        return self.action

    # returns action set for all opponents
    def oppaction(self):

        for i in range(self.numopp):

            self.actions[i] = self.sim_action()
        
        return self.actions

        




















