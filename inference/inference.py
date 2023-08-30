import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.optim as optim
import csv

import os

# neural network for policy learning
class simple_policy(nn.Module):

    def __init__(self, input_size, output_size):
        super(simple_policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, output_size)



    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
    
# neural network for behavior cloning
class behavior_nn(nn.Module):

    def __init__(self, input_size, output_size):
        super(simple_policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# neural network for model learning
class dynamic_nn(nn.Module):

    def __init__(self, in_size, out_size):
        super(dynamic_nn, self).__init__()
        self.fc1 = nn.Linear(in_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, out_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class inf_process:

    # initialise the class attributes
    def __init__(self):        

        # define actions of other agents
        self.actions = {1:'move',2:'shoot',3:'dribble',4:'pass',5:'defend',6:'reduce angle',7:'gotoball',8:'markplayer',
                   9:'orient',10:'catch'}
        self.target = [0,0,0,0,0,0,0,0,0,0]
        self.action = None
        self.inf_self.features = None

        # get the self.features to be used for inference of other agents/models
    def get_inf_features(self,state,agent_no):

        if agent_no < 3: # offense agent
            self.features.append(state[0]) # AHA x
            self.features.append(state[1]) # AHA y
            self.features.append(state[8]) # AHA goal open angel
            self.features.append(state[9]) # AHA distance to nearest opponent
            self.features.append(state[19]) # t1 x
            self.features.append(state[20]) # t1 y
            self.features.append(state[10]) # t1 goal open angel
            self.features.append(state[13]) # t1 distance to nearest opp
            self.features.append(state[22]) # t2 x
            self.features.append(state[23]) # t2 y
            self.features.append(state[11]) # t2 goal open angel
            self.features.append(state[14]) # t2 distance to nearest opp
            self.features.append(state[25]) # t3 x
            self.features.append(state[26]) # t3 y
            self.features.append(state[12]) # t3 goal open angel
            self.features.append(state[15]) # t3 distance to nearest opp   
            self.features.append(state[3]) # ball x
            self.features.append(state[4]) # ball y
            self.features.append(state[28]) # d1 x
            self.features.append(state[29]) # d1 y
            self.features.append(state[31]) # d2 x
            self.features.append(state[32]) # d2 y
            self.features.append(state[34]) # d3 x
            self.features.append(state[35]) # d3 y
            self.features.append(state[37]) # d4 x
            self.features.append(state[38]) 
            self.features.append(state[40]) 
            self.features.append(state[41]) 
        else: 
            self.features.append(state[40])
            self.features.append(state[41])
            self.features.append(state[28])
            self.features.append(state[29])
            self.features.append(state[31])
            self.features.append(state[32])
            self.features.append(state[34])
            self.features.append(state[35])
            self.features.append(state[37])
            self.features.append(state[38])
            self.features.append(state[3]) # ball x
            self.features.append(state[4]) # ball y
            self.features.append(state[19]) 
            self.features.append(state[20]) 
            self.features.append(state[22]) 
            self.features.append(state[23]) 
            self.features.append(state[25]) 
            self.features.append(state[26]) 
            self.features.append(state[0]) 
            self.features.append(state[1]) 
        return self.features

    # collect the data for the previous step
    def data_collector(pre_state,actions): # --> actions is an array of actions taken by each agent


        # process the data 
        datapoint = []
        for i in actions:
            for j in pre_state:

                datapoint.append(pre_state[j])
            
            datapoint.append(actions[i])

        # create a csv file and write data to it
        csv_file = 'data.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(datapoint)

        return datapoint


class policyValueNet(nn.Module):

    def __init__(self):

        pol = simple_policy(48,4)


        # policy value layers
        super(dynamic_nn, self).__init__()
        self.fc11 = nn.Linear(48+1+1, 64)
        self.relu1 = nn.ReLU()
        self.fc21 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.fc31 = nn.Linear(64, 1)

        # policy head layers
        self.fc12 = nn.Linear(48+1+1, 64)
        self.relu2 = nn.ReLU()
        self.fc22 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc32 = nn.Linear(64, 1)

    def policy_head(self,x):

        x = self.fc11(x)
        x = self.relu1 
        x = self.fc21(x) 
        x = self.relu1(x) 
        x = self.fc31(x)

        return x
    
    def policy_value(self,x):

        x = self.fc12(x)
        x = self.relu2 
        x = self.fc22(x) 
        x = self.relu2(x) 
        x = self.fc32(x)

        return x



def get_predictions(state):

    # Initialize a model instance
    model = simple_policy(state)

    # Load the saved state dictionary
    cwd = os.getcwd()
    cwd = str(cwd)
    saved_state_dict = t.load(cwd+'Adhoc_Architecture/inference/models/saved_model.pth')

    # Update the model's state dictionary
    model.load_state_dict(saved_state_dict)

    # Set the model to evaluation mode
    model.eval()

    # return the predictions
    with t.no_grad():
        output = model(state)

    return output


        
        


























