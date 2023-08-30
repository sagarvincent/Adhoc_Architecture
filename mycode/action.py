# support libraries
from csv import writer
import subprocess

# base libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# learning libraries
import torch as tt
import torch.nn as nn
import torch.optim as optim

# agent files
import teammate
import adhoc
import opponent

# import dependancy 
import utils

# set path to asp files
pre_asp_learner = '/home/sagarvincent/projects/HFO/Adhoc_Architecture/ASP/learner_pre.sp'
asp_learner = '/home/sagarvincent/projects/HFO/Adhoc_Architecture/ASP/learner.sp'
display_marker = 'display'

class actor:

    def __init__(self,states):

        # extract information from states
        self.xpos = states[1]
        self.ypos = states[2]
        self.maxopenangle = states[3]
        self.oppdist = states[4]
        self.teamopenangle = states[5]
        self.maxpassangle = states[6]
        self.action = None


    ## -- logical action taken by the agent -- ##
    def logic_action(self,state):

        # if the agent has the ball 
        if int(state[5]) == 1:

            # open the pre_asp file and load the code
            reader = open(pre_asp_learner, 'r')
            pre_asp = reader.read()
            reader.close()

            # split the ASP code
            pre_asp_split = pre_asp.split('\n')
            # gets the index of the 'display' command
            display_marker_index = pre_asp_split.index(display_marker)

            input1,input2 = self.query_terms(state)

        # if agent doesn't have the ball
        else:
            ball_pos = np.array([state[3], state[4]])
            teammates_pos = [np.array([state[19], state[20]]), np.array([state[22], state[23]]), np.array([state[25], state[26]])]
            team_ball_dist = []
            teammate_y = []
            for teammate in teammates_pos:
                team_ball_dist.append(np.linalg.norm(teammate - ball_pos))
                teammate_y.append(teammate[1])
            if any(num < 0.1 for num in team_ball_dist):
                # move to a better position to recieve the ball
                next_actions = self.get_action_terms(state)
                if any(y < 0 for y in teammate_y):
                    actions = (5, 0.4, 0.3)
                else:
                    actions = (5, 0.4, -0.3)
            else:
                actions = 8 # 8 for aut and axiom?others 7
            


    # returns the program for querying the ASP program
    def query_terms(self,state,able_to_kick,step):

        grid = utils.get_gridno(state[0],state[1])
        input = get_ASp



    # Get ASP terms for the ad hoc agent
    def get_ASP_terms(input, agent_name, step, grid,able_to_kick):

        input.append('holds(in('+agent_name+','+str(grid[0])+','+str(grid[1])+'),'+str(step)+').')
        input.append('holds(ball_in('+str(grid[0])+','+str(grid[1])+'),'+str(step)+').')
        
        # if in possession
        if able_to_kick == 1.0:
            input.append('holds(has_ball(learner),'+str(step)+').') # agent has the ball
        else:
            input.append('-holds(has_ball(learner),'+str(step)+').')
        return input
        
    # Get ASP terms for the teammates and opponents
    def get_ASP_terms_other(input, step, agent, teammates, opponents):
        # for agent in teammates: currently do this only for the nearest teammate
        input.append('holds(agent_in(offense2,'+str(teammates[0].grid[0])+','+str(teammates[0].grid[1])+'),'+str(step)+').')
        for i in range(len(opponents)):
            if agent.grid[0] < 13: # cannot shoot
                if (agent.grid[0]-1 <= opponents[i].grid[0] <= agent.grid[0]+1) and (agent.grid[1]-1 <= opponents[i].grid[1] <= agent.grid[1]+1):
                    input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i].grid[0])+','+str(opponents[i].grid[1])+'),'+str(step)+').')
            else:
                input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i].grid[0])+','+str(opponents[i].grid[1])+'),'+str(step)+').')
        return input

    def get_future_ASP_terms(input, step, agent, locations, uniform_numbers, teammate_uniform_num):
        # [[x_A,y_A], [xt1,yt1], [xt2,yt2], [xdg,ydg], [xd1,yd1], [xd2,yd2]] 7.0 8.0 1.0 2.0 3.0
        # nearest teammate
        idx = uniform_numbers.index(teammate_uniform_num)
        grid_nearest_teammate = utils.get_gridno(locations[idx][0], locations[idx][1])
        input.append('holds(agent_in(offense2,'+str(grid_nearest_teammate[0])+','+str(grid_nearest_teammate[1])+'),'+str(step)+').')
        opponents = utils.process_locations_to_opponents(locations, uniform_numbers, True)
        for i in range(len(opponents)):
            if agent.grid[0] < 13: # cannot shoot
                if (agent.grid[0]-1 <= opponents[i][0] <= agent.grid[0]+1) and (agent.grid[1]-1 <= opponents[i][1] <= agent.grid[1]+1):
                    input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i][0])+','+str(opponents[i][1])+'),'+str(step)+').')
            else:
                input.append('holds(agent_in(defense'+str(i+1)+','+str(opponents[i][0])+','+str(opponents[i][1])+'),'+str(step)+').')
        return input
                

    ## -- infer actions of other agents to give to the ASP program -- ##


    ## -- get state terms to pass to asp program -- ##


    ## -- run asp program to get answers -- ##


    ## -- map answer set to actions -- ##


      




















