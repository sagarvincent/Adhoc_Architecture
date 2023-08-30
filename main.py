#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8
# First Start the server: $> bin/start.py
import argparse
import itertools
import random
import hfo

from inference import bremen
import csv
import os

import numpy as np
import adhoc
import sklweka.jvm as jvm
try:
  from hfo import *
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000, help="Server port")
  parser.add_argument('--seed', type=int, default=None,
                      help="Python randomization seed; uses python default if 0 or not given")
  parser.add_argument('--rand-pass', action="store_true",
                      help="Randomize order of checking teammates for a possible pass")
  parser.add_argument('--epsilon', type=float, default=0,
                      help="Probability of a random action if has the ball, to adjust difficulty")
  parser.add_argument('--record', action='store_true',
                      help="If doing HFO --record")
  parser.add_argument('--rdir', type=str, default='log/',
                      help="Set directory to use if doing --record")
  args=parser.parse_args()
  
  if args.seed:
    random.seed(args.seed)
    
  hfo_env = HFOEnvironment()
  if args.record:
    hfo_env.connectToServer(HIGH_LEVEL_FEATURE_SET,
                             'bin/teams/base/config/formations-dt', args.port,
                             'localhost', 'base_left', False,
                             record_dir=args.rdir)

  else:
    hfo_env.connectToServer(HIGH_LEVEL_FEATURE_SET,
                             'bin/teams/base/config/formations-dt', args.port,
                             'localhost', 'base_left', False)
  
  num_teammates = hfo_env.getNumTeammates()
  num_opponents = hfo_env.getNumOpponents()
    
  jvm.start() 
  # create a counter for sample no.
  counter = 0
  
  cwd = os.getcwd()
  cwd =str(cwd)

  for episode in itertools.count():
    adhoc_agent = adhoc.adhoc_Agent(num_teammates, num_opponents)
    num_had_ball = 0
    step = 0
    status = IN_GAME
    while status == IN_GAME:
      state = hfo_env.getState()
      state0 = state
      
      action = hfo.MOVE #adhoc_agent.get_action(state, step)  


      if isinstance(action, tuple):
          hfo_env.act(*action)
      else:
          hfo_env.act(action)
      print(f"Action : {action}")
      num_had_ball += 1
      status=hfo_env.step()
      step += 1

      ##### ----- perform inference model "training" ----- #####

      # get the state before, the action taken for that and the resultant state
      pre_state = state0
      action_now = action 
      new_state = hfo_env.getState()

      # proccess data to suitable format

      # create/update both dataset files
      with open(cwd+'/Adhoc_Architecture/data/', "a", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data_to_append)

      
      # check counter to decide whether a sample batch s ready
      if counter == 5:
        print("Sample batch size reached")

        # launch trainer
        trainer = bremen.bremen()
        trainer.bremen_algorithm(pre_state,action_now,new_state,9) # --> trains the network

        counter = 0
      else:
        counter = counter + 1





    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo_env.act(QUIT)
      exit()
  jvm.stop()
if __name__ == '__main__':
  main()

            














































