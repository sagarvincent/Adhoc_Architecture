#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import sys, itertools
from hfo import *

def main():
  # Create the HFO Environment
  hfo = HFOEnvironment()
  # Connect to the server with the specified feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)
  for episode in itertools.count():
    status = IN_GAME
    while status == IN_GAME:

      # Grab the state features from the environment
      features = hfo.getState()

      print(f'State features: {features}')
      
      # Take an action
      hfo.act(DASH, 20.0, 0.)

      
      # Advance the environment and get the game status
      status = hfo.step()
    
    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()

if __name__ == '__main__':
  main()
