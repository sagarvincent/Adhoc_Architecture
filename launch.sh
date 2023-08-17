#!/bin/bash


/home/sagar/project/HFO/bin/HFO --offense-agents=2 --defense-npcs=1 --no-sync --fullstate &
sleep 5

/home/sagar/project/HFO/adhoc/ad-hoc_agent.py 6000 &
sleep 5

/home/sagar/project/HFO/adhoc/ad-hoc_agent.py 6000 &


# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait







