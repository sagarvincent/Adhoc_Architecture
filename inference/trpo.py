import numpy as np
import inference
import torch as tt
import torch.nn as nn
import torch.optim as optim

from trpo import trpo_agent 



def trpo(num,simple_policy,sample_trajectory):

    ta = trpo_agent.TRPO(simple_policy)


    # for loop
    for i in range(num):

        # define hyper params
        T = 0  # Number of timesteps
        gamma = 0  # Discount factor
        damping = 0
        max_backtracks = 0
        accept_ratio = 0
        policy = simple_policy

        # run policy for T timesteps
        # This might involve interacting with the environment using the policy and collecting trajectories
        trajectories = sample_trajectory


        # Estimate advantage function at all timesteps
        advantages = ta.compute_advantages(trajectories, gamma)


        # Compute policy gradient g
        loss = compute_policy_loss(policy, trajectories, advantages)
        loss.backward()
        optimizer.step()

        
        # use CG(with Hessian vector products) to compute F^-1(g)
        gradient = get_flat_grads(policy.parameters())
        conjugate_gradient = compute_conjugate_gradient(gradient)


        # Do line search on surrogate loss and KL constraint
        step_dir = conjugate_gradient / damping
        expected_improvement = gradient.dot(step_dir)
        status, updated_params = linesearch(policy, step_dir, expected_improvement)

        # Update the policy parameters based on the line search result
        if status:
            set_flat_params(policy.parameters(), updated_params)



# function for creating a trajectory from prestate,action and state
def create_traj(prestate,action,state):

    a = []
    a.append(prestate)
    a.append(action)
    a.append(state)

    return a 


# 




















