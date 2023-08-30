import numpy as np
from inference import *
import torch as tt
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm



class bremen:
    def __init__(self):
        # Define hyperparameters
        self.K = 3
        self.T = 10
        self.I = 5
        self.B = 100
        self.tgp = None

    # Randomly initialize the target policy πθ
    def initialize_target_policy(self,in_size,out_size,in2,out2):

        self.tgp = inference.simple_nn(in_size,out_size)
        self.bcm = inference.behavior_nn(in2,out2)

        return self.tgp, self.bcm
        
    # create an ensemble of dynamic models
    def dynamics_model(self,model_name,inp,outp):

        if model_name == 'nn':
            
            dy = []
            for i in range(self.K):
                mod = inference.dynamic_nn(inp,outp)
                dy.append(mod)
        
        return dy


    # Training dynamics models ˆfφ using Dall
    def train_dynamics_models(self,x_train,y_train,dy):
        
        for i in range(len(dy)):

            model = dy[i]

            # train each model using backprop and mse loss
            learning_rate = 0.001
            num_epochs = 100
            # Define the loss function
            criterion = nn.MSELoss()
            # Define the optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            for epoch in range(num_epochs):
                # Forward pass
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print loss
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f"Model {i} trained.")
        print("model training completed.")
        

    # Train estimated behavior policy ˆπβ using D
    def train_behavior_policy(self, bcm, d):
        # Training logic for behavior policy
        model = bcm

        # train each model using backprop and mse loss
        learning_rate = 0.001
        num_epochs = 100
        # Define the loss function
        criterion = nn.MSELoss()
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f"Behavior Model trained.")
        

    # re initialize the target policy
    def reinitialize_tp(self,bcm,tgp):
    
        # Get the original weights of 'bcm'
        original_weights = []
        for param in bcm.parameters():
            original_weights.append(param.data.clone())

        # Create a new set of weights 'w2' centered around the original weights
        w2 = []
        for param in original_weights:
            sampled_weights = param + torch.randn_like(param)
            w2.append(sampled_weights)

        # Assign the sampled weights 'w2' to the 'tgp' neural network
        with tt.no_grad():
            idx = 0
            for param in tgp.parameters():
                param.copy_(w2[idx])
                idx += 1

        # Print the original and sampled weights for demonstration
        print("Original Weights:")
        for param in original_weights:
            print(param)

        print("\nSampled Weights (w2):")
        for param in w2:
            print(param)

        return tgp

    def combine(self,data,actions):

        data.append(actions)

        return data

    # Generate imaginary rollout
    def generate_imaginary_rollout(self,bcm,data,tgp):
        # Generate rollout logic
        actions = tgp.forward(data)
        rollout = bcm.forward(self.combine(data,actions))
        return rollout

    # Function to calculate the KL divergence
    def compute_kl_divergence(self,p, q):
        return np.sum(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

    # Function to compute advantages (just random values for demonstration)
    def compute_advantages(self,q_value,value):
        advantage = q_value - value
        return advantage

    # Optimize target policy πθ
    def optimize_target_policy(self,rollout,action_prob1,action_prob2,state,param1,param_k,
                               q_value,value):

        delta = 0.4
        actions1 = action_prob1/np.sum(action_prob1)
        actions2 = action_prob2/np.sum(action_prob2)
        states = state


        # Optimization logic for target policy
        # Initialize policy parameters and distribution
        theta = param1
        theta_k = param_k
        
        # Collect data using the current policy theta_k
        advantages = self.compute_advantages(q_value,value)
        
        # Calculate policy probabilities
        policy_theta = norm.pdf(actions, loc=0, scale=1)
        policy_theta_k = norm.pdf(actions, loc=0, scale=1)
        
        # Calculate surrogate loss
        surrogate_loss = np.mean(policy_theta/policy_theta_k * advantages)
        
        # Update theta using gradient ascent on surrogate loss
        gradient = np.mean((advantages / policy_theta_k)[:, np.newaxis] * states, axis=0)
        theta += 0.01 * gradient
        
        # Check if kl constraint is satisfied
        kl_divergence = self.compute_kl_divergence(policy_theta, policy_theta_k)
        if kl_divergence > delta:

            # Use line search to find new theta_k that satisfies constraint
            for _ in range(10):
                new_theta_k = theta_k + 0.01 * gradient
                new_policy_theta_k = norm.pdf(actions, loc=np.dot(states, new_theta_k), scale=1)
                new_kl_divergence = self.compute_kl_divergence(policy_theta, new_policy_theta_k)
                
                if new_kl_divergence <= delta:
                    theta_k = new_theta_k
                    break
        
        

        
        return 0

    # Main algorithm loop
    def bremen_algorithm(self,pre_state,no_of_agents):

        

        # initialise target policy, behavior cloned policy
        tgp, bcm = self.initialize_target_policy()

        inferrer = inf_process()
        agent_states = []
        for agent_no in range(no_of_agents):
            
            agent_states.append(inferrer.get_inf_features(pre_state,agent_no))
            
            # collect B experience samples --> B timesteps in hfo -->load dataset for this
            dall = pd.read_csv('data/dall.csv')
            d = pd.read_csv('data/d.csv')
            
            # Train dynamics models
            self.train_dynamics_models(dall)

            # Train estimated behavior policy
            self.train_behavior_policy(d)

            # Re-initialize target policy πθ0 = Normal(ˆπβ , 1)
            tgp = self.reinitialize_tp(bcm,tgp)

            # Generate imaginary rollout
            for k in range(self.T):
                
                rollout = self.generate_imaginary_rollout()

                # Optimize target policy πθ
                self.optimize_target_policy(rollout)


