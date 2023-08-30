import torch
import torch.nn as nn
import numpy as np
import logging  # Use Python's built-in logging module

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(
            self,
            output_size,
            n_layers=2,
            size=1000,
            activation=nn.ReLU(),
            output_activation=None):
        super(MLP, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        
        # Create a list to hold the layers of the MLP
        layers = []
        in_features = size  # Initial input size
        
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, size))
            layers.append(activation)
            in_features = size  # Update input size for next layer
        
        # Add the output layer
        layers.append(nn.Linear(size, output_size, bias=False))
        
        if output_activation:
            layers.append(output_activation)
        
        # Create the sequential model using the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Dynamics:
    def __init__(self):
        self.int_rewards_only = False
        self.ext_rewards_only = False

    def use_intrinsic_rewards_only(self):
        logger.info("Pre-training enabled. Using only intrinsic reward.")
        self.int_rewards_only = True

    def combine_int_and_ext_rewards(self):
        logger.info("Using a combination of external and intrinsic reward.")
        self.int_rewards_only = False
        self.ext_rewards_only = False

    def use_external_rewards_only(self):
        logger.info("Using external reward only.")
        self.int_rewards_only = False
        self.ext_rewards_only = True

    def information_gain(self, obses, acts, next_obses):
        return np.zeros([len(obses),])

    def information_gain_torch(self, obs, act, next_obs):
        raise NotImplementedError

    def process_rewards(self, ext_rewards, obses, actions, next_obses):
        if self.ext_rewards_only:
            return ext_rewards
        else:
            weighted_intrinsic_reward = self.information_gain(obses, actions, next_obses)
            if self.int_rewards_only:
                return weighted_intrinsic_reward
            else:
                return ext_rewards + weighted_intrinsic_reward

class DynamicsModel(Dynamics):
    def __init__(self, env, normalization, batch_size, epochs, val, device):
        super().__init__()
        self.env = env
        self.normalization = normalization
        self.batch_size = batch_size
        self.epochs = epochs
        self.val = val
        self.device = device

        self.obs_dim = env.observation_space.shape[0]
        self.acts_dim = env.action_space.shape[0]
        self.mlp = None

        self.epsilon = 1e-10

    def get_obs_dim(self):
        raise NotImplementedError

    def update_randomness(self):
        pass

    def update_normalization(self, new_normalization):
        self.normalization = new_normalization

    def _build_placeholders(self):
        self.obs_ph = torch.FloatTensor()
        self.acts_ph = torch.FloatTensor()
        self.next_obs_ph = torch.FloatTensor()

    def _get_feed_dict(self, obs, action, next_obs):
        feed_dict = {
            self.obs_ph: torch.FloatTensor(obs),
            self.acts_ph: torch.FloatTensor(action),
            self.next_obs_ph: torch.FloatTensor(next_obs)
        }
        return feed_dict

    def _get_normalized_obs_and_acts(self, obs, acts):
        normalized_obs = (obs[:, :self.obs_dim] - self.normalization.mean_obs) / (self.normalization.std_obs + self.epsilon)
        normalized_obs = torch.cat([normalized_obs, obs[:, self.obs_dim:]], dim=1)
        normalized_acts = (acts - self.normalization.mean_acts) / (self.normalization.std_acts + self.epsilon)
        return torch.cat([normalized_obs, normalized_acts], dim=1)

    def _get_predicted_normalized_deltas(self, states, actions):
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(states, actions)
        predicted_normalized_deltas = self.mlp(normalized_obs_and_acts)
        return predicted_normalized_deltas

    def _get_unnormalized_deltas(self, normalized_deltas):
        return normalized_deltas * self.normalization.std_deltas + self.normalization.mean_deltas

    def _add_observations_to_unnormalized_deltas(self, states, unnormalized_deltas):
        return states[:, :self.obs_dim] + unnormalized_deltas

    def _get_normalized_deltas(self, deltas):
        return (deltas - self.normalization.mean_deltas) / (self.normalization.std_deltas + self.epsilon)

class NNDynamicsModel(DynamicsModel):
    def __init__(
            self,
            env,
            n_layers,
            size,
            activation,
            output_activation,
            normalization,
            batch_size,
            epochs,
            learning_rate,
            val,
            device,
            reg_coeff=None,
            controller=None):
        super().__init__(env, normalization, batch_size, epochs, val, device)
        self.device = device
        self.controller = controller
        if reg_coeff is None:
            self.reg_coeff = 1.0
        else:
            self.reg_coeff = reg_coeff

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)

        self._build_placeholders()

        # Build NN.
        self.mlp = MLP(
            output_size=self.obs_dim,
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation)

        # Move the model to the specified device
        self.mlp.to(device)

        # Build cost function and optimizer.
        mse_loss, l2_loss, self.predicted_unnormalized_deltas = self._get_loss()

        self.loss = mse_loss + l2_loss * self.reg_coeff
        self.loss_val = mse_loss
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)

    def fit(self, train_data, val_data):
        self.optimizer.zero_grad()

        self.coeff = torch.tensor(1. / len(train_data), requires_grad=False, device=self.device)
        
        loss = 1000
        best_index = 0
        for epoch in range(self.epochs):
            for (itr, (obs, action, next_obs, _)) in enumerate(train_data):
                obs = torch.FloatTensor(obs)
                action = torch.FloatTensor(action)
                next_obs = torch.FloatTensor(next_obs)
                
                feed_dict = self._get_feed_dict(obs, action, next_obs)
                predicted_unnormalized_deltas = self.mlp(self._get_normalized_obs_and_acts(obs, action))
                
                mse_loss = nn.functional.mse_loss(
                    predicted_unnormalized_deltas,
                    self._get_normalized_deltas(next_obs - obs))
                l2_loss = torch.sum(torch.stack([torch.sum(param ** 2) for param in self.mlp.parameters()]))
                
                self.loss = mse_loss + l2_loss * self.coeff * self.reg_coeff
                self.loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if epoch % 5 == 0:
                loss_list = []
                with torch.no_grad():
                    for (itr, (obs, action, next_obs, _)) in enumerate(val_data):
                        obs = torch.FloatTensor(obs)
                        action = torch.FloatTensor(action)
                        next_obs = torch.FloatTensor(next_obs)
                        
                        cur_loss = nn.functional.mse_loss(
                            self.mlp(self._get_normalized_obs_and_acts(obs, action)),
                            self._get_normalized_deltas(next_obs - obs))
                        
                        loss_list.append(cur_loss.item())
                        
                logger.info("Validation loss = {}".format(np.mean(loss_list)))
                if np.mean(loss_list) < loss:
                    loss = np.mean(loss_list)
                    best_index = epoch

                if self.val:
                    if epoch - best_index >= 20:
                        break

    def predict(self, states, actions):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        
        unnormalized_deltas = self.mlp(self._get_normalized_obs_and_acts(states, actions)).cpu().numpy()
        return np.array(states)[:, :self.obs_dim] + self._get_unnormalized_deltas(unnormalized_deltas)

    def predict_torch(self, states, actions):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        
        predicted_normalized_deltas = self._get_predicted_normalized_deltas(states, actions)
        return self._add_observations_to_unnormalized_deltas(
            states, self._get_unnormalized_deltas(predicted_normalized_deltas))

    def _get_loss(self):
        deltas = self.next_obs_ph - self.obs_ph
        labels = self._get_normalized_deltas(deltas)
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(self.obs_ph, self.acts_ph)
        predicted_normalized_deltas = self.mlp(normalized_obs_and_acts)
        
        mse_loss = nn.functional.mse_loss(labels, predicted_normalized_deltas)
        l2_loss = torch.sum(torch.stack([torch.sum(param ** 2) for param in self.mlp.parameters()]))
        
        return mse_loss, l2_loss, self._get_unnormalized_deltas(predicted_normalized_deltas)