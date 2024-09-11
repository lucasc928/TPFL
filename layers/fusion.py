import numpy as np
import gym
import torch
from stable_baselines3 import PPO


class ReputationGroupingEnv(gym.Env):
    def __init__(self, reputation_matrix):
        self.reputation_matrix = reputation_matrix.cpu().numpy()
        self.threshold = 0.5
        self.high_reputation_matrix = np.empty_like(self.reputation_matrix)
        self.low_reputation_matrix = np.empty_like(self.reputation_matrix)
        self.reward = 0
        self.low_reputation_count = 0
        self.high_reputation_count = 0
        self.high_reputation_limit = 64
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=reputation_matrix.shape)

    def reset(self):
        self.high_reputation_matrix = np.zeros_like(self.reputation_matrix)
        self.low_reputation_matrix = np.zeros_like(self.reputation_matrix)
        self.reward = 0
        self.low_reputation_count = 0
        self.high_reputation_count = 0
        return self.reputation_matrix

    def step(self, action):

        if action == 0:  # Select low reputation group
            mask = self.reputation_matrix < self.threshold
            self.low_reputation_matrix[mask] = self.reputation_matrix[mask]
            self.reward -= 1
            self.low_reputation_count += 1
            self.high_reputation_count = 0
            if self.low_reputation_count == 5:
                self.reward -= 10
                done = True
            else:
                done = False
        else:  # Select high reputation group
            mask = self.reputation_matrix >= self.threshold
            self.high_reputation_matrix[mask] = self.reputation_matrix[mask]
            self.reward += 1
            self.high_reputation_count += 1
            self.low_reputation_count = 0
            if self.high_reputation_count >= self.high_reputation_limit:
                self.reward += 20
                done = True
            else:
                done = False

        # Compute the observation, which is the updated reputation matrix
        observation = np.maximum(self.high_reputation_matrix, self.low_reputation_matrix)

        return observation, self.reward, done, {}


def fusion_update(X, Rep):
    new_X = torch.zeros_like(X)
    # Loop through each timestamp T in Rep
    for t in range(Rep.shape[2]):  # 11
        # Convert Rep at timestamp T to (N, C) feature matrix
        vehicle_dict = {}
        for n in range(Rep.shape[0]):
            vehicle_id = Rep[n, 0, t, 0].item()
            vehicle_dict[vehicle_id] = X[n, :, t, :]
        reputation_matrix = Rep[:, :, t, 0]
        # Use PPO algorithm to classify reputation matrix
        env = ReputationGroupingEnv(reputation_matrix)
        model = PPO("MlpPolicy", env=env, policy_kwargs=dict(net_arch=[64, 64]),
                    learning_rate=0.00025, n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    vf_coef=0.5, max_grad_norm=0.5, target_kl=0.01, verbose=1)
        model.learn(total_timesteps=10000)
        # Retrieve the high and low reputation group matrices from the last training
        classified_matrix = env.high_reputation_matrix
        high_reputation_group = np.zeros(X.shape)
        low_reputation_group = np.zeros(X.shape)
        for i in range(classified_matrix.shape[0]):  # 64
            vehicle_id = classified_matrix[i, 0]  # numpy.float32
            reputation = classified_matrix[i, 1]
            if vehicle_id in vehicle_dict:
                x_tensor = vehicle_dict[vehicle_id].cpu().detach().numpy()
                if reputation != 0:
                    high_reputation_group[i, :, t, :] = x_tensor
                else:
                    low_reputation_group[i, :, t, :] = x_tensor

        new_X = (0.88 * high_reputation_group) + (0.12 * low_reputation_group)

    X = torch.tensor(new_X, dtype=torch.float32, device="cuda:0")

    return X
