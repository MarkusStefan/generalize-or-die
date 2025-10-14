from logging import warning
import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym
    warning("gymnasium not found, falling back to gym. Some environments may not be available.")


class Environment:
    def __init__(self, env_name, seed=None, variation='default'):
        self.env_name = env_name
        self.variation = variation
        self.seed = seed
        self.env = self._create_env()
        self.obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()
        self.action_bound = self._get_action_bound()

    def _create_env(self):
        env = gym.make(self.env_name, render_mode=None)
        if self.variation == 'high_gravity' and 'Pendulum' in self.env_name:
            if hasattr(env.unwrapped, 'g'):
                env.unwrapped.g = 15.0
        elif self.variation == 'low_friction' and 'CartPole' in self.env_name:
            if hasattr(env.unwrapped, 'force_mag'):
                env.unwrapped.force_mag *= 0.8
        if self.seed is not None:
            env.reset(seed=self.seed)
        return env

    def _get_obs_dim(self):
        if hasattr(self.env.observation_space, 'shape'):
            return int(np.prod(self.env.observation_space.shape))
        return self.env.observation_space.n

    def _get_action_dim(self):
        if hasattr(self.env.action_space, 'shape'):
            return int(np.prod(self.env.action_space.shape))
        return self.env.action_space.n

    def _get_action_bound(self):
        if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
            return max(abs(self.env.action_space.low[0]), abs(self.env.action_space.high[0]))
        return 1.0

    def reset(self):
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        return np.array(obs).flatten()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        return np.array(obs).flatten(), reward, done, info

    def render(self):
        return self.env.render()


def create_environment(env_name='InvertedPendulum-v5', seed=999, variation='default'):
    return Environment(env_name, seed, variation)