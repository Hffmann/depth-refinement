import gym
import numpy as np
from gym import spaces

class DepthRefinementEnv(gym.Env):
    def __init__(self, dataset, zerodepth, window_size=(256, 256)):
        super().__init__()
        self.dataset = dataset
        self.zerodepth = zerodepth
        self.window_size = window_size

        # Action: Residual depth adjustment
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

        # Observation: RGB + Initial depth + Position
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                          shape=(6, *window_size))

    def reset(self):
        self.rgb, self.lidar = self.dataset.sample()
        self.initial_depth = self.zerodepth.predict(self.rgb)
        self.current_depth = self.initial_depth.copy()
        return self._get_obs()

    def step(self, action):
        # Apply refinement
        self.current_depth += action
        reward = self._calculate_reward()
        done = False
        return self._get_obs(), reward, done, {}

    def _calculate_reward(self):
        # Chamfer distance between refined depth and LiDAR
        refined_3d = self._depth_to_3d(self.current_depth)
        lidar_3d = self._depth_to_3d(self.lidar)
        return -self._chamfer_distance(refined_3d, lidar_3d)
