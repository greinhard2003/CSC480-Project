import gym
import numpy as np

# Source - https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/atari_wrappers.py
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)

        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        done = None
        for i in range(self._skip):
            out = self.env.step(action)
            if len(out) == 4:
                obs, reward, done, info = out
                terminated = done
                truncated = False
            elif len(out) == 5:
                obs, reward, t, tr, info = out
                terminated = t
                truncated = tr
            else:
                raise ValueError(f"Unexpected number of outputs from env.step: {len(out)}")


            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)